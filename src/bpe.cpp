#include "bpe.h"
#include <cctype>
#include <regex>

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
BPETokenizer::BPETokenizer(int target_vocab_size)
    : target_vocab_size_(target_vocab_size) {}

// ─────────────────────────────────────────────────────────────────────────────
// Tokenize text into words (whitespace + lowercase + punctuation separation)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::string> BPETokenizer::split_text(const std::string& text) const {
    std::vector<std::string> words;
    std::string cur;
    for (char c : text) {
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '\'') {
            cur += std::tolower(static_cast<unsigned char>(c));
        } else {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build word frequency table from corpus file
// ─────────────────────────────────────────────────────────────────────────────
Vocab BPETokenizer::build_word_freq(const std::string& corpus_path) const {
    std::ifstream f(corpus_path);
    if (!f) throw std::runtime_error("Cannot open corpus: " + corpus_path);
    Vocab freq;
    std::string line;
    while (std::getline(f, line)) {
        for (auto& w : split_text(line))
            freq[w]++;
    }
    return freq;
}

// ─────────────────────────────────────────────────────────────────────────────
// Count adjacent symbol pair frequencies across all words
// ─────────────────────────────────────────────────────────────────────────────
PairFreq BPETokenizer::get_pair_freqs() const {
    PairFreq pf;
    for (auto& [word, tokens] : word_to_tokens_) {
        int wf = word_freq_.at(word);
        for (size_t i = 0; i + 1 < tokens.size(); ++i)
            pf[{tokens[i], tokens[i+1]}] += wf;
    }
    return pf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Apply a merge rule to all word token sequences (used during initialization)
// ─────────────────────────────────────────────────────────────────────────────
void BPETokenizer::apply_merge(const SymbolPair& pair, const Symbol& merged) {
    for (auto& [word, tokens] : word_to_tokens_) {
        WordTokens updated;
        updated.reserve(tokens.size());
        size_t i = 0;
        while (i < tokens.size()) {
            if (i + 1 < tokens.size() &&
                tokens[i] == pair.first && tokens[i+1] == pair.second) {
                updated.push_back(merged);
                i += 2;
            } else {
                updated.push_back(tokens[i]);
                ++i;
            }
        }
        tokens = std::move(updated);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Apply merge and incrementally update pair frequency cache
// ─────────────────────────────────────────────────────────────────────────────
void BPETokenizer::apply_merge_and_update_pairs(const SymbolPair& pair, const Symbol& merged) {
    // First, invalidate affected pairs in cache
    std::set<SymbolPair> affected_pairs;
    
    for (auto& [word, tokens] : word_to_tokens_) {
        int wf = word_freq_.at(word);
        
        // Find all occurrences of the pair and mark affected positions
        size_t i = 0;
        while (i < tokens.size()) {
            if (i + 1 < tokens.size() &&
                tokens[i] == pair.first && tokens[i+1] == pair.second) {
                // Mark pairs that will be affected by this merge
                if (i > 0) affected_pairs.insert({tokens[i-1], pair.first});
                if (i + 2 < tokens.size()) affected_pairs.insert({pair.second, tokens[i+2]});
                i += 2;
            } else {
                ++i;
            }
        }
    }

    // Remove affected pairs from cache
    for (auto& ap : affected_pairs) {
        cached_pair_freqs_.erase(ap);
        cached_pair_freqs_.erase({ap.second, ap.first});
    }
    
    // Remove the pair itself
    cached_pair_freqs_.erase(pair);
    
    // Now apply the actual merge
    apply_merge(pair, merged);
}

// ─────────────────────────────────────────────────────────────────────────────
// Get pair frequencies incrementally (only recompute affected pairs)
// ─────────────────────────────────────────────────────────────────────────────
PairFreq BPETokenizer::get_pair_freqs_incremental(const SymbolPair& merged_pair) {
    // Incrementally update based on the new merged token
    const Symbol& merged = merge_symbol(merged_pair);
    
    // Add new pairs involving the merged token
    for (auto& [word, tokens] : word_to_tokens_) {
        int wf = word_freq_.at(word);
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            SymbolPair p = {tokens[i], tokens[i+1]};
            cached_pair_freqs_[p] += wf;
        }
    }
    
    return cached_pair_freqs_;
}

// ─────────────────────────────────────────────────────────────────────────────
// TRAIN
// ─────────────────────────────────────────────────────────────────────────────
void BPETokenizer::train(const std::string& corpus_path) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Build word frequencies
    word_freq_ = build_word_freq(corpus_path);
    int unique_words = static_cast<int>(word_freq_.size());
    
    // 2. Initialise each word as sequence of characters + end-of-word marker </w>
    token_vocab_.clear();
    merges_.clear();
    word_to_tokens_.clear();
    cached_pair_freqs_.clear();

    std::set<std::string> char_set;
    for (auto& [word, freq] : word_freq_) {
        WordTokens toks;
        for (char c : word) {
            std::string s(1, c);
            toks.push_back(s);
            char_set.insert(s);
        }
        toks.push_back("</w>");
        char_set.insert("</w>");
        word_to_tokens_[word] = toks;
    }

    // Populate initial vocabulary with character tokens
    int id = 0;
    for (auto& s : char_set) token_vocab_[s] = id++;

    // 3. Cap target vocab size based on corpus realism
    //    For a corpus with N unique words, max sustainable vocab ≈ 10-15x N
    //    Practical limit: with 256KB corpus (~29K words), max is ~50-100K tokens
    int realistic_cap = std::max(4096, unique_words * 8);  // 8x multiplier
    int effective_vocab_size = std::min(target_vocab_size_, realistic_cap);
    int current_size = static_cast<int>(token_vocab_.size());
    int num_merges   = effective_vocab_size - current_size;

    std::cerr << "[BPE] Corpus: " << unique_words << " unique words\n";
    std::cerr << "[BPE] Initial vocab size: " << current_size
              << "  Requested target: " << target_vocab_size_
              << "  Realistic cap: " << realistic_cap
              << "  Effective target: " << effective_vocab_size << "\n";

    // Build initial pair frequencies only once
    std::cerr << "[BPE] Building initial pair frequencies...\n";
    for (auto& [word, tokens] : word_to_tokens_) {
        int wf = word_freq_[word];
        for (size_t i = 0; i + 1 < tokens.size(); ++i)
            cached_pair_freqs_[{tokens[i], tokens[i+1]}] += wf;
    }
    
    std::cerr << "[BPE] Initial pair count: " << cached_pair_freqs_.size() << "\n";

    int successful_merges = 0;
    int min_freq_for_merge = 2;
    int steps_without_good_merge = 0;

    for (int step = 0; step < num_merges; ++step) {
        if (cached_pair_freqs_.empty()) {
            std::cerr << "[BPE] No more pairs to merge at step " << step << "\n";
            break;
        }

        // Find the most frequent pair
        SymbolPair best;
        int best_freq = -1;
        for (auto& [pair, freq] : cached_pair_freqs_) {
            if (freq > best_freq || (freq == best_freq && pair < best)) {
                best_freq = freq;
                best      = pair;
            }
        }

        // Early stopping: if best pair has very low frequency, stop
        if (best_freq < min_freq_for_merge) {
            steps_without_good_merge++;
            if (steps_without_good_merge > 10) {
                std::cerr << "[BPE] Stopping: no pairs with freq >= " << min_freq_for_merge 
                          << " (best=" << best_freq << ") after 10 steps\n";
                break;
            }
        } else {
            steps_without_good_merge = 0;
        }

        Symbol merged = merge_symbol(best);
        merges_.push_back({best, merged});
        token_vocab_[merged] = id++;
        successful_merges++;

        // Apply merge
        apply_merge(best, merged);

        // Remove old pair and its inverse
        cached_pair_freqs_.erase(best);
        cached_pair_freqs_.erase({best.second, best.first});

        // Only recompute pairs that involve the newly created merged token
        // This is much faster than a full recomputation
        try {
            for (auto& [word, tokens] : word_to_tokens_) {
                int wf = word_freq_[word];
                for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                    if (tokens[i] == merged || tokens[i+1] == merged) {
                        cached_pair_freqs_[{tokens[i], tokens[i+1]}] += wf;
                    }
                }
            }
        } catch (...) {
            std::cerr << "[BPE] Error updating pairs at step " << step << "\n";
            break;
        }

        if (step % 500 == 0)
            std::cerr << "[BPE] Step " << step << "  freq=" << best_freq 
                      << "  vocab=" << token_vocab_.size() << "\n";
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    train_time_ms_ = std::chrono::duration<double,std::milli>(t1 - t0).count();

    std::cerr << "[BPE] Training done: " << successful_merges << " merges in " 
              << train_time_ms_ << "ms.  Final vocab: " << token_vocab_.size() << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Encode a single word using trained merge rules
// ─────────────────────────────────────────────────────────────────────────────
WordTokens BPETokenizer::encode_word(const std::string& word) const {
    // Initialise as characters + </w>
    WordTokens tokens;
    for (char c : word) tokens.push_back(std::string(1, c));
    tokens.push_back("</w>");

    // Apply merges in order
    for (auto& [pair, merged] : merges_) {
        WordTokens updated;
        updated.reserve(tokens.size());
        size_t i = 0;
        while (i < tokens.size()) {
            if (i + 1 < tokens.size() &&
                tokens[i] == pair.first && tokens[i+1] == pair.second) {
                updated.push_back(merged);
                i += 2;
            } else {
                updated.push_back(tokens[i]);
                ++i;
            }
        }
        tokens = std::move(updated);
    }
    return tokens;
}

// ─────────────────────────────────────────────────────────────────────────────
// Encode a full sentence
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::string> BPETokenizer::encode(const std::string& text) const {
    std::vector<std::string> result;
    for (auto& word : split_text(text)) {
        auto toks = encode_word(word);
        result.insert(result.end(), toks.begin(), toks.end());
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Save vocabulary
// ─────────────────────────────────────────────────────────────────────────────
void BPETokenizer::save_vocab(const std::string& path) const {
    std::ofstream f(path);
    // Sort by id for readability
    std::vector<std::pair<Symbol,int>> items(token_vocab_.begin(), token_vocab_.end());
    std::sort(items.begin(), items.end(),
              [](auto& a, auto& b){ return a.second < b.second; });
    for (auto& [sym, id] : items)
        f << id << "\t" << sym << "\n";
    std::cerr << "[BPE] Vocab saved to " << path << " (" << items.size() << " entries)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Save merge rules
// ─────────────────────────────────────────────────────────────────────────────
void BPETokenizer::save_merges(const std::string& path) const {
    std::ofstream f(path);
    f << "#version: 0.2\n";
    for (auto& [pair, merged] : merges_)
        f << pair.first << " " << pair.second << "\n";
    std::cerr << "[BPE] Merges saved to " << path << " (" << merges_.size() << " rules)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Load vocabulary
// ─────────────────────────────────────────────────────────────────────────────
void BPETokenizer::load_vocab(const std::string& path) {
    std::ifstream f(path);
    token_vocab_.clear();
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        int id; std::string sym;
        if (ss >> id >> sym) token_vocab_[sym] = id;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Load merge rules
// ─────────────────────────────────────────────────────────────────────────────
void BPETokenizer::load_merges(const std::string& path) {
    std::ifstream f(path);
    merges_.clear();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string a, b;
        if (ss >> a >> b) {
            Symbol merged = a + b;
            merges_.push_back({{a, b}, merged});
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compute evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────
BPEMetrics BPETokenizer::compute_metrics(const std::string& corpus_path) const {
    BPEMetrics m;
    m.vocab_size    = static_cast<int>(token_vocab_.size());
    m.train_time_ms = train_time_ms_;

    // Memory: rough estimate = vocab entries * avg_key_len bytes
    size_t mem = 0;
    for (auto& [sym, id] : token_vocab_)
        mem += sym.size() + sizeof(int) + 32; // 32 bytes hash map overhead
    m.memory_bytes = mem;

    // Re-encode corpus
    std::ifstream f(corpus_path);
    std::string line;
    int total_words = 0, total_tokens = 0, oov_count = 0;

    // We need to know all training words to detect OOV
    // A word is OOV if its full-word token (word</w>) is NOT in vocab —
    // meaning it was never seen as a complete token. We detect it by checking
    // whether encode_word produces any single-char tokens (fragmented beyond
    // what BPE learned).
    std::unordered_map<std::string, int> training_words(word_freq_.begin(), word_freq_.end());

    while (std::getline(f, line)) {
        for (auto& word : split_text(line)) {
            ++total_words;
            auto toks = encode_word(word);
            total_tokens += static_cast<int>(toks.size());
            // OOV: word not seen during training
            if (training_words.find(word) == training_words.end())
                ++oov_count;
        }
    }

    m.total_words  = total_words;
    m.total_tokens = total_tokens;
    m.avg_tokens_per_word = total_words > 0
                          ? static_cast<double>(total_tokens) / total_words
                          : 0.0;
    m.oov_rate = total_words > 0
               ? static_cast<double>(oov_count) / total_words
               : 0.0;

    // Compression ratio: chars in corpus / tokens produced
    std::ifstream f2(corpus_path);
    std::string all; std::string tmp;
    while (std::getline(f2, tmp)) all += tmp + " ";
    m.compression_ratio = total_tokens > 0
                        ? static_cast<double>(all.size()) / total_tokens
                        : 0.0;

    // Top-20 merge labels by position (earlier = more frequent)
    int top_n = std::min(20, static_cast<int>(merges_.size()));
    for (int i = 0; i < top_n; ++i) {
        auto& [pair, merged] = merges_[i];
        std::string label = "\"" + pair.first + "\" + \"" + pair.second + "\" → \"" + merged + "\"";
        m.top_merges.push_back({label, top_n - i}); // proxy frequency = rank-based
    }

    return m;
}
