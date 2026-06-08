/**
 * byte_bpe.cpp  — Part 2: Byte-Level BPE Tokenizer
 * ─────────────────────────────────────────────────
 * Key differences from char-level BPE (Part 1):
 *   • Every byte (0x00-0xFF) is a valid initial symbol → OOV impossible
 *   • Non-ASCII bytes encoded as <0xNN> literals before merging
 *   • Noise injection: spelling mistakes, OCR distortions, transliteration
 *   • Robustness metrics: token drift, length increase, perplexity delta
 *
 * Build:  g++ -std=c++17 -O2 -o byte_bpe byte_bpe.cpp
 * Run:    ./byte_bpe ../../../data/corpus.txt
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <functional>

// ─── Type aliases ─────────────────────────────────────────────────────────────
using Sym      = std::string;
using SymPair  = std::pair<Sym,Sym>;
using Tokens   = std::vector<Sym>;

struct PairHash {
    size_t operator()(const SymPair& p) const {
        return std::hash<std::string>{}(p.first)
             ^ (std::hash<std::string>{}(p.second) << 32);
    }
};
using PairFreq = std::unordered_map<SymPair,int,PairHash>;

// ─── Byte encoding ────────────────────────────────────────────────────────────
// Convert a raw byte to its symbolic form (printable ASCII kept as-is)
static std::string byte_sym(unsigned char b) {
    if (b >= 32 && b < 127) return std::string(1, static_cast<char>(b));
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    return buf;
}

// ─── Tokenise raw string into byte symbols + </w> ────────────────────────────
static Tokens word_to_bytes(const std::string& word) {
    Tokens t;
    for (unsigned char b : word) t.push_back(byte_sym(b));
    t.push_back("</w>");
    return t;
}

// ─── Noise injection engines ──────────────────────────────────────────────────
struct NoiseEngine {
    std::mt19937 rng;

    explicit NoiseEngine(unsigned seed = 42) : rng(seed) {}

    // 1. Spelling mistake: swap adjacent chars, delete, insert, substitute
    std::string spelling_noise(const std::string& w, double rate = 0.15) {
        if (w.size() < 2) return w;
        std::string s = w;
        std::uniform_real_distribution<> prob(0.0, 1.0);
        std::uniform_int_distribution<> pos(0, static_cast<int>(s.size())-1);
        static const std::string alpha = "abcdefghijklmnopqrstuvwxyz";
        std::uniform_int_distribution<> ac(0, 25);

        int n_ops = std::max(1, static_cast<int>(w.size() * rate));
        for (int i = 0; i < n_ops && s.size() >= 2; ++i) {
            double r = prob(rng);
            int p = pos(rng) % static_cast<int>(s.size());
            if (r < 0.25 && p+1 < static_cast<int>(s.size())) {
                std::swap(s[p], s[p+1]);                 // swap
            } else if (r < 0.50) {
                s[p] = alpha[ac(rng)];                   // substitute
            } else if (r < 0.75 && s.size() > 1) {
                s.erase(p, 1);                           // delete
            } else {
                s.insert(p, 1, alpha[ac(rng)]);          // insert
            }
        }
        return s;
    }

    // 2. OCR distortion: visually similar character substitutions
    std::string ocr_noise(const std::string& w, double rate = 0.12) {
        static const std::unordered_map<char,std::string> ocr_map = {
            {'l',"1"},{'I',"l"},{'O',"0"},{'0',"O"},
            {'a',"@"},{'e',"3"},{'o',"0"},{'s',"5"},
            {'t',"7"},{'g',"9"},{'i',"!"},{' ',"_"}
        };
        std::uniform_real_distribution<> prob(0.0,1.0);
        std::string out;
        for (char c : w) {
            if (prob(rng) < rate) {
                auto it = ocr_map.find(c);
                out += (it != ocr_map.end()) ? it->second : std::string(1,c);
            } else {
                out += c;
            }
        }
        return out;
    }

    // 3. Transliteration noise: mix Latin/phonetic approximations
    std::string transliteration_noise(const std::string& w, double rate = 0.10) {
        // Simulates Hinglish/phonetic spelling common in Indian code-mixed text
        static const std::vector<std::pair<std::string,std::string>> rules = {
            {"th","t"},{"ph","f"},{"ck","k"},{"qu","kw"},
            {"tion","shun"},{"ous","us"},{"ght","t"},
            {"wh","w"},{"ee","i"},{"oo","u"},{"ay","ei"},
            {"ow","au"},{"ch","c"},{"sh","s"},{"ea","e"}
        };
        std::uniform_real_distribution<> prob(0.0,1.0);
        std::string s = w;
        for (auto& [from, to] : rules) {
            if (prob(rng) < rate) {
                size_t pos;
                while ((pos = s.find(from)) != std::string::npos)
                    s.replace(pos, from.size(), to);
            }
        }
        return s;
    }
};

// ─── Byte-Level BPE Tokenizer ─────────────────────────────────────────────────
class ByteBPE {
public:
    explicit ByteBPE(int target_vocab) : target_(target_vocab) {}

    void train(const std::string& corpus_path) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Word frequencies
        std::ifstream f(corpus_path);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::string w;
            while (ss >> w) {
                // lowercase, strip punctuation
                std::string clean;
                for (char c : w)
                    if (std::isalpha(c)) clean += std::tolower(c);
                if (!clean.empty()) word_freq_[clean]++;
            }
        }

        // 2. Initialise: each word → byte symbols
        // Byte-level: initial vocab = all 256 bytes + </w>
        for (int b = 0; b < 256; ++b) token_vocab_.insert(byte_sym(b));
        token_vocab_.insert("</w>");

        for (auto& [w, _] : word_freq_)
            word_tokens_[w] = word_to_bytes(w);

        // 3. Merge loop
        int current = static_cast<int>(token_vocab_.size());
        int n_merges = target_ - current;
        std::cerr << "[ByteBPE] Initial vocab=" << current
                  << "  target=" << target_
                  << "  merges=" << n_merges << "\n";

        for (int step = 0; step < n_merges; ++step) {
            PairFreq pf = get_pair_freqs();
            if (pf.empty()) break;

            SymPair best; int best_f = -1;
            for (auto& [p,fr] : pf)
                if (fr > best_f || (fr == best_f && p < best))
                    { best_f = fr; best = p; }

            Sym merged = best.first + best.second;
            merges_.push_back({best, merged});
            token_vocab_.insert(merged);
            apply_merge(best, merged);

            if (step % 500 == 0)
                std::cerr << "[ByteBPE] step=" << step << "/" << n_merges
                          << "  \"" << best.first << "\"+\""
                          << best.second << "\"  freq=" << best_f << "\n";
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        train_ms_ = std::chrono::duration<double,std::milli>(t1-t0).count();
        std::cerr << "[ByteBPE] Done in " << train_ms_ << "ms  vocab="
                  << token_vocab_.size() << "\n";
    }

    // Encode a word (string of bytes) → tokens
    Tokens encode_word(const std::string& word) const {
        Tokens toks = word_to_bytes(word);
        for (auto& [pair, merged] : merges_) {
            Tokens updated;
            for (size_t i = 0; i < toks.size(); ) {
                if (i+1 < toks.size() &&
                    toks[i]==pair.first && toks[i+1]==pair.second)
                    { updated.push_back(merged); i+=2; }
                else { updated.push_back(toks[i]); ++i; }
            }
            toks = std::move(updated);
        }
        return toks;
    }

    // Encode full text
    std::vector<Sym> encode(const std::string& text) const {
        std::vector<Sym> out;
        std::istringstream ss(text);
        std::string w;
        while (ss >> w) {
            std::string clean;
            for (char c : w) if (std::isalpha(c)) clean += std::tolower(c);
            if (clean.empty()) continue;
            auto t = encode_word(clean);
            out.insert(out.end(), t.begin(), t.end());
        }
        return out;
    }

    // Save vocab
    void save_vocab(const std::string& path) const {
        std::ofstream f(path);
        int id = 0;
        for (auto& sym : token_vocab_) f << id++ << "\t" << sym << "\n";
    }

    // Save merges
    void save_merges(const std::string& path) const {
        std::ofstream f(path);
        f << "#byte-bpe-merges\n";
        for (auto& [p, m] : merges_)
            f << p.first << " " << p.second << "\n";
    }

    int vocab_size() const { return static_cast<int>(token_vocab_.size()); }
    double train_ms() const { return train_ms_; }
    const std::unordered_map<std::string,int>& word_freq() const { return word_freq_; }

private:
    int target_;
    std::set<Sym> token_vocab_;
    std::vector<std::pair<SymPair,Sym>> merges_;
    std::unordered_map<std::string,Tokens> word_tokens_;
    std::unordered_map<std::string,int>    word_freq_;
    double train_ms_ = 0.0;

    PairFreq get_pair_freqs() const {
        PairFreq pf;
        for (auto& [w, toks] : word_tokens_) {
            int fr = word_freq_.at(w);
            for (size_t i = 0; i+1 < toks.size(); ++i)
                pf[{toks[i],toks[i+1]}] += fr;
        }
        return pf;
    }
    void apply_merge(const SymPair& pair, const Sym& merged) {
        for (auto& [w, toks] : word_tokens_) {
            Tokens updated;
            for (size_t i = 0; i < toks.size(); ) {
                if (i+1 < toks.size() &&
                    toks[i]==pair.first && toks[i+1]==pair.second)
                    { updated.push_back(merged); i+=2; }
                else { updated.push_back(toks[i]); ++i; }
            }
            toks = std::move(updated);
        }
    }
};

// ─── Robustness Evaluator ─────────────────────────────────────────────────────
struct RobustnessResult {
    std::string noise_type;
    double noise_rate;
    double token_drift;          // fraction of tokens that changed
    double avg_len_increase;     // avg extra tokens per word under noise
    double perplexity_delta;     // proxy: log(len_noisy/len_clean)
    int    vocab_inflation;      // new unique tokens introduced by noise
    std::vector<std::pair<std::string,std::string>> examples; // clean→noisy
};

static RobustnessResult evaluate_robustness(
    const ByteBPE& tok,
    const std::vector<std::string>& words,
    std::function<std::string(const std::string&)> noise_fn,
    const std::string& noise_type, double rate)
{
    RobustnessResult r;
    r.noise_type = noise_type;
    r.noise_rate = rate;

    int total_clean = 0, total_noisy = 0, drifted = 0;
    std::set<Sym> new_tokens;
    std::set<Sym> clean_set;

    for (auto& w : words) {
        auto ct = tok.encode_word(w);
        std::string nw = noise_fn(w);
        auto nt = tok.encode_word(nw);

        for (auto& s : ct) clean_set.insert(s);
        for (auto& s : nt) {
            if (clean_set.find(s) == clean_set.end()) new_tokens.insert(s);
        }

        total_clean += static_cast<int>(ct.size());
        total_noisy += static_cast<int>(nt.size());

        // Token drift: are token sequences different?
        if (ct != nt) ++drifted;

        if (r.examples.size() < 5) {
            std::string ct_str, nt_str;
            for (auto& s : ct) ct_str += "["+s+"]";
            for (auto& s : nt) nt_str += "["+s+"]";
            r.examples.push_back({w+"→"+ct_str, nw+"→"+nt_str});
        }
    }

    int n = static_cast<int>(words.size());
    r.token_drift       = n > 0 ? static_cast<double>(drifted)/n : 0.0;
    r.avg_len_increase  = n > 0 ? static_cast<double>(total_noisy-total_clean)/n : 0.0;
    r.perplexity_delta  = total_clean > 0
                        ? std::log(static_cast<double>(total_noisy)/total_clean)
                        : 0.0;
    r.vocab_inflation   = static_cast<int>(new_tokens.size());
    return r;
}

// ─── Pseudo-perplexity via unigram log-prob ───────────────────────────────────
static double compute_perplexity(
    const ByteBPE& tok,
    const std::vector<std::string>& words,
    const std::unordered_map<Sym,int>& token_counts,
    int total_tokens)
{
    double log_prob = 0.0;
    int n_toks = 0;
    for (auto& w : words) {
        for (auto& sym : tok.encode_word(w)) {
            auto it = token_counts.find(sym);
            double p = it != token_counts.end()
                     ? static_cast<double>(it->second) / total_tokens
                     : 1e-9;
            log_prob += std::log(p);
            ++n_toks;
        }
    }
    return n_toks > 0 ? std::exp(-log_prob / n_toks) : 1e9;
}

// ─── JSON output helper ───────────────────────────────────────────────────────
static void write_robustness_json(
    const std::vector<RobustnessResult>& results,
    const std::string& path)
{
    std::ofstream f(path);
    f << "[\n";
    for (size_t i = 0; i < results.size(); ++i) {
        auto& r = results[i];
        f << "  {\n"
          << "    \"noise_type\": \"" << r.noise_type << "\",\n"
          << "    \"noise_rate\": " << r.noise_rate << ",\n"
          << "    \"token_drift\": " << std::fixed << std::setprecision(4)
          << r.token_drift << ",\n"
          << "    \"avg_len_increase\": " << r.avg_len_increase << ",\n"
          << "    \"perplexity_delta\": " << r.perplexity_delta << ",\n"
          << "    \"vocab_inflation\": " << r.vocab_inflation << ",\n"
          << "    \"examples\": [\n";
        for (size_t e = 0; e < r.examples.size(); ++e) {
            // escape quotes
            auto esc = [](const std::string& s) {
                std::string o;
                for (char c : s) { if (c=='"') o+="\\\""; else o+=c; }
                return o;
            };
            f << "      {\"clean\": \"" << esc(r.examples[e].first)
              << "\", \"noisy\": \"" << esc(r.examples[e].second) << "\"}";
            if (e+1 < r.examples.size()) f << ",";
            f << "\n";
        }
        f << "    ]\n  }";
        if (i+1 < results.size()) f << ",";
        f << "\n";
    }
    f << "]\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string corpus = argc > 1 ? argv[1] : "../../../data/corpus.txt";
    std::string outdir = argc > 2 ? argv[2] : "../output";

    // Train byte-level BPE (vocab=512 for demo)
    ByteBPE tok(512);
    tok.train(corpus);
    tok.save_vocab(outdir + "/byte_bpe_vocab.txt");
    tok.save_merges(outdir + "/byte_bpe_merges.txt");

    // Build token frequency table for perplexity
    std::unordered_map<Sym,int> tok_counts;
    int total_toks = 0;
    for (auto& [w, fr] : tok.word_freq()) {
        for (auto& sym : tok.encode_word(w)) {
            tok_counts[sym] += fr; total_toks += fr;
        }
    }

    // Collect test words
    std::vector<std::string> test_words;
    for (auto& [w, _] : tok.word_freq()) test_words.push_back(w);
    if (test_words.size() > 200) test_words.resize(200);

    NoiseEngine ne(42);

    // Noise rates to sweep
    std::vector<double> rates = {0.05, 0.10, 0.15, 0.20, 0.25};
    std::vector<RobustnessResult> all_results;

    for (double rate : rates) {
        // Spelling
        all_results.push_back(evaluate_robustness(tok, test_words,
            [&](const std::string& w){ return ne.spelling_noise(w, rate); },
            "spelling", rate));
        // OCR
        all_results.push_back(evaluate_robustness(tok, test_words,
            [&](const std::string& w){ return ne.ocr_noise(w, rate); },
            "ocr", rate));
        // Transliteration
        all_results.push_back(evaluate_robustness(tok, test_words,
            [&](const std::string& w){ return ne.transliteration_noise(w, rate); },
            "transliteration", rate));
    }

    write_robustness_json(all_results, outdir + "/robustness.json");

    // Console report
    std::cout << "\n══ BYTE-LEVEL BPE ROBUSTNESS REPORT ══\n";
    std::cout << "Vocab size: " << tok.vocab_size()
              << "  Train time: " << tok.train_ms() << "ms\n\n";

    std::cout << std::left
              << std::setw(18) << "Noise Type"
              << std::setw(8)  << "Rate"
              << std::setw(14) << "Token Drift"
              << std::setw(16) << "Len Increase"
              << std::setw(16) << "Perplx Delta"
              << std::setw(16) << "Vocab Inflat."
              << "\n" << std::string(88,'-') << "\n";

    for (auto& r : all_results) {
        std::cout << std::setw(18) << r.noise_type
                  << std::setw(8)  << r.noise_rate
                  << std::setw(14) << std::fixed << std::setprecision(4) << r.token_drift
                  << std::setw(16) << r.avg_len_increase
                  << std::setw(16) << r.perplexity_delta
                  << std::setw(16) << r.vocab_inflation
                  << "\n";
    }

    // Clean perplexity baseline
    double ppl_clean = compute_perplexity(tok, test_words, tok_counts, total_toks);
    std::cout << "\nBaseline perplexity (clean): " << ppl_clean << "\n";

    // Segmentation under noise examples
    std::cout << "\n── Segmentation Examples Under Noise ──\n";
    std::vector<std::string> demo = {"tokenization","learning","processing","classification"};
    for (auto& w : demo) {
        auto clean_t = tok.encode_word(w);
        auto spell_t = tok.encode_word(ne.spelling_noise(w, 0.15));
        auto ocr_t   = tok.encode_word(ne.ocr_noise(w, 0.12));

        auto fmt = [](const Tokens& t) {
            std::string s;
            for (auto& x : t) s += "["+x+"]";
            return s;
        };
        std::cout << "  " << w << "\n"
                  << "    clean : " << fmt(clean_t) << " (" << clean_t.size() << ")\n"
                  << "    spell : " << fmt(spell_t) << " (" << spell_t.size() << ")\n"
                  << "    ocr   : " << fmt(ocr_t)   << " (" << ocr_t.size()   << ")\n\n";
    }

    std::cout << "\n[Done] JSON written to " << outdir << "/robustness.json\n";
    return 0;
}