/**
 * fairness.cpp — Part 3: Tokenizer Fairness & Bias Analysis
 * ──────────────────────────────────────────────────────────
 * Constructs labeled groups (English, Hindi-romanised, Tamil-romanised,
 * code-mixed, dialects) and measures:
 *   • OOV rate per group
 *   • Avg tokens per word per group
 *   • Segmentation consistency (same word → same split across runs)
 *   • Disparate Impact ratio (DI)
 *   • OOV Disparity Score (ODS)
 *   • Subgroup performance gap using a naive Bayes–style toxicity proxy
 *
 * Build:  g++ -std=c++17 -O2 -o fairness fairness.cpp
 * Run:    ./fairness [vocab_file] [merges_file] [output_dir]
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
#include <cmath>
#include <iomanip>

// ─── Minimal BPE loader (re-uses merge rules from Part 1) ─────────────────────
using Sym     = std::string;
using SymPair = std::pair<Sym,Sym>;
using Tokens  = std::vector<Sym>;

static std::vector<std::pair<SymPair,Sym>> load_merges(const std::string& path) {
    std::vector<std::pair<SymPair,Sym>> merges;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0]=='#') continue;
        std::istringstream ss(line);
        std::string a, b;
        if (ss >> a >> b) merges.push_back({{a,b}, a+b});
    }
    return merges;
}

static std::set<std::string> load_vocab_set(const std::string& path) {
    std::set<std::string> vs;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        int id; std::string sym;
        if (ss >> id >> sym) vs.insert(sym);
    }
    return vs;
}

static Tokens encode_word(
    const std::string& word,
    const std::vector<std::pair<SymPair,Sym>>& merges)
{
    Tokens t;
    for (char c : word) t.push_back(std::string(1,std::tolower(c)));
    t.push_back("</w>");
    for (auto& [pair, merged] : merges) {
        Tokens updated;
        for (size_t i = 0; i < t.size(); ) {
            if (i+1 < t.size() && t[i]==pair.first && t[i+1]==pair.second)
                { updated.push_back(merged); i+=2; }
            else { updated.push_back(t[i]); ++i; }
        }
        t = std::move(updated);
    }
    return t;
}

// ─── Linguistic Group Dataset ─────────────────────────────────────────────────
struct Example {
    std::string text;
    std::string label;   // "positive" / "negative" / "neutral" (sentiment)
};

struct LinguisticGroup {
    std::string name;
    std::string description;
    std::vector<Example> examples;
};

static std::vector<LinguisticGroup> build_dataset() {
    return {
        {
            "english_formal",
            "Standard formal English text",
            {
                {"the model processes natural language efficiently", "positive"},
                {"tokenization splits text into subword units effectively", "positive"},
                {"the algorithm fails to handle edge cases properly", "negative"},
                {"performance degrades significantly on noisy input data", "negative"},
                {"the system produces reasonable output for most inputs", "neutral"},
                {"vocabulary size affects tokenization quality and speed", "neutral"},
                {"deep learning models require large amounts of training data", "neutral"},
                {"attention mechanisms improve sequence-to-sequence translation", "positive"},
                {"the baseline model underperforms on low-resource languages", "negative"},
                {"preprocessing steps normalize text before tokenization begins", "neutral"},
            }
        },
        {
            "english_informal",
            "Informal/dialect English (AAVE, internet slang)",
            {
                {"gonna tokenize this rn its lowkey fire", "positive"},
                {"ngl this model bussin fr fr no cap", "positive"},
                {"bruh the output is mid and kinda trash tbh", "negative"},
                {"aint no way this thing actually works lmao", "negative"},
                {"its giving main character energy with the predictions", "positive"},
                {"slay the tokenizer ate that whole sentence periodt", "positive"},
                {"lowkey the oov rate be wildin out here", "negative"},
                {"yall this preprocessing step hitting different fr", "positive"},
                {"the vocab size is sus and kinda slept on", "neutral"},
                {"bestie the model said nah and dipped out", "negative"},
            }
        },
        {
            "hindi_romanized",
            "Hindi written in Roman script (Hinglish)",
            {
                {"yeh model bahut achha kaam karta hai", "positive"},
                {"tokenization bahut important process hai NLP mein", "positive"},
                {"yeh system bilkul kaam nahi karta properly", "negative"},
                {"performance bahut kharab hai is model ki", "negative"},
                {"thoda theek hai but improvements chahiye isme", "neutral"},
                {"vocabulary bahut chota hai is tokenizer ka", "negative"},
                {"deep learning mein bohot data chahiye training ke liye", "neutral"},
                {"attention mechanism ne bahut improvement diya hai", "positive"},
                {"low resource languages ke liye problem hoti hai", "negative"},
                {"preprocessing se text normalize hota hai tokenization se pehle", "neutral"},
            }
        },
        {
            "tamil_romanized",
            "Tamil written in Roman script (Tanglish)",
            {
                {"ippa model romba nalla work panrathu", "positive"},
                {"tokenization process text ah subword units la split pannum", "positive"},
                {"system seri illa edge cases handle pannala", "negative"},
                {"performance romba kurainjuchu noisy input la", "negative"},
                {"output reasonable ah iruku most inputs kaga", "neutral"},
                {"vocab size tokenization quality ah affect pannum", "neutral"},
                {"deep learning ku romba training data venum", "neutral"},
                {"attention mechanism translation improve pannuchu", "positive"},
                {"low resource languages la performance down aagum", "negative"},
                {"preprocessing text normalize pannum tokenization ku munna", "neutral"},
            }
        },
        {
            "code_mixed",
            "English-Hindi code-mixed text",
            {
                {"yeh model is very efficient at processing karta hai", "positive"},
                {"tokenization splits text into subword units bahut well", "positive"},
                {"this system fails to handle edge cases properly matlab", "negative"},
                {"performance degrades bahut on noisy input data yaar", "negative"},
                {"the system thoda okay produces reasonable output", "neutral"},
                {"vocabulary size bahut affects tokenization quality na", "neutral"},
                {"deep learning models require large amounts of data chahiye", "neutral"},
                {"attention mechanism ne improve kiya sequence translation", "positive"},
                {"baseline model underperforms low-resource languages mein", "negative"},
                {"preprocessing normalizes text before tokenization shuru hoti", "neutral"},
            }
        },
        {
            "technical_jargon",
            "NLP/ML technical terminology",
            {
                {"hyperparameter optimization improves downstream task performance", "positive"},
                {"byte-pair-encoding vocabulary compression reduces OOV rate", "positive"},
                {"vanishing gradient descent destabilizes recurrent network training", "negative"},
                {"suboptimal tokenization increases perplexity by significant margin", "negative"},
                {"wordpiece tokenizer uses likelihood-based merge criterion", "neutral"},
                {"sentencepiece implements both BPE and unigram language model", "neutral"},
                {"transformer self-attention quadratically scales with sequence length", "neutral"},
                {"BLEU score correlates with human judgments on translation quality", "positive"},
                {"catastrophic forgetting degrades pretrained model fine-tuning", "negative"},
                {"contrastive pretraining improves zero-shot cross-lingual transfer", "positive"},
            }
        }
    };
}

// ─── Metrics per group ────────────────────────────────────────────────────────
struct GroupMetrics {
    std::string group_name;
    int    n_words       = 0;
    int    n_oov         = 0;
    double oov_rate      = 0.0;
    double avg_tok_word  = 0.0;
    double avg_seg_len   = 0.0;   // avg chars per token
    int    total_tokens  = 0;
    // Downstream task: sentiment accuracy (simulated)
    double sentiment_acc = 0.0;
    // Consistency: re-tokenizing same word always gives same result (always 1 for deterministic BPE)
    double consistency   = 1.0;
};

// ─── Naive Bayes Sentiment Classifier (proxy for downstream bias) ─────────────
struct NaiveBayes {
    std::unordered_map<std::string, std::unordered_map<Sym,int>> class_tok_counts;
    std::unordered_map<std::string,int> class_counts;
    int vocab_size = 0;
    std::set<Sym> seen_toks;

    void train(const std::vector<LinguisticGroup>& groups,
               const std::vector<std::pair<SymPair,Sym>>& merges,
               const std::string& train_group = "english_formal") {
        for (auto& g : groups) {
            if (g.name != train_group) continue;
            for (auto& ex : g.examples) {
                class_counts[ex.label]++;
                std::istringstream ss(ex.text);
                std::string w;
                while (ss >> w) {
                    auto toks = encode_word(w, merges);
                    for (auto& t : toks) {
                        class_tok_counts[ex.label][t]++;
                        seen_toks.insert(t);
                    }
                }
            }
        }
        vocab_size = static_cast<int>(seen_toks.size());
    }

    std::string predict(const std::string& text,
                        const std::vector<std::pair<SymPair,Sym>>& merges) const {
        std::unordered_map<std::string,double> log_probs;
        int total = 0;
        for (auto& [c,cnt] : class_counts) total += cnt;

        for (auto& [c, cnt] : class_counts) {
            log_probs[c] = std::log(static_cast<double>(cnt)/total);
            int class_total = 0;
            if (class_tok_counts.count(c))
                for (auto& [t,f] : class_tok_counts.at(c)) class_total += f;

            std::istringstream ss(text);
            std::string w;
            while (ss >> w) {
                for (auto& tok : encode_word(w, merges)) {
                    int freq = 0;
                    if (class_tok_counts.count(c)) {
                        auto it = class_tok_counts.at(c).find(tok);
                        if (it != class_tok_counts.at(c).end()) freq = it->second;
                    }
                    // Laplace smoothed
                    double p = static_cast<double>(freq+1) / (class_total+vocab_size+1);
                    log_probs[c] += std::log(p);
                }
            }
        }
        return std::max_element(log_probs.begin(), log_probs.end(),
            [](auto& a, auto& b){ return a.second < b.second; })->first;
    }
};

// ─── Disparate Impact computation ─────────────────────────────────────────────
static double disparate_impact(double privileged_rate, double unprivileged_rate) {
    if (privileged_rate == 0) return 0.0;
    return unprivileged_rate / privileged_rate;
}

// ─── JSON writer ──────────────────────────────────────────────────────────────
static void write_fairness_json(
    const std::vector<GroupMetrics>& metrics,
    const std::string& path)
{
    std::ofstream f(path);
    f << std::fixed << std::setprecision(4);
    f << "{\n  \"groups\": [\n";
    for (size_t i=0; i<metrics.size(); ++i) {
        auto& m = metrics[i];
        f << "    {\n"
          << "      \"group\": \"" << m.group_name << "\",\n"
          << "      \"n_words\": " << m.n_words << ",\n"
          << "      \"n_oov\": " << m.n_oov << ",\n"
          << "      \"oov_rate\": " << m.oov_rate << ",\n"
          << "      \"avg_tokens_per_word\": " << m.avg_tok_word << ",\n"
          << "      \"total_tokens\": " << m.total_tokens << ",\n"
          << "      \"sentiment_accuracy\": " << m.sentiment_acc << ",\n"
          << "      \"consistency\": " << m.consistency << "\n"
          << "    }";
        if (i+1<metrics.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n}\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string vocab_file  = argc>1 ? argv[1] : "../../output/vocabs/1024_vocab.txt";
    std::string merges_file = argc>2 ? argv[2] : "../../output/merges/1024_merges.txt";
    std::string outdir      = argc>3 ? argv[3] : "../output";

    auto merges  = load_merges(merges_file);
    auto vocab   = load_vocab_set(vocab_file);

    auto groups  = build_dataset();

    // Train NB classifier on english_formal
    NaiveBayes nb;
    nb.train(groups, merges, "english_formal");

    std::vector<GroupMetrics> all_metrics;

    for (auto& g : groups) {
        GroupMetrics gm;
        gm.group_name = g.name;

        int correct = 0, total_ex = 0;
        int sum_toks = 0, sum_chars = 0, n_words = 0, n_oov = 0;

        for (auto& ex : g.examples) {
            // Sentiment prediction
            std::string pred = nb.predict(ex.text, merges);
            if (pred == ex.label) ++correct;
            ++total_ex;

            // Tokenize each word
            std::istringstream ss(ex.text);
            std::string w;
            while (ss >> w) {
                // strip non-alpha
                std::string clean;
                for (char c : w) if (std::isalpha(c)) clean += std::tolower(c);
                if (clean.empty()) continue;
                ++n_words;

                auto toks = encode_word(clean, merges);
                sum_toks += static_cast<int>(toks.size());
                sum_chars += static_cast<int>(clean.size());

                // OOV: word itself (word</w>) not in vocab
                std::string full = clean + "</w>";
                if (vocab.find(full) == vocab.end()) ++n_oov;
            }
        }

        gm.n_words      = n_words;
        gm.n_oov        = n_oov;
        gm.oov_rate     = n_words>0 ? static_cast<double>(n_oov)/n_words : 0.0;
        gm.avg_tok_word = n_words>0 ? static_cast<double>(sum_toks)/n_words : 0.0;
        gm.total_tokens = sum_toks;
        gm.avg_seg_len  = sum_toks>0 ? static_cast<double>(sum_chars)/sum_toks : 0.0;
        gm.sentiment_acc= total_ex>0 ? static_cast<double>(correct)/total_ex : 0.0;
        gm.consistency  = 1.0; // deterministic BPE

        all_metrics.push_back(gm);
    }

    // Print report
    std::cout << "\n══ FAIRNESS & BIAS ANALYSIS REPORT ══\n\n";
    std::cout << std::left
              << std::setw(22) << "Group"
              << std::setw(10) << "Words"
              << std::setw(10) << "OOV"
              << std::setw(12) << "OOV Rate"
              << std::setw(14) << "AvgTok/Word"
              << std::setw(14) << "Sent.Acc"
              << "\n" << std::string(82,'-') << "\n";

    for (auto& m : all_metrics) {
        std::cout << std::setw(22) << m.group_name
                  << std::setw(10) << m.n_words
                  << std::setw(10) << m.n_oov
                  << std::setw(12) << std::fixed << std::setprecision(4) << m.oov_rate
                  << std::setw(14) << m.avg_tok_word
                  << std::setw(14) << m.sentiment_acc
                  << "\n";
    }

    // Disparate impact
    double privileged_oov = all_metrics[0].oov_rate; // english_formal = privileged
    double privileged_acc = all_metrics[0].sentiment_acc;
    std::cout << "\n── Disparate Impact Analysis (base: english_formal) ──\n";
    std::cout << std::left
              << std::setw(22) << "Group"
              << std::setw(16) << "DI (OOV)"
              << std::setw(16) << "DI (Accuracy)"
              << std::setw(20) << "OOV Disparity Score"
              << "\n" << std::string(74,'-') << "\n";

    for (size_t i=1; i<all_metrics.size(); ++i) {
        auto& m = all_metrics[i];
        double di_oov = (privileged_oov>0)
                      ? m.oov_rate / privileged_oov : 99.9;
        double di_acc = (privileged_acc>0)
                      ? m.sentiment_acc / privileged_acc : 0.0;
        double ods    = std::abs(m.oov_rate - privileged_oov);
        std::cout << std::setw(22) << m.group_name
                  << std::setw(16) << std::fixed << std::setprecision(4) << di_oov
                  << std::setw(16) << di_acc
                  << std::setw(20) << ods
                  << "\n";
    }

    std::cout << "\n── Mitigation Strategies ──\n"
              << "  1. Train on multilingual corpus (DI threshold: OOV ratio < 1.25x)\n"
              << "  2. Use byte-level BPE (Part 2) — eliminates OOV disparity entirely\n"
              << "  3. Script-aware tokenization for Devanagari/Tamil (Part 4)\n"
              << "  4. Augment vocab with romanized Hindi/Tamil top-5000 words\n"
              << "  5. Use SentencePiece with language-ID prefix tokens\n\n";

    write_fairness_json(all_metrics, outdir + "/fairness_metrics.json");
    std::cout << "[Done] Written to " << outdir << "/fairness_metrics.json\n";
    return 0;
}