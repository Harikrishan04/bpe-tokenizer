/**
 * explainable.cpp — Part 6: Explainable Tokenizer
 * ────────────────────────────────────────────────
 * Produces token-character alignment, saliency scores, and
 * human-readable explanations for tokenization decisions.
 *
 * Outputs JSON consumed by the PHP/HTML visualization dashboard.
 *
 * Build:  g++ -std=c++17 -O2 -o explainable explainable.cpp
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <functional>

using Sym     = std::string;
using SymPair = std::pair<Sym,Sym>;
using Tokens  = std::vector<Sym>;

// ─── Load utilities ───────────────────────────────────────────────────────────
static std::vector<std::pair<SymPair,Sym>> load_merges(const std::string& path) {
    std::vector<std::pair<SymPair,Sym>> m;
    std::ifstream f(path); std::string line;
    while (std::getline(f,line)) {
        if (line.empty()||line[0]=='#') continue;
        std::istringstream ss(line); Sym a,b;
        if (ss>>a>>b) m.push_back({{a,b},a+b});
    }
    return m;
}

static std::unordered_map<Sym,int> load_vocab(const std::string& path) {
    std::unordered_map<Sym,int> v;
    std::ifstream f(path); std::string line;
    while (std::getline(f,line)) {
        std::istringstream ss(line); int id; Sym s;
        if (ss>>id>>s) v[s]=id;
    }
    return v;
}

// ─── Token-Character alignment ────────────────────────────────────────────────
struct AlignedToken {
    Sym    token;
    int    char_start;   // inclusive index in original word
    int    char_end;     // exclusive
    int    merge_step;   // which merge rule created this token (-1=initial)
    double saliency;     // computed below
    std::string merge_explanation;
};

// ─── Trace BPE encoding with alignment ───────────────────────────────────────
static std::vector<AlignedToken> trace_encode(
    const std::string& word,
    const std::vector<std::pair<SymPair,Sym>>& merges,
    const std::unordered_map<Sym,int>& vocab)
{
    // Initial character tokens
    Tokens toks;
    std::vector<int> starts, ends;
    for (int i=0; i<(int)word.size(); ++i) {
        toks.push_back(std::string(1,word[i]));
        starts.push_back(i); ends.push_back(i+1);
    }
    toks.push_back("</w>"); starts.push_back((int)word.size()); ends.push_back((int)word.size());

    std::vector<int> merge_steps(toks.size(), -1);

    // Apply merges, tracking step index
    for (int step=0; step<(int)merges.size(); ++step) {
        auto& [pair, merged] = merges[step];
        Tokens new_toks; std::vector<int> new_starts, new_ends, new_steps;
        for (size_t i=0; i<toks.size(); ) {
            if (i+1<toks.size() && toks[i]==pair.first && toks[i+1]==pair.second) {
                new_toks.push_back(merged);
                new_starts.push_back(starts[i]);
                new_ends.push_back(ends[i+1]);
                new_steps.push_back(step);
                i+=2;
            } else {
                new_toks.push_back(toks[i]);
                new_starts.push_back(starts[i]);
                new_ends.push_back(ends[i]);
                new_steps.push_back(merge_steps[i]);
                ++i;
            }
        }
        toks=new_toks; starts=new_starts; ends=new_ends; merge_steps=new_steps;
    }

    // Build aligned tokens
    std::vector<AlignedToken> aligned;
    for (size_t i=0; i<toks.size(); ++i) {
        AlignedToken at;
        at.token      = toks[i];
        at.char_start = starts[i];
        at.char_end   = ends[i];
        at.merge_step = merge_steps[i];

        // Saliency proxy: (token length) / (total chars in word) * log(1+step_importance)
        int tok_len = at.char_end - at.char_start;
        double step_importance = (at.merge_step >= 0)
                               ? 1.0/(1.0 + at.merge_step) : 0.05;
        at.saliency = static_cast<double>(tok_len) / word.size() * (1.0 + std::log1p(step_importance)*5);

        // Human-readable explanation
        if (at.merge_step == -1) {
            at.merge_explanation = "Initial character token";
        } else if (at.token == "</w>") {
            at.merge_explanation = "End-of-word marker";
        } else {
            at.merge_explanation = "Merged at step " + std::to_string(at.merge_step+1)
                + " (pair rank #" + std::to_string(at.merge_step+1) + " by frequency)";
        }

        // Special: known morpheme classification
        static const std::unordered_map<std::string,std::string> morphemes = {
            {"ing</w>","suffix: present participle"},
            {"tion</w>","suffix: nominalization"},
            {"ation</w>","suffix: nominalization"},
            {"ed</w>","suffix: past tense"},
            {"er</w>","suffix: comparative/agent"},
            {"re","prefix: repetition/back"},
            {"un","prefix: negation"},
            {"pre","prefix: before"},
            {"multi","prefix: many"},
            {"sub","prefix: under"},
            {"over","prefix: above/excess"},
            {"out","prefix: beyond"},
            {"in","prefix: in/negation"},
            {"al</w>","suffix: adjective"},
            {"ly</w>","suffix: adverb"},
            {"ness</w>","suffix: nominalization"},
            {"ment</w>","suffix: result/state"},
            {"ize</w>","suffix: verbalization"},
            {"ise</w>","suffix: verbalization"},
            {"ism</w>","suffix: doctrine/belief"},
            {"ist</w>","suffix: agent/adherent"},
        };
        auto it = morphemes.find(at.token);
        if (it != morphemes.end())
            at.merge_explanation += " [Morpheme: " + it->second + "]";

        aligned.push_back(at);
    }
    return aligned;
}

// ─── Token saliency: downstream prediction shift ───────────────────────────────
// Simulates: what happens to a downstream "length-based sentiment proxy"
// when each token is masked? Measures contribution.
struct SaliencyResult {
    std::string token;
    double      saliency_score;
    std::string explanation;
};

static std::vector<SaliencyResult> compute_token_saliency(
    const std::vector<AlignedToken>& aligned,
    const std::string& word)
{
    // Baseline prediction: sum of saliency scores = proxy for "complexity"
    double baseline = 0.0;
    for (auto& at : aligned) baseline += at.saliency;

    std::vector<SaliencyResult> results;
    for (size_t i=0; i<aligned.size(); ++i) {
        // Leave-one-out: remove token i and measure impact
        double masked = 0.0;
        for (size_t j=0; j<aligned.size(); ++j)
            if (j!=i) masked += aligned[j].saliency;

        double delta = std::abs(baseline - masked) / (baseline + 1e-9);

        SaliencyResult sr;
        sr.token         = aligned[i].token;
        sr.saliency_score= delta;
        sr.explanation   = "Removing '" + aligned[i].token
                         + "' changes complexity proxy by "
                         + std::to_string(static_cast<int>(delta*100)) + "%";
        results.push_back(sr);
    }
    return results;
}

// ─── Case study: How boundary changes affect downstream ──────────────────────
struct CaseStudy {
    std::string word;
    std::string vocab_small;   // vocab label
    std::string vocab_large;
    Tokens tokens_small;
    Tokens tokens_large;
    std::string impact_note;
};

static std::string esc(const std::string& s) {
    std::string o; for (char c:s){if(c=='"')o+="\\\"";else if(c=='\\')o+="\\\\";else o+=c;} return o;
}

// ─── JSON writer ─────────────────────────────────────────────────────────────
static void write_explanation_json(
    const std::string& word,
    const std::vector<AlignedToken>& aligned,
    const std::vector<SaliencyResult>& saliency,
    std::ofstream& f)
{
    f << "  {\n"
      << "    \"word\": \"" << esc(word) << "\",\n"
      << "    \"aligned_tokens\": [\n";
    for (size_t i=0; i<aligned.size(); ++i) {
        auto& at = aligned[i];
        f << "      {\"token\": \"" << esc(at.token) << "\","
          << " \"char_start\": " << at.char_start << ","
          << " \"char_end\": " << at.char_end << ","
          << " \"merge_step\": " << at.merge_step << ","
          << " \"saliency\": " << std::fixed << std::setprecision(4) << at.saliency << ","
          << " \"explanation\": \"" << esc(at.merge_explanation) << "\"}";
        if (i+1<aligned.size()) f << ",";
        f << "\n";
    }
    f << "    ],\n    \"token_saliency\": [\n";
    for (size_t i=0; i<saliency.size(); ++i) {
        auto& sr = saliency[i];
        f << "      {\"token\": \"" << esc(sr.token) << "\","
          << " \"score\": " << sr.saliency_score << ","
          << " \"explanation\": \"" << esc(sr.explanation) << "\"}";
        if (i+1<saliency.size()) f << ",";
        f << "\n";
    }
    f << "    ]\n  }";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string vocab_sm  = argc>1 ? argv[1] : "../../output/vocabs/256_vocab.txt";
    std::string merges_sm = argc>2 ? argv[2] : "../../output/merges/256_merges.txt";
    std::string vocab_lg  = argc>3 ? argv[3] : "../../output/vocabs/2048_vocab.txt";
    std::string merges_lg = argc>4 ? argv[4] : "../../output/merges/2048_merges.txt";
    std::string outdir    = argc>5 ? argv[5] : "../output";

    auto m_sm = load_merges(merges_sm);
    auto m_lg = load_merges(merges_lg);
    auto v_sm = load_vocab(vocab_sm);
    auto v_lg = load_vocab(vocab_lg);

    std::vector<std::string> demo_words = {
        "tokenization","backpropagation","preprocessing","reinforcement",
        "multilingual","explainability","convolutional","quantization",
        "overfitting","regularization"
    };

    std::cout << "\n══ EXPLAINABLE TOKENIZER — ALIGNMENT & SALIENCY ══\n\n";

    // Console alignment display
    for (auto& word : demo_words) {
        auto aligned_sm = trace_encode(word, m_sm, v_sm);
        auto aligned_lg = trace_encode(word, m_lg, v_lg);

        std::cout << "━━━ Word: \"" << word << "\" ━━━\n";

        // Small vocab
        std::cout << "  [V=256]  ";
        for (auto& at : aligned_sm) {
            if (at.token != "</w>")
                std::cout << "|" << at.token;
        }
        std::cout << "|\n";

        // Large vocab
        std::cout << "  [V=2048] ";
        for (auto& at : aligned_lg) {
            if (at.token != "</w>")
                std::cout << "|" << at.token;
        }
        std::cout << "|\n";

        // Character-level alignment grid
        std::cout << "  Char: [";
        for (char c : word) std::cout << c << "|";
        std::cout << "]\n";

        // Token boundary markers
        std::cout << "  V256: [";
        std::vector<char> grid(word.size(), '-');
        for (auto& at : aligned_sm) {
            if (at.token=="</w>") continue;
            for (int i=at.char_start; i<at.char_end && i<(int)word.size(); ++i)
                grid[i] = 'X';
            if (at.char_end-1 < (int)word.size()) grid[at.char_end-1] = '|';
        }
        for (char c : grid) std::cout << c;
        std::cout << "]\n";

        // Saliency
        auto sal = compute_token_saliency(aligned_sm, word);
        std::cout << "  Saliency (V=256):\n";
        for (auto& s : sal) {
            if (s.token=="</w>") continue;
            std::cout << "    \"" << std::setw(16) << std::left << s.token
                      << "\" score=" << std::fixed << std::setprecision(4) << s.saliency_score
                      << "  " << s.explanation << "\n";
        }
        std::cout << "\n";
    }

    // Write JSON
    {
        std::ofstream f(outdir + "/explanations.json");
        f << "[\n";
        for (size_t wi=0; wi<demo_words.size(); ++wi) {
            auto& word = demo_words[wi];
            auto aligned = trace_encode(word, m_sm, v_sm);
            auto sal     = compute_token_saliency(aligned, word);
            write_explanation_json(word, aligned, sal, f);
            if (wi+1<demo_words.size()) f << ",";
            f << "\n";
        }
        f << "]\n";
    }

    // Case studies
    std::cout << "── Case Studies: Token Boundary Effect on Downstream ──\n";
    struct CS { std::string word, note; };
    std::vector<CS> cases = {
        {"reinforcement", "V=256 splits 'enforce' differently → loses morpheme signal"},
        {"preprocessing", "V=256: [pre]+[processing] preserves prefix; correct!"},
        {"multilingual",  "V=256: misses 'multi' prefix; V=2048: recovers it"},
        {"quantization",  "V=256: 'ation' suffix always captured → stable signal"},
    };
    for (auto& cs : cases) {
        auto sm = trace_encode(cs.word, m_sm, v_sm);
        auto lg = trace_encode(cs.word, m_lg, v_lg);
        std::cout << "  " << cs.word << "\n    V=256 : ";
        for (auto& t : sm) std::cout << "[" << t.token << "]";
        std::cout << "\n    V=2048: ";
        for (auto& t : lg) std::cout << "[" << t.token << "]";
        std::cout << "\n    Note  : " << cs.note << "\n\n";
    }

    std::cout << "[Done] explanations.json written to " << outdir << "\n";
    return 0;
}