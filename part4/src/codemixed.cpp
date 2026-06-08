/**
 * codemixed.cpp — Part 4: Code-Mixed Tokenizer with Script-Aware Heuristics
 * ───────────────────────────────────────────────────────────────────────────
 * Extends BPE with:
 *   • Script detection: Latin, Devanagari (U+0900–U+097F),
 *     Tamil (U+0B80–U+0BFF), Bangla (U+0980–U+09FF)
 *   • Language-ID tagging: <EN> <HI> <TA> <BN> <MIX>
 *   • Script-boundary token insertion for code-switched text
 *   • Fragmentation rate and token efficiency metrics
 *   • Indian multilingual scenario analysis
 *
 * Build:  g++ -std=c++17 -O2 -o codemixed codemixed.cpp
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
#include <cassert>
#include <cstdint>
#include <locale>
#include <codecvt>

// ─── UTF-8 decode helpers ─────────────────────────────────────────────────────
static uint32_t utf8_codepoint(const std::string& s, size_t& pos) {
    unsigned char c = s[pos];
    uint32_t cp = 0;
    int bytes = 0;
    if      (c < 0x80) { cp = c; bytes = 1; }
    else if (c < 0xE0) { cp = c & 0x1F; bytes = 2; }
    else if (c < 0xF0) { cp = c & 0x0F; bytes = 3; }
    else               { cp = c & 0x07; bytes = 4; }
    for (int i = 1; i < bytes && pos+i < s.size(); ++i)
        cp = (cp << 6) | (static_cast<unsigned char>(s[pos+i]) & 0x3F);
    pos += bytes;
    return cp;
}

// ─── Script classifier ────────────────────────────────────────────────────────
enum class Script {
    LATIN, DEVANAGARI, TAMIL, BANGLA, ARABIC, UNKNOWN
};

static Script detect_script(uint32_t cp) {
    if (cp >= 0x0041 && cp <= 0x007A) return Script::LATIN;      // A-z
    if (cp >= 0x0900 && cp <= 0x097F) return Script::DEVANAGARI;
    if (cp >= 0x0B80 && cp <= 0x0BFF) return Script::TAMIL;
    if (cp >= 0x0980 && cp <= 0x09FF) return Script::BANGLA;
    if (cp >= 0x0600 && cp <= 0x06FF) return Script::ARABIC;
    return Script::UNKNOWN;
}

static std::string script_tag(Script s) {
    switch (s) {
        case Script::LATIN:      return "<EN>";
        case Script::DEVANAGARI: return "<HI>";
        case Script::TAMIL:      return "<TA>";
        case Script::BANGLA:     return "<BN>";
        case Script::ARABIC:     return "<AR>";
        default:                 return "<UNK>";
    }
}

static std::string script_name(Script s) {
    switch (s) {
        case Script::LATIN:      return "Latin";
        case Script::DEVANAGARI: return "Devanagari";
        case Script::TAMIL:      return "Tamil";
        case Script::BANGLA:     return "Bangla";
        case Script::ARABIC:     return "Arabic";
        default:                 return "Unknown";
    }
}

// ─── Segment text into script-homogeneous chunks ──────────────────────────────
struct Chunk {
    std::string text;
    Script script;
    bool is_boundary = false;  // code-switch boundary?
};

static std::vector<Chunk> segment_by_script(const std::string& text) {
    std::vector<Chunk> chunks;
    std::string cur;
    Script cur_script = Script::UNKNOWN;
    size_t pos = 0;

    while (pos < text.size()) {
        if (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n') {
            if (!cur.empty()) {
                bool boundary = !chunks.empty() && chunks.back().script != cur_script
                                && cur_script != Script::UNKNOWN;
                chunks.push_back({cur, cur_script, boundary});
                cur.clear();
            }
            ++pos;
            continue;
        }
        size_t p2 = pos;
        uint32_t cp = utf8_codepoint(text, p2);
        Script ws = detect_script(cp);
        if (ws == Script::UNKNOWN) ws = cur_script;

        if (!cur.empty() && ws != cur_script && cur_script != Script::UNKNOWN) {
            bool boundary = !chunks.empty() && chunks.back().script != cur_script;
            chunks.push_back({cur, cur_script, boundary});
            cur.clear();
        }
        cur_script = ws;
        cur += text.substr(pos, p2-pos);
        pos = p2;
    }
    if (!cur.empty()) {
        bool boundary = !chunks.empty() && chunks.back().script != cur_script;
        chunks.push_back({cur, cur_script, boundary});
    }
    return chunks;
}

// ─── BPE loader ──────────────────────────────────────────────────────────────
using Sym     = std::string;
using SymPair = std::pair<Sym,Sym>;
using Tokens  = std::vector<Sym>;

static std::vector<std::pair<SymPair,Sym>> load_merges(const std::string& path) {
    std::vector<std::pair<SymPair,Sym>> m;
    std::ifstream f(path); std::string line;
    while (std::getline(f,line)) {
        if (line.empty()||line[0]=='#') continue;
        std::istringstream ss(line); std::string a,b;
        if (ss>>a>>b) m.push_back({{a,b},a+b});
    }
    return m;
}

static Tokens bpe_encode(const std::string& word,
                          const std::vector<std::pair<SymPair,Sym>>& merges)
{
    if (word.empty()) return {};
    Tokens t;
    // Byte-aware: encode each UTF-8 character as symbol
    size_t pos=0;
    while (pos<word.size()) {
        size_t p2=pos;
        uint32_t cp = utf8_codepoint(word, p2);
        // For non-ASCII, use U+XXXX representation
        if (cp < 128) t.push_back(std::string(1,static_cast<char>(cp)));
        else {
            char buf[16]; std::snprintf(buf,sizeof(buf),"U+%04X",cp);
            t.push_back(buf);
        }
        pos=p2;
    }
    t.push_back("</w>");
    for (auto& [pair,merged]:merges) {
        Tokens u;
        for (size_t i=0;i<t.size();) {
            if (i+1<t.size()&&t[i]==pair.first&&t[i+1]==pair.second)
                {u.push_back(merged);i+=2;}
            else {u.push_back(t[i]);++i;}
        }
        t=std::move(u);
    }
    return t;
}

// ─── Code-Mixed Tokenizer ─────────────────────────────────────────────────────
class CodeMixedTokenizer {
public:
    explicit CodeMixedTokenizer(const std::string& merges_path)
        : merges_(load_merges(merges_path)) {}

    struct TokenResult {
        std::vector<std::string> tokens;
        std::vector<Script>      token_scripts;
        int  n_switches = 0;       // number of script switch events
        int  n_words    = 0;
        bool is_codemixed = false;
    };

    TokenResult tokenize(const std::string& text) const {
        TokenResult res;
        auto chunks = segment_by_script(text);

        Script prev_script = Script::UNKNOWN;
        for (auto& ch : chunks) {
            if (ch.script != prev_script && prev_script != Script::UNKNOWN) {
                ++res.n_switches;
                res.tokens.push_back("<SWITCH:" + script_name(ch.script) + ">");
                res.token_scripts.push_back(Script::UNKNOWN);
            }
            // Add script tag
            res.tokens.push_back(script_tag(ch.script));
            res.token_scripts.push_back(ch.script);

            // BPE encode the chunk
            auto bpe = bpe_encode(ch.text, merges_);
            for (auto& t : bpe) {
                res.tokens.push_back(t);
                res.token_scripts.push_back(ch.script);
            }
            prev_script = ch.script;
            ++res.n_words;
        }
        res.is_codemixed = res.n_switches > 0;
        return res;
    }

    // Fragmentation rate: avg tokens per character
    double fragmentation_rate(const std::string& text) const {
        auto r = tokenize(text);
        int chars = 0;
        for (char c : text) if (!std::isspace(c)) ++chars;
        return chars > 0 ? static_cast<double>(r.tokens.size())/chars : 0.0;
    }

    // Token efficiency: avg meaningful chars per token (excl. special tokens)
    double token_efficiency(const std::string& text) const {
        auto r = tokenize(text);
        int content_toks = 0, content_chars = 0;
        for (size_t i=0; i<r.tokens.size(); ++i) {
            auto& t = r.tokens[i];
            if (t[0]=='<') continue;  // skip special tokens
            ++content_toks;
            content_chars += static_cast<int>(t.size()) - (t.back()=='>' ? 0 : 0);
        }
        return content_toks > 0 ? static_cast<double>(content_chars)/content_toks : 0.0;
    }

private:
    std::vector<std::pair<SymPair,Sym>> merges_;
};

// ─── Test corpus (Indian multilingual scenarios) ──────────────────────────────
static std::vector<std::pair<std::string,std::string>> build_test_corpus() {
    return {
        // (text, description)
        {"the model processes text efficiently", "Pure English"},
        {"yeh model bahut achha hai processing ke liye", "Hinglish (romanized)"},
        {"ippa system romba nalla work panrathu", "Tanglish (Tamil romanized)"},
        {"this model bahut fast hai and accurate bhi", "Hindi-English code-mix"},
        {"mera model is very good at NLP tasks yaar", "Casual Hindi-English"},
        {"attention mechanism ne improve kiya translation quality ko", "Technical Hinglish"},
        {"tokenization process text ah split pannum into subwords", "Tamil-English mix"},
        {"deep learning models require bohot saara training data", "Hindi-English formal"},
        {"BLEU score acha hai but perplexity thodi zyada hai", "Technical code-mix"},
        {"embedding vectors capture semantic meaning bahut well", "Dense code-mix"},
        // Devanagari + Latin mix (simulated with placeholder)
        {"NLP mein tokenization ek important step hai preprocessing ka", "Heavy Hinglish"},
        {"yaar the OOV rate is too high for Hindi words na", "Complaint code-mix"},
    };
}

// ─── Metrics per scenario ─────────────────────────────────────────────────────
struct ScenarioMetrics {
    std::string description;
    int   n_tokens;
    int   n_switches;
    double fragmentation;
    double efficiency;
    bool  is_codemixed;
};

// ─── Write JSON ───────────────────────────────────────────────────────────────
static void write_json(const std::vector<ScenarioMetrics>& ms, const std::string& path) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(4);
    f << "[\n";
    for (size_t i=0; i<ms.size(); ++i) {
        auto& m = ms[i];
        f << "  {\n"
          << "    \"description\": \"" << m.description << "\",\n"
          << "    \"n_tokens\": " << m.n_tokens << ",\n"
          << "    \"n_switches\": " << m.n_switches << ",\n"
          << "    \"fragmentation_rate\": " << m.fragmentation << ",\n"
          << "    \"token_efficiency\": " << m.efficiency << ",\n"
          << "    \"is_codemixed\": " << (m.is_codemixed?"true":"false") << "\n"
          << "  }";
        if (i+1<ms.size()) f << ",";
        f << "\n";
    }
    f << "]\n";
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    std::string merges_path = argc>1 ? argv[1] : "../../output/merges/1024_merges.txt";
    std::string outdir      = argc>2 ? argv[2] : "../output";

    CodeMixedTokenizer tok(merges_path);
    auto corpus = build_test_corpus();

    std::cout << "\n══ CODE-MIXED TOKENIZER — EVALUATION ══\n\n";
    std::cout << std::left
              << std::setw(40) << "Description"
              << std::setw(8)  << "Tokens"
              << std::setw(10) << "Switches"
              << std::setw(14) << "Fragm.Rate"
              << std::setw(12) << "Efficiency"
              << "\n" << std::string(84,'-') << "\n";

    std::vector<ScenarioMetrics> all_m;

    for (auto& [text, desc] : corpus) {
        auto res = tok.tokenize(text);
        double fr  = tok.fragmentation_rate(text);
        double eff = tok.token_efficiency(text);

        std::cout << std::setw(40) << desc
                  << std::setw(8)  << res.tokens.size()
                  << std::setw(10) << res.n_switches
                  << std::setw(14) << std::fixed << std::setprecision(4) << fr
                  << std::setw(12) << eff << "\n";

        all_m.push_back({desc, (int)res.tokens.size(), res.n_switches, fr, eff, res.is_codemixed});
    }

    // Detailed example
    std::cout << "\n── Detailed Token Output ──\n";
    std::string demo = "this model bahut fast hai and accurate bhi";
    auto res = tok.tokenize(demo);
    std::cout << "Input: \"" << demo << "\"\n";
    std::cout << "Tokens: ";
    for (auto& t : res.tokens) std::cout << "[" << t << "]";
    std::cout << "\nScript switches: " << res.n_switches << "\n";

    // Best practices
    std::cout << "\n── Best Practices for Indian Multilingual Scenarios ──\n"
              << "  1. Use script-boundary tokens (<SWITCH:>) to signal language change.\n"
              << "  2. Train separate sub-vocabularies per script; merge at BPE init.\n"
              << "  3. For Devanagari: use character-level init with Hindi morphemes.\n"
              << "  4. For Tamil: agglutinative structure → smaller vocab sufficient.\n"
              << "  5. Accept higher fragmentation for low-resource scripts (OK <0.6).\n"
              << "  6. Use language-ID prefix tokens for downstream task disambiguation.\n"
              << "  7. Minimum 64K vocab recommended for 3-language code-mixed corpus.\n\n";

    write_json(all_m, outdir + "/codemixed_metrics.json");
    std::cout << "[Done] Written to " << outdir << "/codemixed_metrics.json\n";
    return 0;
}