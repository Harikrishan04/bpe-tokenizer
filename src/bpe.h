#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <utility>
#include <functional>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <memory>

// ─── Type aliases ─────────────────────────────────────────────────────────────
using Symbol      = std::string;
using SymbolPair  = std::pair<Symbol, Symbol>;
using WordTokens  = std::vector<Symbol>;
using Vocab       = std::unordered_map<std::string, int>;     // word → freq
using TokenVocab  = std::unordered_map<Symbol, int>;          // subword → id
using MergeRule   = std::pair<SymbolPair, Symbol>;            // (a,b) → ab
using MergeList   = std::vector<MergeRule>;

// ─── Pair hash for unordered_map ─────────────────────────────────────────────
struct PairHash {
    size_t operator()(const SymbolPair& p) const {
        size_t h1 = std::hash<std::string>{}(p.first);
        size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 << 32) ^ (h2 >> 32);
    }
};
using PairFreq = std::unordered_map<SymbolPair, int, PairHash>;

// ─── Metrics struct returned after training ───────────────────────────────────
struct BPEMetrics {
    int    vocab_size          = 0;
    double avg_tokens_per_word = 0.0;
    double oov_rate            = 0.0;
    double compression_ratio   = 0.0;
    double train_time_ms       = 0.0;
    size_t memory_bytes        = 0;
    int    total_tokens        = 0;
    int    total_words         = 0;
    std::vector<std::pair<std::string,int>> top_merges;  // merge label → freq
};

// ─── Main BPE class ───────────────────────────────────────────────────────────
class BPETokenizer {
public:
    explicit BPETokenizer(int target_vocab_size);

    // Training
    void train(const std::string& corpus_path);

    // Encode a single word into subword tokens
    WordTokens encode_word(const std::string& word) const;

    // Encode a full sentence
    std::vector<std::string> encode(const std::string& text) const;

    // Save / load vocabulary and merge rules
    void save_vocab(const std::string& path) const;
    void save_merges(const std::string& path) const;
    void load_vocab(const std::string& path);
    void load_merges(const std::string& path);

    // Metrics
    BPEMetrics compute_metrics(const std::string& corpus_path) const;

    // Accessors
    const TokenVocab& get_token_vocab() const { return token_vocab_; }
    const MergeList&  get_merges()      const { return merges_; }
    int               get_vocab_size()  const { return static_cast<int>(token_vocab_.size()); }

private:
    int target_vocab_size_;
    TokenVocab token_vocab_;   // subword → id
    MergeList  merges_;        // ordered merge rules

    // Internal training state
    std::unordered_map<std::string, WordTokens> word_to_tokens_; // word → token list
    std::unordered_map<std::string, int>        word_freq_;      // word → freq
    PairFreq   cached_pair_freqs_;  // cached pair frequencies for incremental updates

    // helpers
    Vocab        build_word_freq(const std::string& corpus_path) const;
    PairFreq     get_pair_freqs() const;
    PairFreq     get_pair_freqs_incremental(const SymbolPair& merged_pair);
    void         apply_merge(const SymbolPair& pair, const Symbol& merged);
    void         apply_merge_and_update_pairs(const SymbolPair& pair, const Symbol& merged);
    std::string  merge_symbol(const SymbolPair& p) const { return p.first + p.second; }
    std::vector<std::string> split_text(const std::string& text) const;
    double       train_time_ms_ = 0.0;
};
