#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace edgevdb {

struct TokenizerOutput {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> token_type_ids;
    int actual_length;
};

class WordPieceTokenizer {
public:
    WordPieceTokenizer();
    explicit WordPieceTokenizer(const std::string& vocab_path);

    bool loadVocab(const std::string& vocab_path);
    TokenizerOutput encode(const std::string& text, int max_length = 128) const;
    bool isLoaded() const { return vocab_loaded_; }

    static constexpr int CLS_ID = 101;
    static constexpr int SEP_ID = 102;
    static constexpr int PAD_ID = 0;
    static constexpr int UNK_ID = 100;

private:
    std::unordered_map<std::string, int> vocab_;
    bool vocab_loaded_;

    std::string normalizeText(const std::string& text) const;
    std::vector<std::string> basicTokenize(const std::string& text) const;
    std::vector<int> wordPieceTokenize(const std::string& word) const;
};

} // namespace edgevdb
