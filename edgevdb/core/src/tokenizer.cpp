#include "tokenizer.hpp"
#include "log.hpp"
#include <fstream>
#include <algorithm>
#include <cctype>
#include <sstream>

namespace edgevdb {

WordPieceTokenizer::WordPieceTokenizer() : vocab_loaded_(false) {}

WordPieceTokenizer::WordPieceTokenizer(const std::string& vocab_path) : vocab_loaded_(false) {
    loadVocab(vocab_path);
}

bool WordPieceTokenizer::loadVocab(const std::string& vocab_path) {
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        EVDB_LOG_ERROR("Tokenizer: Cannot open vocab file %s", vocab_path.c_str());
        return false;
    }

    vocab_.clear();
    std::string line;
    int id = 0;
    while (std::getline(file, line)) {
        // Remove trailing whitespace
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' '))
            line.pop_back();
        if (!line.empty()) {
            vocab_[line] = id;
        }
        id++;
    }

    vocab_loaded_ = !vocab_.empty();
    EVDB_LOG_INFO("Tokenizer: Loaded %zu vocab entries", vocab_.size());
    return vocab_loaded_;
}

std::string WordPieceTokenizer::normalizeText(const std::string& text) const {
    std::string result;
    result.reserve(text.size());
    for (unsigned char c : text) {
        // Convert to lowercase
        if (c >= 'A' && c <= 'Z') {
            result += static_cast<char>(c + 32);
        }
        // Basic ASCII accent stripping (common accented chars)
        else if (c == 0xC0 || c == 0xC1 || c == 0xC2 || c == 0xC3 || c == 0xC4 || c == 0xC5 ||
                 c == 0xE0 || c == 0xE1 || c == 0xE2 || c == 0xE3 || c == 0xE4 || c == 0xE5) {
            result += 'a';
        } else if (c == 0xC8 || c == 0xC9 || c == 0xCA || c == 0xCB ||
                   c == 0xE8 || c == 0xE9 || c == 0xEA || c == 0xEB) {
            result += 'e';
        } else if (c == 0xCC || c == 0xCD || c == 0xCE || c == 0xCF ||
                   c == 0xEC || c == 0xED || c == 0xEE || c == 0xEF) {
            result += 'i';
        } else if (c == 0xD2 || c == 0xD3 || c == 0xD4 || c == 0xD5 || c == 0xD6 ||
                   c == 0xF2 || c == 0xF3 || c == 0xF4 || c == 0xF5 || c == 0xF6) {
            result += 'o';
        } else if (c == 0xD9 || c == 0xDA || c == 0xDB || c == 0xDC ||
                   c == 0xF9 || c == 0xFA || c == 0xFB || c == 0xFC) {
            result += 'u';
        } else if (c < 0x80) {
            result += static_cast<char>(c);
        }
        // Skip combining characters / non-ASCII for simplicity
    }
    return result;
}

std::vector<std::string> WordPieceTokenizer::basicTokenize(const std::string& text) const {
    std::string normalized = normalizeText(text);
    std::vector<std::string> tokens;
    std::string token;

    for (size_t i = 0; i < normalized.size(); i++) {
        char c = normalized[i];

        // Split on whitespace
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
            continue;
        }

        // Split punctuation as separate tokens
        if (std::ispunct(static_cast<unsigned char>(c))) {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
            tokens.push_back(std::string(1, c));
            continue;
        }

        token += c;
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }

    return tokens;
}

std::vector<int> WordPieceTokenizer::wordPieceTokenize(const std::string& word) const {
    std::vector<int> ids;

    if (word.empty()) return ids;

    // Check if whole word is in vocab
    auto it = vocab_.find(word);
    if (it != vocab_.end()) {
        ids.push_back(it->second);
        return ids;
    }

    // WordPiece segmentation
    size_t start = 0;
    bool failed = false;

    while (start < word.size()) {
        size_t end = word.size();
        bool found = false;
        std::string substr;

        while (end > start) {
            substr = word.substr(start, end - start);
            if (start > 0) {
                substr = "##" + substr;
            }

            auto vit = vocab_.find(substr);
            if (vit != vocab_.end()) {
                ids.push_back(vit->second);
                found = true;
                break;
            }
            end--;
        }

        if (!found) {
            // Unknown token
            ids.clear();
            ids.push_back(UNK_ID);
            return ids;
        }

        start = end;
    }

    return ids;
}

TokenizerOutput WordPieceTokenizer::encode(const std::string& text, int max_length) const {
    TokenizerOutput output;
    output.input_ids.resize(max_length, PAD_ID);
    output.attention_mask.resize(max_length, 0);
    output.token_type_ids.resize(max_length, 0);

    // Basic tokenize
    auto basic_tokens = basicTokenize(text);

    // WordPiece tokenize each basic token
    std::vector<int> all_ids;
    for (const auto& token : basic_tokens) {
        auto wp_ids = wordPieceTokenize(token);
        all_ids.insert(all_ids.end(), wp_ids.begin(), wp_ids.end());
    }

    // Truncate to max_length - 2 (leaving room for CLS and SEP)
    int max_tokens = max_length - 2;
    if (static_cast<int>(all_ids.size()) > max_tokens) {
        all_ids.resize(max_tokens);
    }

    // Build final sequence: [CLS] + tokens + [SEP] + [PAD...]
    output.input_ids[0] = CLS_ID;
    output.attention_mask[0] = 1;

    for (size_t i = 0; i < all_ids.size(); i++) {
        output.input_ids[i + 1] = all_ids[i];
        output.attention_mask[i + 1] = 1;
    }

    int sep_pos = static_cast<int>(all_ids.size()) + 1;
    output.input_ids[sep_pos] = SEP_ID;
    output.attention_mask[sep_pos] = 1;

    output.actual_length = sep_pos + 1; // CLS + tokens + SEP

    return output;
}

} // namespace edgevdb
