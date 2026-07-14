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

    // Resolve special-token ids from the vocab itself instead of trusting
    // the standard BERT positions; a shifted or custom vocab would
    // otherwise frame every sequence with the wrong ids.
    auto lookup = [this](const char* tok, int fallback) {
        auto it = vocab_.find(tok);
        return it != vocab_.end() ? it->second : fallback;
    };
    cls_id_ = lookup("[CLS]", CLS_ID);
    sep_id_ = lookup("[SEP]", SEP_ID);
    pad_id_ = lookup("[PAD]", PAD_ID);
    unk_id_ = lookup("[UNK]", UNK_ID);

    EVDB_LOG_INFO("Tokenizer: Loaded %zu vocab entries", vocab_.size());
    return vocab_loaded_;
}

std::string WordPieceTokenizer::normalizeText(const std::string& text) const {
    // UTF-8-aware normalization: lowercase ASCII, fold Latin-1 accented
    // letters (decoded as codepoints, not raw bytes) to their base letter,
    // and pass every other codepoint through unchanged so non-Latin scripts
    // survive intact (unknown words become [UNK] later, not empty).
    std::string result;
    result.reserve(text.size());

    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        if (c < 0x80) {
            // ASCII fast path
            if (c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c + 32);
            result += static_cast<char>(c);
            i++;
            continue;
        }

        // Decode one UTF-8 codepoint (with validation).
        uint32_t cp = 0;
        size_t len = 0;
        if ((c & 0xE0) == 0xC0)      { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else { i++; continue; } // stray continuation/invalid byte: drop it

        if (i + len > n) break; // truncated sequence at end of string
        bool valid = true;
        for (size_t k = 1; k < len; k++) {
            unsigned char cc = static_cast<unsigned char>(text[i + k]);
            if ((cc & 0xC0) != 0x80) { valid = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!valid) { i++; continue; }

        // Latin-1 supplement accent folding on the decoded codepoint.
        char folded = 0;
        if ((cp >= 0x00C0 && cp <= 0x00C5) || (cp >= 0x00E0 && cp <= 0x00E5)) folded = 'a';
        else if ((cp >= 0x00C8 && cp <= 0x00CB) || (cp >= 0x00E8 && cp <= 0x00EB)) folded = 'e';
        else if ((cp >= 0x00CC && cp <= 0x00CF) || (cp >= 0x00EC && cp <= 0x00EF)) folded = 'i';
        else if ((cp >= 0x00D2 && cp <= 0x00D6) || (cp >= 0x00F2 && cp <= 0x00F6)) folded = 'o';
        else if ((cp >= 0x00D9 && cp <= 0x00DC) || (cp >= 0x00F9 && cp <= 0x00FC)) folded = 'u';
        else if (cp == 0x00C7 || cp == 0x00E7) folded = 'c';
        else if (cp == 0x00D1 || cp == 0x00F1) folded = 'n';

        if (folded) {
            result += folded;
        } else {
            // Preserve the original (valid) multi-byte sequence untouched.
            result.append(text, i, len);
        }
        i += len;
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
            ids.push_back(unk_id_);
            return ids;
        }

        start = end;
    }

    return ids;
}

TokenizerOutput WordPieceTokenizer::encode(const std::string& text, int max_length) const {
    TokenizerOutput output;
    output.input_ids.resize(max_length, pad_id_);
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
    output.input_ids[0] = cls_id_;
    output.attention_mask[0] = 1;

    for (size_t i = 0; i < all_ids.size(); i++) {
        output.input_ids[i + 1] = all_ids[i];
        output.attention_mask[i + 1] = 1;
    }

    int sep_pos = static_cast<int>(all_ids.size()) + 1;
    output.input_ids[sep_pos] = sep_id_;
    output.attention_mask[sep_pos] = 1;

    output.actual_length = sep_pos + 1; // CLS + tokens + SEP

    return output;
}

} // namespace edgevdb
