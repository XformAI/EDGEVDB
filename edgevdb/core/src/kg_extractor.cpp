#include "kg_extractor.hpp"
#include <algorithm>
#include <cctype>
#include <unordered_set>
#include <sstream>

namespace edgevdb {

KGExtractor::KGExtractor() {}

void KGExtractor::addDomainTerms(const std::vector<std::string>& terms) {
    for (const auto& t : terms) {
        domain_terms_.push_back(toLower(t));
    }
}

std::string KGExtractor::toLower(const std::string& s) const {
    std::string result = s;
    for (auto& c : result) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return result;
}

std::string KGExtractor::normalize(const std::string& text) const {
    std::string result = toLower(text);
    // Trim whitespace
    size_t start = result.find_first_not_of(" \t\r\n");
    size_t end = result.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    result = result.substr(start, end - start + 1);
    // Remove punctuation from ends
    while (!result.empty() && std::ispunct(static_cast<unsigned char>(result.front())))
        result.erase(result.begin());
    while (!result.empty() && std::ispunct(static_cast<unsigned char>(result.back())))
        result.pop_back();
    return result;
}

bool KGExtractor::isCapitalized(const std::string& word) const {
    if (word.empty()) return false;
    return std::isupper(static_cast<unsigned char>(word[0]));
}

std::vector<std::string> KGExtractor::splitSentences(const std::string& text) const {
    std::vector<std::string> sentences;
    std::string current;
    for (size_t i = 0; i < text.size(); i++) {
        current += text[i];
        if (text[i] == '.' || text[i] == '!' || text[i] == '?') {
            sentences.push_back(current);
            current.clear();
        }
    }
    if (!current.empty()) sentences.push_back(current);
    return sentences;
}

std::vector<std::pair<std::string, uint32_t>> KGExtractor::tokenizeWithPositions(const std::string& text) const {
    std::vector<std::pair<std::string, uint32_t>> tokens;
    std::string token;
    uint32_t start = 0;
    for (uint32_t i = 0; i <= static_cast<uint32_t>(text.size()); i++) {
        char c = (i < text.size()) ? text[i] : ' ';
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '\'') {
            if (token.empty()) start = i;
            token += c;
        } else {
            if (!token.empty()) {
                tokens.push_back({token, start});
                token.clear();
            }
        }
    }
    return tokens;
}

std::vector<Entity> KGExtractor::extract(const std::string& text) const {
    std::unordered_map<std::string, Entity> entity_map; // dedup by normalized text

    auto sentences = splitSentences(text);

    for (const auto& sentence : sentences) {
        auto tokens = tokenizeWithPositions(sentence);
        if (tokens.empty()) continue;

        // Proper noun detection: skip first token in sentence
        for (size_t i = 0; i < tokens.size(); i++) {
            const auto& [word, pos] = tokens[i];

            // Technical term detection (case-insensitive)
            std::string lower_word = toLower(word);
            for (const auto& term : domain_terms_) {
                if (lower_word == term) {
                    std::string norm = normalize(word);
                    if (!norm.empty() && entity_map.find(norm) == entity_map.end()) {
                        Entity e;
                        e.text = norm;
                        e.type = "TECHNICAL_TERM";
                        e.start_pos = pos;
                        e.end_pos = pos + static_cast<uint32_t>(word.size());
                        e.confidence = 0.85f;
                        entity_map[norm] = e;
                    }
                }
            }

            // Proper noun: capitalized word not at sentence start
            if (i > 0 && isCapitalized(word) && word.size() > 1) {
                // Try to merge consecutive capitalized tokens (up to 5 words)
                std::string multi_word = word;
                uint32_t end_pos = pos + static_cast<uint32_t>(word.size());
                size_t j = i + 1;
                while (j < tokens.size() && j - i < 5 && isCapitalized(tokens[j].first)) {
                    multi_word += " " + tokens[j].first;
                    end_pos = tokens[j].second + static_cast<uint32_t>(tokens[j].first.size());
                    j++;
                }

                std::string norm = normalize(multi_word);
                if (!norm.empty()) {
                    float conf = (j - i > 1) ? 0.90f : 0.75f; // multi-word = higher confidence
                    auto it = entity_map.find(norm);
                    if (it == entity_map.end() || it->second.confidence < conf) {
                        Entity e;
                        e.text = norm;
                        e.type = "PROPER_NOUN";
                        e.start_pos = pos;
                        e.end_pos = end_pos;
                        e.confidence = conf;
                        entity_map[norm] = e;
                    }
                }
            }

            // Also check single all-caps words as proper nouns (like "BERT")
            if (word.size() >= 2) {
                bool all_upper = true;
                for (char c : word) {
                    if (!std::isupper(static_cast<unsigned char>(c)) &&
                        !std::isdigit(static_cast<unsigned char>(c))) {
                        all_upper = false;
                        break;
                    }
                }
                if (all_upper) {
                    std::string norm = normalize(word);
                    if (!norm.empty() && entity_map.find(norm) == entity_map.end()) {
                        Entity e;
                        e.text = norm;
                        e.type = "PROPER_NOUN";
                        e.start_pos = pos;
                        e.end_pos = pos + static_cast<uint32_t>(word.size());
                        e.confidence = 0.80f;
                        entity_map[norm] = e;
                    }
                }
            }
        }
    }

    // Common short words to filter
    static const std::unordered_set<std::string> skip_words = {
        "she","he","the","a","an","is","in","of","and","to","for","with",
        "on","at","by","it","or","as","be","we","they","i","me","my","his",
        "her","its","not","this","that","do","did","was","are","were","has",
        "have","had","but","if","so","no","yes","how","what","when","where",
        "who","why","dr","mr","mrs","ms"
    };

    std::vector<Entity> result;
    for (const auto& [key, entity] : entity_map) {
        if (skip_words.count(key)) continue;
        if (key.size() <= 1) continue;
        result.push_back(entity);
    }

    // Sort by confidence descending, cap at 20
    std::sort(result.begin(), result.end(), [](const Entity& a, const Entity& b) {
        return a.confidence > b.confidence;
    });
    if (result.size() > 20) result.resize(20);

    return result;
}

} // namespace edgevdb
