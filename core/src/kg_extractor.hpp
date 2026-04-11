#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace edgevdb {

struct Entity {
    std::string text;           // normalized entity surface form (lowercased)
    std::string type;           // "PROPER_NOUN" | "TECHNICAL_TERM" | "NGRAM"
    uint32_t start_pos;         // character offset in chunk
    uint32_t end_pos;
    float confidence;           // 0.0–1.0
};

class KGExtractor {
public:
    KGExtractor();

    void addDomainTerms(const std::vector<std::string>& terms);
    std::vector<Entity> extract(const std::string& text) const;
    std::string normalize(const std::string& text) const;

private:
    std::vector<std::string> domain_terms_;

    std::vector<std::string> splitSentences(const std::string& text) const;
    std::vector<std::pair<std::string, uint32_t>> tokenizeWithPositions(const std::string& text) const;
    bool isCapitalized(const std::string& word) const;
    std::string toLower(const std::string& s) const;
};

} // namespace edgevdb
