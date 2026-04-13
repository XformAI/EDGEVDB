#pragma once

#include "schema.hpp"
#include "chunk_store.hpp"
#include "page_index.hpp"
#include "log.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_set>
#include <array>
#include <cstring>

namespace edgevdb {

// ── Stopwords ─────────────────────────────────────────────
static const std::array<const char*, 50> STOPWORDS = {{
    "the","a","an","is","in","of","and","to","for","with",
    "on","at","by","it","or","as","be","was","are","were",
    "been","has","have","had","do","does","did","will","would","could",
    "should","may","might","can","this","that","these","those","he","she",
    "they","we","you","i","me","my","his","her","its","not"
}};

struct RankerInput {
    std::vector<std::pair<uint64_t, float>> hnsw_results; // (chunk_id, cosine_distance)
    const float* query_embedding;
    std::string query_text;
    const ChunkStore* chunk_store;
    const PageIndex* page_index;
    int top_k;
};

class HybridRanker {
public:
    HybridRanker(float alpha = 0.70f, float beta = 0.20f, float gamma = 0.10f);

    std::vector<QueryResult> rerank(const RankerInput& input) const;

    void setWeights(float alpha, float beta, float gamma);
    void resetWeights();

    static float computeKeywordOverlap(const std::string& query, const std::string& chunk_text);

private:
    float alpha_;
    float beta_;
    float gamma_;

    static std::vector<std::string> tokenize(const std::string& text);
    static std::string toLowerStr(const std::string& s);
    static bool isStopword(const std::string& word);
};

// ── Implementation ────────────────────────────────────────

inline HybridRanker::HybridRanker(float alpha, float beta, float gamma)
    : alpha_(alpha), beta_(beta), gamma_(gamma) {}

inline void HybridRanker::setWeights(float alpha, float beta, float gamma) {
    alpha_ = alpha; beta_ = beta; gamma_ = gamma;
}

inline void HybridRanker::resetWeights() {
    alpha_ = 0.70f; beta_ = 0.20f; gamma_ = 0.10f;
}

inline std::string HybridRanker::toLowerStr(const std::string& s) {
    std::string result = s;
    for (auto& c : result) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return result;
}

inline bool HybridRanker::isStopword(const std::string& word) {
    for (const auto* sw : STOPWORDS) {
        if (word == sw) return true;
    }
    return false;
}

inline std::vector<std::string> HybridRanker::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string lower = toLowerStr(text);
    std::string token;
    for (char c : lower) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            token += c;
        } else {
            if (!token.empty() && !isStopword(token)) {
                tokens.push_back(token);
            }
            token.clear();
        }
    }
    if (!token.empty() && !isStopword(token)) {
        tokens.push_back(token);
    }
    return tokens;
}

inline float HybridRanker::computeKeywordOverlap(const std::string& query, const std::string& chunk_text) {
    auto q_tokens = tokenize(query);
    auto c_tokens = tokenize(chunk_text);

    if (q_tokens.empty() || c_tokens.empty()) return 0.0f;

    std::unordered_set<std::string> q_set(q_tokens.begin(), q_tokens.end());
    std::unordered_set<std::string> c_set(c_tokens.begin(), c_tokens.end());

    // Jaccard coefficient
    size_t intersection = 0;
    for (const auto& w : q_set) {
        if (c_set.count(w)) intersection++;
    }

    std::unordered_set<std::string> union_set = q_set;
    union_set.insert(c_set.begin(), c_set.end());

    if (union_set.empty()) return 0.0f;
    return static_cast<float>(intersection) / static_cast<float>(union_set.size());
}

inline std::vector<QueryResult> HybridRanker::rerank(const RankerInput& input) const {
    if (input.hnsw_results.empty()) return {};

    // Find best neighbour (highest cosine similarity)
    uint64_t best_chunk_id = input.hnsw_results[0].first;
    float best_sim = 1.0f - input.hnsw_results[0].second;
    for (const auto& [cid, dist] : input.hnsw_results) {
        float sim = 1.0f - dist;
        if (sim > best_sim) {
            best_sim = sim;
            best_chunk_id = cid;
        }
    }

    std::vector<QueryResult> results;
    results.reserve(input.hnsw_results.size());

    for (const auto& [chunk_id, distance] : input.hnsw_results) {
        ChunkNode chunk;
        if (!input.chunk_store->get(chunk_id, chunk)) continue;

        QueryResult qr;
        qr.chunk_id = chunk_id;
        qr.doc_id = chunk.doc_id;
        qr.page_number = chunk.page_number;
        std::memcpy(qr.text, chunk.text, MAX_TEXT_LEN);
        qr.text[MAX_TEXT_LEN - 1] = '\0';

        // Cosine score
        qr.cosine_score = 1.0f - distance;

        // Page proximity score
        if (chunk_id == best_chunk_id) {
            qr.page_score = 1.0f;
        } else if (input.page_index) {
            qr.page_score = input.page_index->computePageProximity(chunk_id, best_chunk_id);
        } else {
            qr.page_score = 0.0f;
        }

        // Keyword overlap score
        qr.keyword_score = computeKeywordOverlap(input.query_text, std::string(chunk.text));

        // Combined score
        qr.score = alpha_ * qr.cosine_score + beta_ * qr.page_score + gamma_ * qr.keyword_score;

        results.push_back(qr);
    }

    // Sort by final score descending
    std::sort(results.begin(), results.end(), [](const QueryResult& a, const QueryResult& b) {
        return a.score > b.score;
    });

    // Trim to top_k
    if (input.top_k > 0 && static_cast<int>(results.size()) > input.top_k) {
        results.resize(input.top_k);
    }

    return results;
}

} // namespace edgevdb
