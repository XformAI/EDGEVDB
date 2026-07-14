#pragma once

#include "schema.hpp"
#include "chunk_store.hpp"
#include "page_index.hpp"
#include "knowledge_graph.hpp"
#include "kg_extractor.hpp"
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

// Ranking strategy.
//  LinearBlend — original: alpha*cosine + beta*page + gamma*keyword.
//  RRF         — Reciprocal Rank Fusion over the same three signals:
//                score = Σ w_s / (k + rank_s). Rank fusion is robust to the
//                signals' incompatible score scales (Cormack et al., 2009).
//  GraphRRF    — RRF plus a fourth signal: knowledge-graph entity overlap
//                between the query's extracted entities and each chunk.
enum class RankerMode { LinearBlend = 0, RRF = 1, GraphRRF = 2 };

class HybridRanker {
public:
    HybridRanker(float alpha = 0.70f, float beta = 0.20f, float gamma = 0.10f);

    std::vector<QueryResult> rerank(const RankerInput& input) const;

    void setWeights(float alpha, float beta, float gamma);
    void resetWeights();

    void setMode(RankerMode mode, float rrf_k = 60.0f) {
        mode_ = mode;
        rrf_k_ = (rrf_k > 0.0f) ? rrf_k : 60.0f;
    }
    RankerMode mode() const { return mode_; }

    // Optional graph inputs for GraphRRF; when either is null the mode
    // degrades gracefully to plain RRF.
    void setKnowledgeGraph(const KnowledgeGraph* kg, const KGExtractor* extractor) {
        kg_ = kg;
        kg_extractor_ = extractor;
    }

    static float computeKeywordOverlap(const std::string& query, const std::string& chunk_text);

private:
    float alpha_;
    float beta_;
    float gamma_;
    RankerMode mode_ = RankerMode::LinearBlend;
    float rrf_k_ = 60.0f;
    const KnowledgeGraph* kg_ = nullptr;
    const KGExtractor* kg_extractor_ = nullptr;

    void applyRrfScores(std::vector<QueryResult>& results,
                        const std::vector<float>& graph_scores) const;

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

        // Combined score (LinearBlend; RRF modes overwrite below)
        qr.score = alpha_ * qr.cosine_score + beta_ * qr.page_score + gamma_ * qr.keyword_score;

        results.push_back(qr);
    }

    if (mode_ != RankerMode::LinearBlend) {
        // GraphRRF's fourth signal: how many of the query's entities also
        // occur in each chunk (via the on-device knowledge graph).
        std::vector<float> graph_scores(results.size(), 0.0f);
        if (mode_ == RankerMode::GraphRRF && kg_ && kg_extractor_ && !input.query_text.empty()) {
            auto query_entities = kg_extractor_->extract(input.query_text);
            std::unordered_map<uint64_t, float> chunk_overlap;
            for (const auto& entity : query_entities) {
                for (uint64_t cid : kg_->getChunksForEntity(entity.text)) {
                    chunk_overlap[cid] += 1.0f;
                }
            }
            for (size_t i = 0; i < results.size(); i++) {
                auto it = chunk_overlap.find(results[i].chunk_id);
                if (it != chunk_overlap.end()) graph_scores[i] = it->second;
            }
        }
        applyRrfScores(results, graph_scores);
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

inline void HybridRanker::applyRrfScores(std::vector<QueryResult>& results,
                                         const std::vector<float>& graph_scores) const {
    if (results.empty()) return;
    const size_t n = results.size();

    // Rank items per signal (1 = best) and fuse: Σ w_s / (k + rank_s).
    // The linear weights double as fusion weights so existing alpha/beta/
    // gamma tuning carries over; the graph signal reuses gamma's weight.
    auto ranksBy = [n](const std::vector<float>& values) {
        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return values[a] > values[b]; });
        std::vector<int> ranks(n);
        for (size_t r = 0; r < n; r++) ranks[order[r]] = static_cast<int>(r) + 1;
        return ranks;
    };

    std::vector<float> cosine(n), page(n), keyword(n);
    for (size_t i = 0; i < n; i++) {
        cosine[i] = results[i].cosine_score;
        page[i] = results[i].page_score;
        keyword[i] = results[i].keyword_score;
    }

    auto cosine_ranks = ranksBy(cosine);
    auto page_ranks = ranksBy(page);
    auto keyword_ranks = ranksBy(keyword);
    std::vector<int> graph_ranks;
    bool use_graph = false;
    for (float g : graph_scores) {
        if (g > 0.0f) { use_graph = true; break; }
    }
    if (use_graph) graph_ranks = ranksBy(graph_scores);

    for (size_t i = 0; i < n; i++) {
        float score = alpha_ / (rrf_k_ + static_cast<float>(cosine_ranks[i]))
                    + beta_  / (rrf_k_ + static_cast<float>(page_ranks[i]))
                    + gamma_ / (rrf_k_ + static_cast<float>(keyword_ranks[i]));
        // The graph signal is sparse: a chunk sharing no entity with the
        // query is absent from that ranked list, so it contributes nothing
        // (standard RRF treatment of missing documents).
        if (use_graph && graph_scores[i] > 0.0f) {
            score += gamma_ / (rrf_k_ + static_cast<float>(graph_ranks[i]));
        }
        results[i].score = score;
    }
}

} // namespace edgevdb
