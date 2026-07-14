#pragma once

// In-memory BM25 inverted index over chunk text.
//
// Enables three retrieval modes: pure lexical retrieval (no embedding model
// required at all), pure vector retrieval, and true hybrid retrieval where
// lexical and vector candidate sets are unioned before re-ranking — so a
// chunk with an exact rare-term match is retrievable even when its embedding
// is not among the nearest vectors.
//
// Scaling: postings are vector<(chunk_id, tf)> per term; memory is O(total
// tokens). Query cost is O(Σ posting-list lengths of query terms) with a
// top-k heap — independent of corpus size for selective terms. The index is
// rebuilt from the chunk store at open (tokenization of 50k×512-byte chunks
// takes well under a second), so it needs no on-disk format of its own.

#include "schema.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace edgevdb {

class LexicalIndex {
public:
    static constexpr float K1 = 1.2f;
    static constexpr float B = 0.75f;

    void addChunk(uint64_t chunk_id, const std::string& text) {
        auto terms = tokenize(text);
        if (terms.empty()) return;

        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        // Re-adding an id replaces the previous posting entries.
        removeUnlocked(chunk_id);

        std::unordered_map<std::string, uint32_t> tf;
        for (const auto& t : terms) tf[t]++;

        for (const auto& [term, count] : tf) {
            postings_[term].push_back({chunk_id, count});
        }
        doc_len_[chunk_id] = static_cast<uint32_t>(terms.size());
        total_len_ += terms.size();
    }

    void removeChunk(uint64_t chunk_id) {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        removeUnlocked(chunk_id);
    }

    void clear() {
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);
        postings_.clear();
        doc_len_.clear();
        total_len_ = 0;
    }

    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        return doc_len_.size();
    }

    // BM25 top-k retrieval. Returns (chunk_id, bm25_score) sorted descending.
    std::vector<std::pair<uint64_t, float>> search(const std::string& query, int top_k) const {
        auto terms = tokenize(query);
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);

        const size_t N = doc_len_.size();
        if (N == 0 || terms.empty() || top_k <= 0) return {};
        const float avg_len = static_cast<float>(total_len_) / static_cast<float>(N);

        // Deduplicate query terms (BM25 uses per-term idf once).
        std::sort(terms.begin(), terms.end());
        terms.erase(std::unique(terms.begin(), terms.end()), terms.end());

        std::unordered_map<uint64_t, float> scores;
        for (const auto& term : terms) {
            auto it = postings_.find(term);
            if (it == postings_.end()) continue;
            const auto& plist = it->second;
            const float df = static_cast<float>(plist.size());
            const float idf = std::log(1.0f + (static_cast<float>(N) - df + 0.5f) / (df + 0.5f));
            for (const auto& [id, tf] : plist) {
                auto dl_it = doc_len_.find(id);
                if (dl_it == doc_len_.end()) continue;
                const float dl = static_cast<float>(dl_it->second);
                const float tf_f = static_cast<float>(tf);
                scores[id] += idf * (tf_f * (K1 + 1.0f)) /
                              (tf_f + K1 * (1.0f - B + B * dl / avg_len));
            }
        }

        std::vector<std::pair<uint64_t, float>> results(scores.begin(), scores.end());
        std::partial_sort(results.begin(),
                          results.begin() + std::min<size_t>(top_k, results.size()),
                          results.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        if (results.size() > static_cast<size_t>(top_k)) results.resize(top_k);
        return results;
    }

    static std::vector<std::string> tokenize(const std::string& text) {
        std::vector<std::string> terms;
        std::string term;
        for (char raw : text) {
            unsigned char c = static_cast<unsigned char>(raw);
            if (std::isalnum(c) || c >= 0x80) {
                term += static_cast<char>(std::tolower(c));
            } else if (!term.empty()) {
                if (term.size() > 1) terms.push_back(term);
                term.clear();
            }
        }
        if (term.size() > 1) terms.push_back(term);
        return terms;
    }

private:
    void removeUnlocked(uint64_t chunk_id) {
        auto it = doc_len_.find(chunk_id);
        if (it == doc_len_.end()) return;
        total_len_ -= it->second;
        doc_len_.erase(it);
        for (auto& [term, plist] : postings_) {
            plist.erase(std::remove_if(plist.begin(), plist.end(),
                                       [&](const auto& p) { return p.first == chunk_id; }),
                        plist.end());
        }
    }

    // term → [(chunk_id, term_frequency)]
    std::unordered_map<std::string, std::vector<std::pair<uint64_t, uint32_t>>> postings_;
    std::unordered_map<uint64_t, uint32_t> doc_len_;
    uint64_t total_len_ = 0;
    mutable std::shared_mutex rw_mutex_;
};

} // namespace edgevdb
