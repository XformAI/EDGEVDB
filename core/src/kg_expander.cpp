#include "kg_expander.hpp"

namespace edgevdb {

KGExpander::KGExpander(const KnowledgeGraph* kg, const KGExtractor* extractor,
                       const ChunkStore* chunk_store)
    : kg_(kg), extractor_(extractor), chunk_store_(chunk_store) {}

std::vector<uint64_t> KGExpander::expand(const std::vector<QueryResult>& base_results,
                                          int max_expansion) {
    if (!kg_ || !extractor_ || !chunk_store_ || base_results.empty()) return {};

    // Collect base chunk IDs to exclude
    std::unordered_set<uint64_t> base_ids;
    for (const auto& r : base_results) {
        base_ids.insert(r.chunk_id);
    }

    // Extract entities from top-3 base results
    std::unordered_set<std::string> query_entities;
    int count = 0;
    for (const auto& r : base_results) {
        if (count >= 3) break;
        auto entities = extractor_->extract(std::string(r.text));
        for (const auto& e : entities) {
            query_entities.insert(e.text);
        }
        count++;
    }

    // For each entity, do 1-hop graph traversal
    std::unordered_set<std::string> related_entities;
    for (const auto& entity : query_entities) {
        auto related = kg_->getRelatedEntities(entity, 1);
        related_entities.insert(related.begin(), related.end());
    }
    // Also include the query entities themselves
    related_entities.insert(query_entities.begin(), query_entities.end());

    // Get all chunk_ids associated with related entities
    std::unordered_map<uint64_t, int> expansion_scores; // chunk_id → count of matching entities
    for (const auto& entity : related_entities) {
        auto chunks = kg_->getChunksForEntity(entity);
        for (uint64_t cid : chunks) {
            if (base_ids.count(cid) == 0) {
                expansion_scores[cid]++;
            }
        }
    }

    // Sort by score descending
    std::vector<std::pair<uint64_t, int>> scored;
    scored.reserve(expansion_scores.size());
    for (const auto& [cid, score] : expansion_scores) {
        scored.push_back({cid, score});
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Return top max_expansion
    std::vector<uint64_t> result;
    for (const auto& [cid, score] : scored) {
        if (static_cast<int>(result.size()) >= max_expansion) break;
        result.push_back(cid);
    }

    return result;
}

} // namespace edgevdb
