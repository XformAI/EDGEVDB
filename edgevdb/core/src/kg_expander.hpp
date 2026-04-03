#pragma once

#include "knowledge_graph.hpp"
#include "kg_extractor.hpp"
#include "chunk_store.hpp"
#include "schema.hpp"

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

namespace edgevdb {

class KGExpander {
public:
    KGExpander(const KnowledgeGraph* kg, const KGExtractor* extractor,
               const ChunkStore* chunk_store);

    std::vector<uint64_t> expand(const std::vector<QueryResult>& base_results,
                                  int max_expansion = 5);

private:
    const KnowledgeGraph* kg_;
    const KGExtractor* extractor_;
    const ChunkStore* chunk_store_;
};

} // namespace edgevdb
