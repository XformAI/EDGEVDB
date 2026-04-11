#pragma once

#include "schema.hpp"
#include "hnsw_index.hpp"
#include "chunk_store.hpp"
#include "page_index.hpp"
#include "hybrid_ranker.hpp"
#include "kg_expander.hpp"
#include "knowledge_graph.hpp"
#include "kg_extractor.hpp"
#include "object_store.hpp"
#include "relation_index.hpp"
#include "token_budget.hpp"
#include "embedder.hpp"
#include "log.hpp"

#include <string>
#include <vector>
#include <optional>
#include <unordered_set>

namespace edgevdb {

struct CombinedQuery {
    std::string text_query;
    std::string filter_type_name;
    std::string filter_property;
    std::string filter_value;
    bool has_filter = false;
    int top_k = 5;
    bool use_kg_expansion = false;
    int token_budget = 3200;
    float token_estimate_ratio = 3.5f;
};

struct CombinedResult {
    std::vector<QueryResult> chunks;
    std::string assembled_context;
    bool context_truncated = false;
    int total_tokens_estimated = 0;
};

class QueryEngine {
public:
    QueryEngine(ChunkStore* chunk_store, HNSWIndex* hnsw_index,
                PageIndex* page_index, HybridRanker* ranker,
                KGExtractor* kg_extractor, KnowledgeGraph* kg,
                KGExpander* kg_expander, ObjectStore* obj_store,
                RelationIndex* rel_index);

    CombinedResult query(const CombinedQuery& q, const float* query_embedding) const;

    std::string assembleContext(const std::vector<QueryResult>& results,
                                 int token_budget_val, float token_ratio,
                                 bool& truncated_out) const;

private:
    ChunkStore* chunk_store_;
    HNSWIndex* hnsw_index_;
    PageIndex* page_index_;
    HybridRanker* ranker_;
    KGExtractor* kg_extractor_;
    KnowledgeGraph* kg_;
    KGExpander* kg_expander_;
    ObjectStore* obj_store_;
    RelationIndex* rel_index_;
};

} // namespace edgevdb
