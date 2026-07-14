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
#include "lexical_index.hpp"
#include "token_budget.hpp"
#include "embedder.hpp"
#include "log.hpp"

#include <string>
#include <vector>
#include <optional>
#include <unordered_set>

namespace edgevdb {

// Retrieval strategy: which candidate sources feed the ranker.
enum class RetrievalMode {
    Hybrid = 0, // union of vector + BM25 candidates (default)
    Vector = 1, // HNSW only (semantic)
    Bm25 = 2,   // lexical only — no embedding required
};

struct CombinedQuery {
    std::string text_query;
    std::string filter_type_name;
    std::string filter_property;
    std::string filter_value;
    // Relation type linking filter objects to chunks (configurable; the
    // previous implementation hard-coded "sourceDocument").
    std::string filter_relation = "sourceDocument";
    bool has_filter = false;
    int top_k = 5;
    bool use_kg_expansion = false;
    int token_budget = 3200;
    float token_estimate_ratio = 3.5f;
    RetrievalMode retrieval_mode = RetrievalMode::Hybrid;
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

    void setLexicalIndex(LexicalIndex* lexical) { lexical_index_ = lexical; }

    // query_embedding may be null only in Bm25 retrieval mode.
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
    LexicalIndex* lexical_index_ = nullptr;
};

} // namespace edgevdb
