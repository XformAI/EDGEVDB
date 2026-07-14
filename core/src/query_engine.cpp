#include "query_engine.hpp"
#include <algorithm>

namespace edgevdb {

QueryEngine::QueryEngine(ChunkStore* chunk_store, HNSWIndex* hnsw_index,
                         PageIndex* page_index, HybridRanker* ranker,
                         KGExtractor* kg_extractor, KnowledgeGraph* kg,
                         KGExpander* kg_expander, ObjectStore* obj_store,
                         RelationIndex* rel_index)
    : chunk_store_(chunk_store), hnsw_index_(hnsw_index),
      page_index_(page_index), ranker_(ranker),
      kg_extractor_(kg_extractor), kg_(kg), kg_expander_(kg_expander),
      obj_store_(obj_store), rel_index_(rel_index) {}

CombinedResult QueryEngine::query(const CombinedQuery& q, const float* query_embedding) const {
    CombinedResult result;

    if (!chunk_store_) return result;
    if (q.retrieval_mode != RetrievalMode::Bm25 && (!hnsw_index_ || !query_embedding)) return result;

    // Step 1: candidate generation, per retrieval mode.
    int over_fetch = q.top_k * static_cast<int>(HNSW_OVER_FETCH_FACTOR);
    std::vector<std::pair<uint64_t, float>> raw_candidates;

    if (q.retrieval_mode != RetrievalMode::Bm25) {
        raw_candidates = hnsw_index_->knnSearch(query_embedding, over_fetch);
    }

    if (q.retrieval_mode != RetrievalMode::Vector &&
        lexical_index_ && !q.text_query.empty()) {
        // Lexical candidates. Union with vector candidates in Hybrid mode:
        // an exact rare-term match is retrievable even when its embedding
        // is not among the nearest vectors. Cosine distance for these is
        // computed against the stored embedding so the ranker sees honest
        // vector scores for every candidate.
        auto lexical = lexical_index_->search(q.text_query, over_fetch);
        std::unordered_set<uint64_t> seen;
        for (const auto& [cid, dist] : raw_candidates) seen.insert(cid);
        for (const auto& [cid, bm25] : lexical) {
            if (seen.count(cid)) continue;
            float dist = 1.0f; // worst-case cosine distance when no embedding available
            ChunkNode chunk;
            if (query_embedding && chunk_store_->get(cid, chunk)) {
                dist = cosineDistance(query_embedding, chunk.embedding, EMBEDDING_DIM);
            }
            raw_candidates.push_back({cid, dist});
        }
    }

    // Step 2: Apply relational filter if set
    if (q.has_filter && obj_store_ && rel_index_) {
        auto matching_objects = obj_store_->query(q.filter_type_name,
                                                   q.filter_property,
                                                   q.filter_value, 10000);

        // Get all chunk IDs linked to matching objects
        std::unordered_set<uint64_t> allowed_chunks;
        for (const auto& obj : matching_objects) {
            if (obj.contains("id")) {
                uint64_t obj_id = obj["id"].get<uint64_t>();
                auto targets = rel_index_->getTargets(q.filter_relation, obj_id);
                allowed_chunks.insert(targets.begin(), targets.end());
                auto sources = rel_index_->getSources(q.filter_relation, obj_id);
                allowed_chunks.insert(sources.begin(), sources.end());
            }
        }

        // The filter is authoritative: when it yields no linked chunks the
        // result set is empty, never silently unfiltered.
        std::vector<std::pair<uint64_t, float>> filtered;
        for (const auto& [cid, dist] : raw_candidates) {
            if (allowed_chunks.count(cid)) {
                filtered.push_back({cid, dist});
            }
        }
        raw_candidates = filtered;
    }

    // Step 3: rank. Pure BM25 mode keeps the lexical ordering (tf-idf) —
    // the hybrid ranker's signals assume an embedding exists.
    if (q.retrieval_mode == RetrievalMode::Bm25) {
        auto lexical = lexical_index_ ? lexical_index_->search(q.text_query, q.top_k)
                                      : std::vector<std::pair<uint64_t, float>>{};
        float best = lexical.empty() ? 1.0f : std::max(1e-6f, lexical[0].second);
        for (const auto& [cid, bm25] : lexical) {
            ChunkNode chunk;
            if (!chunk_store_->get(cid, chunk)) continue;
            QueryResult qr;
            qr.chunk_id = cid;
            qr.score = bm25 / best; // normalised to [0,1]
            qr.keyword_score = qr.score;
            qr.doc_id = chunk.doc_id;
            qr.page_number = chunk.page_number;
            std::memcpy(qr.text, chunk.text, MAX_TEXT_LEN);
            qr.text[MAX_TEXT_LEN - 1] = '\0';
            result.chunks.push_back(qr);
        }
    } else if (ranker_) {
        RankerInput ri;
        ri.hnsw_results = raw_candidates;
        ri.query_embedding = query_embedding;
        ri.query_text = q.text_query;
        ri.chunk_store = chunk_store_;
        ri.page_index = page_index_;
        ri.top_k = q.top_k;
        result.chunks = ranker_->rerank(ri);
    } else {
        // No ranker, just return raw results
        for (const auto& [cid, dist] : raw_candidates) {
            if (static_cast<int>(result.chunks.size()) >= q.top_k) break;
            ChunkNode chunk;
            if (chunk_store_->get(cid, chunk)) {
                QueryResult qr;
                qr.chunk_id = cid;
                qr.score = 1.0f - dist;
                qr.cosine_score = 1.0f - dist;
                qr.doc_id = chunk.doc_id;
                qr.page_number = chunk.page_number;
                std::memcpy(qr.text, chunk.text, MAX_TEXT_LEN);
                qr.text[MAX_TEXT_LEN - 1] = '\0';
                result.chunks.push_back(qr);
            }
        }
    }

    // Step 4: KG expansion
    if (q.use_kg_expansion && kg_expander_) {
        auto expansion_ids = kg_expander_->expand(result.chunks, 5);
        for (uint64_t exp_id : expansion_ids) {
            ChunkNode chunk;
            if (chunk_store_->get(exp_id, chunk)) {
                QueryResult qr;
                qr.chunk_id = exp_id;
                qr.score = 0.0f; // expansion chunks have no score
                qr.doc_id = chunk.doc_id;
                qr.page_number = chunk.page_number;
                std::memcpy(qr.text, chunk.text, MAX_TEXT_LEN);
                qr.text[MAX_TEXT_LEN - 1] = '\0';
                result.chunks.push_back(qr);
            }
        }
    }

    // Step 5: Assemble context
    result.assembled_context = assembleContext(result.chunks, q.token_budget,
                                                q.token_estimate_ratio,
                                                result.context_truncated);
    result.total_tokens_estimated = static_cast<int>(
        result.assembled_context.size() / q.token_estimate_ratio);

    return result;
}

std::string QueryEngine::assembleContext(const std::vector<QueryResult>& results,
                                           int token_budget_val, float token_ratio,
                                           bool& truncated_out) const {
    TokenBudget budget(token_budget_val, token_ratio);
    return budget.assembleContext(results, truncated_out);
}

} // namespace edgevdb
