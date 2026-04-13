/**
 * @file vectordb_c_api.cpp
 * @brief Implementation of the public C API declared in vectordb.h
 */

#include "edgevdb/vectordb.h"
#include "edgevdb/version.h"

#include "schema.hpp"
#include "chunk_store.hpp"
#include "hnsw_index.hpp"
#include "page_index.hpp"
#include "hybrid_ranker.hpp"
#include "kg_extractor.hpp"
#include "knowledge_graph.hpp"
#include "kg_expander.hpp"
#include "embedder.hpp"
#include "object_store.hpp"
#include "relation_index.hpp"
#include "query_engine.hpp"
#include "sync_engine.hpp"
#include "token_budget.hpp"
#include "log.hpp"

#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>

using namespace edgevdb;

// ── Internal handle structs ───────────────────────────────

struct EvdbHandle_ {
    std::unique_ptr<ChunkStore>     chunk_store;
    std::unique_ptr<HNSWIndex>      hnsw_index;
    std::unique_ptr<PageIndex>      page_index;
    std::unique_ptr<HybridRanker>   ranker;
    std::unique_ptr<KGExtractor>    kg_extractor;
    std::unique_ptr<KnowledgeGraph> kg;
    std::unique_ptr<KGExpander>     kg_expander;
    std::unique_ptr<ObjectStore>    obj_store;
    std::unique_ptr<RelationIndex>  rel_index;
    std::unique_ptr<QueryEngine>    query_engine;
    EvdbConfig config;
    std::string storage_dir;
};

struct EvdbEmbedder_ {
    std::unique_ptr<OnnxEmbedder> embedder;
};

struct EvdbQueryHandle_ {
    std::vector<QueryResult> results;
    std::string context_string;
};

struct EvdbSyncEngine_ {
    std::unique_ptr<SyncEngine> engine;
};

// ── Helper ────────────────────────────────────────────────

static std::string pathJoin(const std::string& dir, const std::string& file) {
    if (dir.empty()) return file;
    char last = dir.back();
    if (last == '/' || last == '\\') return dir + file;
    return dir + "/" + file;
}

// ── Lifecycle ─────────────────────────────────────────────

extern "C" {

void evdb_default_config(EvdbConfig* out) {
    if (!out) return;
    std::memset(out, 0, sizeof(EvdbConfig));
    out->storage_dir = nullptr;
    out->hnsw_M = 16;
    out->hnsw_ef_construction = 200;
    out->hnsw_ef_search = 64;
    out->ranker_alpha = 0.70f;
    out->ranker_beta = 0.20f;
    out->ranker_gamma = 0.10f;
    out->token_budget = 3200;
    out->embedding_threads = 2;
    out->enable_knowledge_graph = 1;
    out->enable_sync = 0;
    out->device_id = nullptr;
}

EvdbHandle* evdb_open(const EvdbConfig* config) {
    if (!config || !config->storage_dir) return nullptr;

    try {
        auto h = new EvdbHandle_();
        h->config = *config;
        h->storage_dir = config->storage_dir;

        h->chunk_store = std::make_unique<ChunkStore>(pathJoin(h->storage_dir, "chunks.bin"));
        h->hnsw_index = std::make_unique<HNSWIndex>(
            pathJoin(h->storage_dir, "hnsw.bin"),
            config->hnsw_M, config->hnsw_ef_construction, config->hnsw_ef_search);
        h->page_index = std::make_unique<PageIndex>(pathJoin(h->storage_dir, "page.bin"));
        h->ranker = std::make_unique<HybridRanker>(
            config->ranker_alpha, config->ranker_beta, config->ranker_gamma);
        h->kg_extractor = std::make_unique<KGExtractor>();
        h->kg = std::make_unique<KnowledgeGraph>(pathJoin(h->storage_dir, "kg.bin"));
        h->kg_expander = std::make_unique<KGExpander>(
            h->kg.get(), h->kg_extractor.get(), h->chunk_store.get());
        h->obj_store = std::make_unique<ObjectStore>(pathJoin(h->storage_dir, "objects.bin"));
        h->rel_index = std::make_unique<RelationIndex>();

        h->query_engine = std::make_unique<QueryEngine>(
            h->chunk_store.get(), h->hnsw_index.get(), h->page_index.get(),
            h->ranker.get(), h->kg_extractor.get(), h->kg.get(), h->kg_expander.get(),
            h->obj_store.get(), h->rel_index.get());

        // Open all stores
        h->chunk_store->open();
        h->hnsw_index->open();
        h->page_index->open();
        h->kg->open();
        h->obj_store->open();
        h->rel_index->open(pathJoin(h->storage_dir, "relations.bin"));

        EVDB_LOG_INFO("EdgeVDB: Opened at %s", h->storage_dir.c_str());
        return h;
    } catch (const std::exception& e) {
        EVDB_LOG_ERROR("EdgeVDB: Failed to open: %s", e.what());
        return nullptr;
    }
}

void evdb_close(EvdbHandle* h) {
    if (!h) return;
    try {
        evdb_save(h);
    } catch (...) {}
    delete h;
}

EvdbError evdb_save(EvdbHandle* h) {
    if (!h) return EVDB_ERR_NULL_HANDLE;
    try {
        h->chunk_store->save();
        h->hnsw_index->save();
        h->page_index->save();
        h->kg->save();
        h->obj_store->save();
        h->rel_index->save(pathJoin(h->storage_dir, "relations.bin"));
        return EVDB_OK;
    } catch (const std::exception& e) {
        EVDB_LOG_ERROR("EdgeVDB: Save failed: %s", e.what());
        return EVDB_ERR_IO;
    }
}

// ── Embedding ─────────────────────────────────────────────

EvdbEmbedder* evdb_embedder_create(const char* model_path, const char* vocab_path, int threads) {
    if (!model_path || !vocab_path) return nullptr;
    try {
        auto e = new EvdbEmbedder_();
        e->embedder = std::make_unique<OnnxEmbedder>();
        if (!e->embedder->initialize(model_path, vocab_path, threads)) {
            delete e;
            return nullptr;
        }
        return e;
    } catch (...) {
        return nullptr;
    }
}

void evdb_embedder_destroy(EvdbEmbedder* e) {
    if (e) {
        e->embedder->shutdown();
        delete e;
    }
}

EvdbError evdb_embed_text(EvdbEmbedder* e, const char* text, float* out_384) {
    if (!e) return EVDB_ERR_NULL_HANDLE;
    if (!text || !out_384) return EVDB_ERR_INVALID_ARG;
    try {
        if (!e->embedder->embed(text, out_384)) return EVDB_ERR_ONNX;
        return EVDB_OK;
    } catch (...) {
        return EVDB_ERR_IO;
    }
}

// ── Insert ────────────────────────────────────────────────

EvdbError evdb_insert_text(EvdbHandle* h, EvdbEmbedder* e,
                            const char* text, uint32_t doc_id, uint32_t page_number,
                            uint64_t* out_chunk_id) {
    if (!h || !e) return EVDB_ERR_NULL_HANDLE;
    if (!text) return EVDB_ERR_INVALID_ARG;

    try {
        float embedding[384];
        if (!e->embedder->embed(text, embedding)) return EVDB_ERR_ONNX;
        return evdb_insert_chunk(h, text, embedding, doc_id, page_number, out_chunk_id);
    } catch (const std::exception& ex) {
        EVDB_LOG_ERROR("insert_text: %s", ex.what());
        return EVDB_ERR_IO;
    }
}

EvdbError evdb_insert_chunk(EvdbHandle* h,
                             const char* text, const float* embedding_384,
                             uint32_t doc_id, uint32_t page_number,
                             uint64_t* out_chunk_id) {
    if (!h) return EVDB_ERR_NULL_HANDLE;
    if (!text || !embedding_384) return EVDB_ERR_INVALID_ARG;

    try {
        ChunkNode chunk;
        std::strncpy(chunk.text, text, MAX_TEXT_LEN - 1);
        chunk.text[MAX_TEXT_LEN - 1] = '\0';
        std::memcpy(chunk.embedding, embedding_384, EMBEDDING_DIM * sizeof(float));
        chunk.doc_id = doc_id;
        chunk.page_number = page_number;

        uint64_t id = h->chunk_store->put(chunk);

        // Update HNSW index
        ChunkNode stored;
        h->chunk_store->get(id, stored);
        h->hnsw_index->insert(id, stored.embedding);

        // Update page index
        h->page_index->insert(id, doc_id, page_number);

        // Extract entities and update KG
        if (h->config.enable_knowledge_graph) {
            auto entities = h->kg_extractor->extract(text);
            h->kg->addChunkEntities(id, entities);
        }

        if (out_chunk_id) *out_chunk_id = id;
        return EVDB_OK;
    } catch (const std::exception& ex) {
        EVDB_LOG_ERROR("insert_chunk: %s", ex.what());
        return EVDB_ERR_IO;
    }
}

EvdbError evdb_remove_chunk(EvdbHandle* h, uint64_t chunk_id) {
    if (!h) return EVDB_ERR_NULL_HANDLE;
    try {
        h->chunk_store->remove(chunk_id);
        h->hnsw_index->remove(chunk_id);
        h->page_index->remove(chunk_id);
        h->kg->removeChunk(chunk_id);
        return EVDB_OK;
    } catch (...) {
        return EVDB_ERR_IO;
    }
}

// ── Query ─────────────────────────────────────────────────

EvdbQueryHandle* evdb_query_text(EvdbHandle* h, EvdbEmbedder* e,
                                  const char* query_text, int top_k,
                                  int use_kg_expansion) {
    if (!h || !e || !query_text) return nullptr;

    try {
        float query_emb[384];
        if (!e->embedder->embed(query_text, query_emb)) return nullptr;

        CombinedQuery cq;
        cq.text_query = query_text;
        cq.top_k = top_k;
        cq.token_budget = h->config.token_budget;
        cq.use_kg_expansion = (use_kg_expansion != 0) && (h->config.enable_knowledge_graph != 0);

        auto combined = h->query_engine->query(cq, query_emb);

        auto qh = new EvdbQueryHandle_();
        qh->results = combined.chunks;
        qh->context_string = combined.assembled_context;
        return qh;
    } catch (...) {
        return nullptr;
    }
}

EvdbQueryHandle* evdb_query_vector(EvdbHandle* h,
                                    const float* query_embedding_384,
                                    const char* query_text_for_keyword,
                                    int top_k) {
    if (!h || !query_embedding_384) return nullptr;

    try {
        CombinedQuery cq;
        cq.text_query = query_text_for_keyword ? query_text_for_keyword : "";
        cq.top_k = top_k;
        cq.token_budget = h->config.token_budget;
        cq.use_kg_expansion = (h->config.enable_knowledge_graph != 0);

        auto combined = h->query_engine->query(cq, query_embedding_384);

        auto qh = new EvdbQueryHandle_();
        qh->results = combined.chunks;
        qh->context_string = combined.assembled_context;
        return qh;
    } catch (...) {
        return nullptr;
    }
}

int evdb_result_count(EvdbQueryHandle* q) {
    return q ? static_cast<int>(q->results.size()) : 0;
}

const char* evdb_result_text(EvdbQueryHandle* q, int index) {
    if (!q || index < 0 || index >= static_cast<int>(q->results.size())) return "";
    return q->results[index].text;
}

float evdb_result_score(EvdbQueryHandle* q, int index) {
    if (!q || index < 0 || index >= static_cast<int>(q->results.size())) return 0.0f;
    return q->results[index].score;
}

uint64_t evdb_result_chunk_id(EvdbQueryHandle* q, int index) {
    if (!q || index < 0 || index >= static_cast<int>(q->results.size())) return 0;
    return q->results[index].chunk_id;
}

uint32_t evdb_result_page(EvdbQueryHandle* q, int index) {
    if (!q || index < 0 || index >= static_cast<int>(q->results.size())) return 0;
    return q->results[index].page_number;
}

const char* evdb_result_context_string(EvdbQueryHandle* q) {
    if (!q) return "";
    return q->context_string.c_str();
}

void evdb_query_free(EvdbQueryHandle* q) {
    delete q;
}

// ── Object store ──────────────────────────────────────────

EvdbError evdb_object_put(EvdbHandle* h, const char* type_name,
                           const char* json_properties, uint64_t* out_id) {
    if (!h) return EVDB_ERR_NULL_HANDLE;
    if (!type_name || !json_properties) return EVDB_ERR_INVALID_ARG;

    try {
        auto props = nlohmann::json::parse(json_properties);
        uint64_t id = h->obj_store->put(type_name, props);
        if (out_id) *out_id = id;
        return EVDB_OK;
    } catch (...) {
        return EVDB_ERR_IO;
    }
}

EvdbError evdb_object_get(EvdbHandle* h, uint64_t id, char* out_json, int out_json_size) {
    if (!h) return EVDB_ERR_NULL_HANDLE;
    if (!out_json || out_json_size <= 0) return EVDB_ERR_INVALID_ARG;

    try {
        nlohmann::json obj;
        if (!h->obj_store->get(id, obj)) return EVDB_ERR_NOT_FOUND;
        std::string json_str = obj.dump();
        std::strncpy(out_json, json_str.c_str(), out_json_size - 1);
        out_json[out_json_size - 1] = '\0';
        return EVDB_OK;
    } catch (...) {
        return EVDB_ERR_IO;
    }
}

EvdbError evdb_object_remove(EvdbHandle* h, uint64_t id) {
    if (!h) return EVDB_ERR_NULL_HANDLE;
    try {
        if (!h->obj_store->remove(id)) return EVDB_ERR_NOT_FOUND;
        h->rel_index->removeAllRelationsForRecord(id);
        return EVDB_OK;
    } catch (...) {
        return EVDB_ERR_IO;
    }
}

EvdbError evdb_object_query(EvdbHandle* h, const char* type_name,
                             const char* filter_property, const char* filter_value,
                             char* out_json_array, int out_size, int limit) {
    if (!h) return EVDB_ERR_NULL_HANDLE;
    if (!type_name || !out_json_array) return EVDB_ERR_INVALID_ARG;

    try {
        std::vector<nlohmann::json> results;
        if (filter_property && filter_value) {
            results = h->obj_store->query(type_name, filter_property, filter_value, limit);
        } else {
            results = h->obj_store->getAll(type_name, 0, limit);
        }

        auto arr = nlohmann::json::array();
        for (const auto& r : results) arr.push_back(r);
        std::string json_str = arr.dump();
        std::strncpy(out_json_array, json_str.c_str(), out_size - 1);
        out_json_array[out_size - 1] = '\0';
        return EVDB_OK;
    } catch (...) {
        return EVDB_ERR_IO;
    }
}

// ── Relations ─────────────────────────────────────────────

EvdbError evdb_relation_add(EvdbHandle* h, const char* relation_name,
                             uint64_t from_id, uint64_t to_id) {
    if (!h) return EVDB_ERR_NULL_HANDLE;
    if (!relation_name) return EVDB_ERR_INVALID_ARG;
    try {
        h->rel_index->addRelation(relation_name, from_id, to_id);
        return EVDB_OK;
    } catch (...) {
        return EVDB_ERR_IO;
    }
}

EvdbError evdb_relation_get_targets(EvdbHandle* h, const char* relation_name,
                                     uint64_t from_id,
                                     uint64_t* out_ids, int* inout_count) {
    if (!h) return EVDB_ERR_NULL_HANDLE;
    if (!relation_name || !out_ids || !inout_count) return EVDB_ERR_INVALID_ARG;
    try {
        auto targets = h->rel_index->getTargets(relation_name, from_id);
        int max = *inout_count;
        int count = std::min(static_cast<int>(targets.size()), max);
        for (int i = 0; i < count; i++) {
            out_ids[i] = targets[i];
        }
        *inout_count = count;
        return EVDB_OK;
    } catch (...) {
        return EVDB_ERR_IO;
    }
}

// ── Sync ──────────────────────────────────────────────────

EvdbSyncEngine* evdb_sync_create(EvdbHandle* h, const char* device_id) {
    if (!h) return nullptr;
    try {
        auto s = new EvdbSyncEngine_();
        std::string did = device_id ? device_id : "device-default";
        s->engine = std::make_unique<SyncEngine>(
            did, h->obj_store.get(), h->rel_index.get(), h->chunk_store.get());
        return s;
    } catch (...) {
        return nullptr;
    }
}

void evdb_sync_destroy(EvdbSyncEngine* s) {
    delete s;
}

EvdbError evdb_sync_export_delta(EvdbSyncEngine* s, uint64_t since_clock,
                                  char* out_json, int out_size) {
    if (!s) return EVDB_ERR_NULL_HANDLE;
    if (!out_json) return EVDB_ERR_INVALID_ARG;
    try {
        auto delta = s->engine->exportDelta(since_clock);
        std::string json = delta.serialize();
        std::strncpy(out_json, json.c_str(), out_size - 1);
        out_json[out_size - 1] = '\0';
        return EVDB_OK;
    } catch (...) {
        return EVDB_ERR_SYNC;
    }
}

EvdbError evdb_sync_apply_delta(EvdbSyncEngine* s, const char* delta_json,
                                 int* out_applied, int* out_skipped) {
    if (!s) return EVDB_ERR_NULL_HANDLE;
    if (!delta_json) return EVDB_ERR_INVALID_ARG;
    try {
        auto delta = SyncDelta::deserialize(delta_json);
        auto result = s->engine->applyDelta(delta);
        if (out_applied) *out_applied = result.records_applied;
        if (out_skipped) *out_skipped = result.records_skipped_older;
        return EVDB_OK;
    } catch (...) {
        return EVDB_ERR_SYNC;
    }
}

EvdbError evdb_sync_export_to_file(EvdbSyncEngine* s, const char* path, uint64_t since_clock) {
    if (!s) return EVDB_ERR_NULL_HANDLE;
    if (!path) return EVDB_ERR_INVALID_ARG;
    return s->engine->exportToFile(path, since_clock) ? EVDB_OK : EVDB_ERR_IO;
}

EvdbError evdb_sync_import_from_file(EvdbSyncEngine* s, const char* path) {
    if (!s) return EVDB_ERR_NULL_HANDLE;
    if (!path) return EVDB_ERR_INVALID_ARG;
    MergeResult result;
    return s->engine->importFromFile(path, result) ? EVDB_OK : EVDB_ERR_IO;
}

uint64_t evdb_sync_current_clock(EvdbSyncEngine* s) {
    if (!s) return 0;
    return s->engine->getCurrentClock();
}

// ── Utilities ─────────────────────────────────────────────

const char* evdb_version_string(void) {
    return EDGEVDB_VERSION_STRING;
}

const char* evdb_error_string(EvdbError err) {
    switch (err) {
        case EVDB_OK:              return "OK";
        case EVDB_ERR_NULL_HANDLE: return "Null handle";
        case EVDB_ERR_IO:          return "I/O error";
        case EVDB_ERR_OOM:         return "Out of memory";
        case EVDB_ERR_NOT_FOUND:   return "Not found";
        case EVDB_ERR_INVALID_ARG: return "Invalid argument";
        case EVDB_ERR_ONNX:        return "ONNX Runtime error";
        case EVDB_ERR_SYNC:        return "Sync error";
        default:                   return "Unknown error";
    }
}

void evdb_set_log_level(int level) {
    edgevdb::setLogLevel(static_cast<edgevdb::LogLevel>(level));
}

} // extern "C"
