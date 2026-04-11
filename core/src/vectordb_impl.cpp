/**
 * @file vectordb_impl.cpp
 * @brief Main VectorDB class that ties all components together
 */

#include "schema.hpp"
#include "chunk_store.hpp"
#include "hnsw_index.hpp"
#include "page_index.hpp"
#include "hybrid_ranker.hpp"
#include "kg_extractor.hpp"
#include "knowledge_graph.hpp"
#include "kg_expander.hpp"
#include "embedder.hpp"
#include "tokenizer.hpp"
#include "object_store.hpp"
#include "relation_index.hpp"
#include "query_engine.hpp"
#include "sync_engine.hpp"
#include "token_budget.hpp"
#include "log.hpp"

// This file serves as the compilation unit that includes all headers
// to verify they compile correctly together.
// The actual VectorDB implementation is in vectordb_c_api.cpp via EvdbHandle_.
