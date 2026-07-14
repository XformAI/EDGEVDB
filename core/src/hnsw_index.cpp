#include "hnsw_index.hpp"
#include "crc32.hpp"
#include "persist.hpp"

#include <fstream>
#include <cstring>
#include <chrono>
#include <numeric>
#include <mutex>

namespace edgevdb {

HNSWIndex::HNSWIndex()
    : entry_point_(0), max_layer_(-1),
      M_(16), M0_(32), ef_construction_(200), ef_search_(64) {
    mL_ = 1.0 / std::log(static_cast<double>(M_));
    rng_.seed(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
}

HNSWIndex::HNSWIndex(int M, int ef_construction, int ef_search)
    : entry_point_(0), max_layer_(-1),
      M_(M), M0_(2 * M), ef_construction_(ef_construction), ef_search_(ef_search) {
    mL_ = 1.0 / std::log(static_cast<double>(M_));
    rng_.seed(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
}

HNSWIndex::HNSWIndex(const std::string& file_path, int M, int ef_construction, int ef_search)
    : entry_point_(0), max_layer_(-1),
      M_(M), M0_(2 * M), ef_construction_(ef_construction), ef_search_(ef_search),
      file_path_(file_path) {
    mL_ = 1.0 / std::log(static_cast<double>(M_));
    rng_.seed(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
}

int HNSWIndex::randomLevel() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng_);
    if (r == 0.0) r = 1e-10;
    int level = static_cast<int>(-std::log(r) * mL_);
    return std::min(level, 32); // cap at 32 layers
}

const float* HNSWIndex::getEmbedding(uint32_t internal_idx) const {
    return embeddings_[internal_idx].data();
}

// ── Quantization ──────────────────────────────────────────

void HNSWIndex::quantizeVector(const float* v, std::vector<int8_t>& code, float& scale) const {
    float max_abs = 0.0f;
    for (size_t d = 0; d < EMBEDDING_DIM; d++) {
        float a = std::fabs(v[d]);
        if (a > max_abs) max_abs = a;
    }
    scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    code.resize(EMBEDDING_DIM);
    const float inv = 1.0f / scale;
    for (size_t d = 0; d < EMBEDDING_DIM; d++) {
        int q = static_cast<int>(std::lround(v[d] * inv));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        code[d] = static_cast<int8_t>(q);
    }
}

void HNSWIndex::setQuantizedSearch(bool enabled) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    quantized_ = enabled;
    codes_.clear();
    scales_.clear();
    if (enabled) {
        codes_.resize(embeddings_.size());
        scales_.resize(embeddings_.size());
        for (size_t i = 0; i < embeddings_.size(); i++) {
            quantizeVector(embeddings_[i].data(), codes_[i], scales_[i]);
        }
    }
}

float HNSWIndex::SearchContext::distanceTo(uint32_t idx) const {
    if (quantized) {
        int32_t dot = dotProductInt8(q_code.data(), index->codes_[idx].data(), EMBEDDING_DIM);
        return 1.0f - static_cast<float>(dot) * q_scale * index->scales_[idx];
    }
    return cosineDistance(query, index->getEmbedding(idx), EMBEDDING_DIM);
}

HNSWIndex::SearchContext HNSWIndex::makeContext(const float* query) const {
    SearchContext ctx;
    ctx.query = query;
    ctx.index = this;
    ctx.quantized = quantized_ && !codes_.empty();
    if (ctx.quantized) {
        quantizeVector(query, ctx.q_code, ctx.q_scale);
    } else {
        ctx.q_scale = 1.0f;
    }
    return ctx;
}

// ── Core traversal ────────────────────────────────────────

HNSWIndex::MaxHeap
HNSWIndex::searchLayer(const SearchContext& ctx, uint32_t entry_pt, int ef, int layer,
                       VisitedList& vl) const {
    // Max-heap of results (furthest at top)
    MaxHeap top_candidates;
    // Min-heap of candidates to explore (closest at top)
    std::priority_queue<DistPair, std::vector<DistPair>, std::greater<DistPair>> candidates;

    float dist = ctx.distanceTo(entry_pt);

    vl.mark(entry_pt);
    candidates.push({dist, entry_pt});
    top_candidates.push({dist, entry_pt});

    while (!candidates.empty()) {
        auto [cand_dist, cand_idx] = candidates.top();
        candidates.pop();

        // If candidate is further than the furthest in top_candidates, we're done
        if (cand_dist > top_candidates.top().first && top_candidates.size() >= static_cast<size_t>(ef)) {
            break;
        }

        // Explore neighbors of candidate at this layer
        if (layer < static_cast<int>(nodes_[cand_idx].neighbors.size())) {
            for (uint32_t neighbor_idx : nodes_[cand_idx].neighbors[layer]) {
                if (neighbor_idx >= nodes_.size()) continue;
                if (nodes_[neighbor_idx].deleted) continue;

                if (vl.visited(neighbor_idx)) continue;
                vl.mark(neighbor_idx);

                float d = ctx.distanceTo(neighbor_idx);

                if (top_candidates.size() < static_cast<size_t>(ef) || d < top_candidates.top().first) {
                    candidates.push({d, neighbor_idx});
                    top_candidates.push({d, neighbor_idx});
                    if (top_candidates.size() > static_cast<size_t>(ef)) {
                        top_candidates.pop();
                    }
                }
            }
        }
    }

    return top_candidates;
}

uint32_t HNSWIndex::greedyDescend(const SearchContext& ctx, uint32_t entry, int from_layer,
                                  int to_layer, VisitedList& vl) const {
    uint32_t curr = entry;
    for (int layer = from_layer; layer > to_layer; layer--) {
        vl.prepare(nodes_.size());
        auto candidates = searchLayer(ctx, curr, 1, layer, vl);
        DistPair best{std::numeric_limits<float>::max(), curr};
        while (!candidates.empty()) {
            if (candidates.top().first < best.first) best = candidates.top();
            candidates.pop();
        }
        curr = best.second;
    }
    return curr;
}

// ── Neighbor selection ────────────────────────────────────

std::vector<uint32_t> HNSWIndex::selectNeighborsSimple(const MaxHeap& candidates, int M_max) const {
    // Copy to vector and sort by distance (closest first)
    auto tmp = candidates;
    std::vector<DistPair> sorted;
    while (!tmp.empty()) {
        sorted.push_back(tmp.top());
        tmp.pop();
    }
    std::sort(sorted.begin(), sorted.end());

    std::vector<uint32_t> result;
    result.reserve(M_max);
    for (const auto& [dist, idx] : sorted) {
        if (static_cast<int>(result.size()) >= M_max) break;
        result.push_back(idx);
    }
    return result;
}

std::vector<uint32_t> HNSWIndex::selectNeighborsHeuristic(const MaxHeap& candidates, int M_max) const {
    auto tmp = candidates;
    std::vector<DistPair> sorted;
    while (!tmp.empty()) {
        sorted.push_back(tmp.top());
        tmp.pop();
    }
    std::sort(sorted.begin(), sorted.end());

    // Algorithm 4 (Malkov & Yashunin): keep candidate c only when it is
    // closer to the query point than to every already-selected neighbor.
    // This spreads connections across clusters instead of packing them
    // into the nearest one, improving recall on clustered data.
    std::vector<uint32_t> selected;
    std::vector<DistPair> pruned;
    selected.reserve(M_max);

    for (const auto& [dist_to_query, idx] : sorted) {
        if (static_cast<int>(selected.size()) >= M_max) break;
        bool keep = true;
        for (uint32_t s : selected) {
            float dist_to_selected = cosineDistance(getEmbedding(idx), getEmbedding(s), EMBEDDING_DIM);
            if (dist_to_selected < dist_to_query) {
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.push_back(idx);
        } else {
            pruned.push_back({dist_to_query, idx});
        }
    }

    // keepPrunedConnections: fill remaining slots with the closest pruned
    // candidates so nodes are never left under-connected.
    for (const auto& [dist, idx] : pruned) {
        if (static_cast<int>(selected.size()) >= M_max) break;
        selected.push_back(idx);
    }

    return selected;
}

std::vector<uint32_t> HNSWIndex::selectNeighbors(const MaxHeap& candidates, int M_max) const {
    return use_heuristic_ ? selectNeighborsHeuristic(candidates, M_max)
                          : selectNeighborsSimple(candidates, M_max);
}

// ── Insert / remove ───────────────────────────────────────

void HNSWIndex::insert(uint64_t external_id, const float* embedding) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    insertUnlocked(external_id, embedding);
}

void HNSWIndex::insertUnlocked(uint64_t external_id, const float* embedding) {
    // Check if already exists
    auto existing = id_to_index_.find(external_id);
    if (existing != id_to_index_.end()) {
        // Update: remove old, insert new
        nodes_[existing->second].deleted = true;
        deleted_count_++;
        id_to_index_.erase(existing);
    }

    uint32_t new_idx = static_cast<uint32_t>(nodes_.size());
    int new_level = randomLevel();

    HNSWNode node;
    node.external_id = external_id;
    node.level = new_level;
    node.neighbors.resize(new_level + 1);
    node.deleted = false;
    nodes_.push_back(node);

    // Store embedding
    std::vector<float> emb(embedding, embedding + EMBEDDING_DIM);
    embeddings_.push_back(std::move(emb));

    if (quantized_) {
        codes_.emplace_back();
        scales_.emplace_back(1.0f);
        quantizeVector(embedding, codes_.back(), scales_.back());
    }

    id_to_index_[external_id] = new_idx;

    // First node: just set as entry point
    if (nodes_.size() == 1 || max_layer_ < 0) {
        entry_point_ = new_idx;
        max_layer_ = new_level;
        return;
    }

    // Construction always uses exact float distances: graph quality is
    // decided here, and insert throughput is not the quantized path's goal.
    SearchContext ctx;
    ctx.query = embedding;
    ctx.index = this;
    ctx.quantized = false;
    ctx.q_scale = 1.0f;

    VisitedListGuard guard(visited_pool_);
    VisitedList& vl = guard.get();

    // Phase 1: Greedy descent from top layer to new_level + 1
    uint32_t curr_entry = greedyDescend(ctx, entry_point_, max_layer_, new_level, vl);

    // Phase 2: Insert at layers min(new_level, max_layer) down to 0
    for (int layer = std::min(new_level, max_layer_); layer >= 0; layer--) {
        vl.prepare(nodes_.size());
        auto candidates = searchLayer(ctx, curr_entry, ef_construction_, layer, vl);

        int M_max = (layer == 0) ? M0_ : M_;
        auto neighbors = selectNeighbors(candidates, M_max);

        // Set bidirectional links
        nodes_[new_idx].neighbors[layer] = neighbors;
        for (uint32_t neighbor_idx : neighbors) {
            if (layer < static_cast<int>(nodes_[neighbor_idx].neighbors.size())) {
                nodes_[neighbor_idx].neighbors[layer].push_back(new_idx);

                // Trim if over capacity
                int neighbor_M_max = (layer == 0) ? M0_ : M_;
                if (static_cast<int>(nodes_[neighbor_idx].neighbors[layer].size()) > neighbor_M_max) {
                    // Re-select: keep closest
                    std::vector<DistPair> pairs;
                    for (uint32_t n : nodes_[neighbor_idx].neighbors[layer]) {
                        float d = cosineDistance(getEmbedding(neighbor_idx), getEmbedding(n), EMBEDDING_DIM);
                        pairs.push_back({d, n});
                    }
                    std::sort(pairs.begin(), pairs.end());
                    nodes_[neighbor_idx].neighbors[layer].clear();
                    for (int i = 0; i < neighbor_M_max && i < static_cast<int>(pairs.size()); i++) {
                        nodes_[neighbor_idx].neighbors[layer].push_back(pairs[i].second);
                    }
                }
            }
        }

        // Update entry for next layer's search
        if (!neighbors.empty()) {
            curr_entry = neighbors[0];
        }
    }

    // Update entry point if new node has higher level
    if (new_level > max_layer_) {
        entry_point_ = new_idx;
        max_layer_ = new_level;
    }

    maybeCompactUnlocked();
}

bool HNSWIndex::remove(uint64_t external_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = id_to_index_.find(external_id);
    if (it == id_to_index_.end()) return false;
    nodes_[it->second].deleted = true;
    deleted_count_++;
    id_to_index_.erase(it);
    maybeCompactUnlocked();
    return true;
}

void HNSWIndex::maybeCompactUnlocked() {
    if (deleted_count_ < COMPACT_FLOOR) return;
    if (static_cast<double>(deleted_count_) <= COMPACT_THRESHOLD * nodes_.size()) return;

    // Rebuild the graph from live nodes only. Tombstones otherwise stay in
    // the node array, embeddings and neighbor lists forever, and update-by-
    // reinsert would grow memory without bound on mutable workloads.
    std::vector<std::pair<uint64_t, std::vector<float>>> live;
    live.reserve(id_to_index_.size());
    for (uint32_t i = 0; i < nodes_.size(); i++) {
        if (!nodes_[i].deleted) {
            live.emplace_back(nodes_[i].external_id, std::move(embeddings_[i]));
        }
    }

    clearUnlocked();

    for (auto& [id, emb] : live) {
        insertUnlocked(id, emb.data());
    }
    EVDB_LOG_INFO("HNSWIndex: Compacted to %zu live nodes", nodes_.size());
}

// ── Query ─────────────────────────────────────────────────

std::vector<std::pair<uint64_t, float>> HNSWIndex::knnSearch(const float* query, int k, int ef) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    if (nodes_.empty() || max_layer_ < 0) return {};

    if (ef < 0) ef = ef_search_;
    if (ef < k) ef = k;

    SearchContext ctx = makeContext(query);

    VisitedListGuard guard(visited_pool_);
    VisitedList& vl = guard.get();

    // Greedy descent from top to layer 0's entry
    uint32_t curr_entry = greedyDescend(ctx, entry_point_, max_layer_, 0, vl);

    // Quantized traversal is noisier per-distance, so it over-fetches and
    // relies on the exact float re-rank below to recover full precision.
    // int8 distances are ~4x cheaper, so a 2x beam still wins on cost.
    int ef_traverse = ctx.quantized ? ef * 2 : ef;

    // Search at layer 0
    vl.prepare(nodes_.size());
    auto candidates = searchLayer(ctx, curr_entry, ef_traverse, 0, vl);

    // Adaptive ef: estimate query difficulty from the spread of the top-k
    // distances. When the k-th result is nearly as close as the 1st the
    // neighborhood is ambiguous, so one wider re-search buys recall; easy
    // queries keep the cheap base beam.
    if (adaptive_ef_) {
        std::vector<float> dists;
        auto tmp = candidates;
        while (!tmp.empty()) { dists.push_back(tmp.top().first); tmp.pop(); }
        std::sort(dists.begin(), dists.end());
        if (static_cast<int>(dists.size()) >= k && k > 1) {
            float d1 = dists[0];
            float dk = dists[k - 1];
            float gap = (dk - d1) / std::max(d1, 1e-6f);
            if (gap < 0.15f) {
                int boosted_ef = std::min(ef * 2, 4 * ef_search_);
                if (boosted_ef > ef) {
                    vl.prepare(nodes_.size());
                    candidates = searchLayer(ctx, curr_entry, boosted_ef, 0, vl);
                }
            }
        }
    }

    // Extract candidates
    std::vector<DistPair> sorted;
    while (!candidates.empty()) {
        sorted.push_back(candidates.top());
        candidates.pop();
    }

    // Quantized traversal is approximate: re-rank the surviving candidates
    // with exact float distances before returning.
    if (ctx.quantized) {
        for (auto& [dist, idx] : sorted) {
            dist = cosineDistance(query, getEmbedding(idx), EMBEDDING_DIM);
        }
    }
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::pair<uint64_t, float>> results;
    for (const auto& [dist, idx] : sorted) {
        if (static_cast<int>(results.size()) >= k) break;
        if (!nodes_[idx].deleted) {
            results.push_back({nodes_[idx].external_id, dist});
        }
    }
    return results;
}

size_t HNSWIndex::size() const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);
    return id_to_index_.size();
}

void HNSWIndex::setEfSearch(int ef) {
    ef_search_ = ef;
}

void HNSWIndex::clearUnlocked() {
    nodes_.clear();
    embeddings_.clear();
    id_to_index_.clear();
    codes_.clear();
    scales_.clear();
    entry_point_ = 0;
    max_layer_ = -1;
    deleted_count_ = 0;
}

void HNSWIndex::clear() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    clearUnlocked();
}

bool HNSWIndex::rebuildFromChunkStore(const ChunkStore& store) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    clearUnlocked();
    std::vector<uint64_t> ids = store.getAllIds();
    for (uint64_t id : ids) {
        ChunkNode chunk;
        if (store.get(id, chunk)) {
            insertUnlocked(chunk.id, chunk.embedding);
        }
    }
    EVDB_LOG_INFO("HNSWIndex: Rebuilt with %zu nodes", id_to_index_.size());
    return true;
}

// ── Persistence ───────────────────────────────────────────

bool HNSWIndex::save() {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) return false;

    bool ok = persist::atomicSave(file_path_, [this](std::ofstream& file) {
        const char magic[8] = {'E','V','D','B','H','N','S','\0'};
        file.write(magic, 8);
        file.write(reinterpret_cast<const char*>(&FILE_VERSION), 4);

        uint32_t crc_state = CRC32_INIT;
        auto put = [&](const void* data, size_t len) {
            file.write(static_cast<const char*>(data), static_cast<std::streamsize>(len));
            crc_state = crc32Update(crc_state, data, len);
        };

        put(&M_, 4);
        put(&M0_, 4);
        put(&ef_construction_, 4);

        uint32_t node_count = static_cast<uint32_t>(nodes_.size());
        put(&node_count, 4);
        put(&entry_point_, 4);
        int32_t ml = static_cast<int32_t>(max_layer_);
        put(&ml, 4);

        for (uint32_t i = 0; i < node_count; i++) {
            const auto& node = nodes_[i];
            put(&node.external_id, 8);
            int32_t level = static_cast<int32_t>(node.level);
            put(&level, 4);
            uint8_t del = node.deleted ? 1 : 0;
            put(&del, 1);

            put(embeddings_[i].data(), EMBEDDING_DIM * sizeof(float));

            for (int layer = 0; layer <= node.level; layer++) {
                int32_t count = static_cast<int32_t>(node.neighbors[layer].size());
                put(&count, 4);
                for (int32_t j = 0; j < count; j++) {
                    uint32_t n = node.neighbors[layer][j];
                    put(&n, 4);
                }
            }
        }

        uint32_t crc = crc32Finalize(crc_state);
        file.write(reinterpret_cast<const char*>(&crc), 4);
        return file.good();
    });

    if (ok) {
        EVDB_LOG_INFO("HNSWIndex: Saved %zu nodes to %s", nodes_.size(), file_path_.c_str());
    }
    return ok;
}

bool HNSWIndex::open() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) return true;

    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) {
        EVDB_LOG_INFO("HNSWIndex: No existing file, starting empty");
        return true;
    }

    const uint64_t total_size = persist::fileSize(file);

    auto fail = [this](const char* what) {
        EVDB_LOG_ERROR("HNSWIndex: Corrupt file (%s) — refusing to load", what);
        clearUnlocked();
        return false;
    };

    char magic[8];
    file.read(magic, 8);
    const char expected_magic[8] = {'E','V','D','B','H','N','S','\0'};
    if (std::memcmp(magic, expected_magic, 8) != 0) {
        EVDB_LOG_ERROR("HNSWIndex: Invalid magic");
        return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);
    if (version != FILE_VERSION) {
        EVDB_LOG_ERROR("HNSWIndex: Unsupported version %u (expected %u) — rebuild the index",
                       version, FILE_VERSION);
        return false;
    }

    uint32_t crc_state = CRC32_INIT;
    auto get = [&](void* data, size_t len) -> bool {
        file.read(static_cast<char*>(data), static_cast<std::streamsize>(len));
        if (!file.good()) return false;
        crc_state = crc32Update(crc_state, data, len);
        return true;
    };

    if (!get(&M_, 4) || !get(&M0_, 4) || !get(&ef_construction_, 4)) return fail("params");
    if (M_ <= 0 || M_ > 1024 || M0_ <= 0 || M0_ > 2048) return fail("param range");
    mL_ = 1.0 / std::log(static_cast<double>(M_));

    uint32_t node_count;
    int32_t ml;
    if (!get(&node_count, 4) || !get(&entry_point_, 4) || !get(&ml, 4)) return fail("header");

    // Each node occupies at least 8+4+1 + embedding bytes on disk.
    const uint64_t MIN_NODE_BYTES = 8 + 4 + 1 + EMBEDDING_DIM * sizeof(float);
    if (!persist::boundedCount(node_count, MIN_NODE_BYTES, total_size)) return fail("node count");
    if (node_count > 0 && entry_point_ >= node_count) return fail("entry point");

    clearUnlocked();
    max_layer_ = ml;
    nodes_.resize(node_count);
    embeddings_.resize(node_count);

    for (uint32_t i = 0; i < node_count; i++) {
        int32_t level;
        uint8_t del;
        if (!get(&nodes_[i].external_id, 8) || !get(&level, 4) || !get(&del, 1))
            return fail("node header");
        if (level < 0 || level > 32) return fail("node level");
        nodes_[i].level = level;
        nodes_[i].deleted = (del != 0);
        if (nodes_[i].deleted) deleted_count_++;

        embeddings_[i].resize(EMBEDDING_DIM);
        if (!get(embeddings_[i].data(), EMBEDDING_DIM * sizeof(float))) return fail("embedding");

        nodes_[i].neighbors.resize(level + 1);
        for (int layer = 0; layer <= level; layer++) {
            int32_t count;
            if (!get(&count, 4)) return fail("neighbor count");
            if (count < 0 || !persist::boundedCount(static_cast<uint64_t>(count), 4, total_size))
                return fail("neighbor count bound");
            nodes_[i].neighbors[layer].resize(count);
            for (int32_t j = 0; j < count; j++) {
                if (!get(&nodes_[i].neighbors[layer][j], 4)) return fail("neighbor id");
                if (nodes_[i].neighbors[layer][j] >= node_count) return fail("neighbor range");
            }
        }

        if (!nodes_[i].deleted) {
            id_to_index_[nodes_[i].external_id] = i;
        }
    }

    uint32_t stored_crc;
    file.read(reinterpret_cast<char*>(&stored_crc), 4);
    if (!file.good() || crc32Finalize(crc_state) != stored_crc) return fail("CRC mismatch");

    if (quantized_) {
        codes_.resize(embeddings_.size());
        scales_.resize(embeddings_.size());
        for (size_t i = 0; i < embeddings_.size(); i++) {
            quantizeVector(embeddings_[i].data(), codes_[i], scales_[i]);
        }
    }

    EVDB_LOG_INFO("HNSWIndex: Loaded %u nodes from %s", node_count, file_path_.c_str());
    return true;
}

} // namespace edgevdb
