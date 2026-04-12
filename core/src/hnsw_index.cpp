#include "hnsw_index.hpp"
#include <fstream>
#include <cstring>
#include <chrono>
#include <numeric>
#include <mutex>

namespace edgevdb {

HNSWIndex::HNSWIndex()
    : entry_point_(0), max_layer_(-1),
      M_(16), M0_(32), ef_construction_(200), ef_search_(64),
      visit_counter_(0) {
    mL_ = 1.0 / std::log(static_cast<double>(M_));
    rng_.seed(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
}

HNSWIndex::HNSWIndex(int M, int ef_construction, int ef_search)
    : entry_point_(0), max_layer_(-1),
      M_(M), M0_(2 * M), ef_construction_(ef_construction), ef_search_(ef_search),
      visit_counter_(0) {
    mL_ = 1.0 / std::log(static_cast<double>(M_));
    rng_.seed(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
}

HNSWIndex::HNSWIndex(const std::string& file_path, int M, int ef_construction, int ef_search)
    : entry_point_(0), max_layer_(-1),
      M_(M), M0_(2 * M), ef_construction_(ef_construction), ef_search_(ef_search),
      file_path_(file_path), visit_counter_(0) {
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

std::priority_queue<HNSWIndex::DistPair, std::vector<HNSWIndex::DistPair>, std::less<HNSWIndex::DistPair>>
HNSWIndex::searchLayer(const float* query, uint32_t entry_pt, int ef, int layer) const {
    // Max-heap of candidates (furthest at top)
    std::priority_queue<DistPair, std::vector<DistPair>, std::less<DistPair>> top_candidates;
    // Min-heap of candidates to explore (closest at top)
    std::priority_queue<DistPair, std::vector<DistPair>, std::greater<DistPair>> candidates;

    float dist = cosineDistance(query, getEmbedding(entry_pt), EMBEDDING_DIM);

    // Visited tracking using generation counter
    visit_counter_++;
    if (visited_gen_.size() < nodes_.size()) {
        visited_gen_.resize(nodes_.size(), 0);
    }

    visited_gen_[entry_pt] = visit_counter_;
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

                if (visited_gen_[neighbor_idx] == visit_counter_) continue;
                visited_gen_[neighbor_idx] = visit_counter_;

                float d = cosineDistance(query, getEmbedding(neighbor_idx), EMBEDDING_DIM);

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

std::vector<uint32_t> HNSWIndex::selectNeighbors(
    const std::priority_queue<DistPair, std::vector<DistPair>, std::less<DistPair>>& candidates,
    int M_max) const {

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

void HNSWIndex::insert(uint64_t external_id, const float* embedding) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    // Check if already exists
    if (id_to_index_.count(external_id)) {
        // Update: remove old, insert new
        uint32_t old_idx = id_to_index_[external_id];
        nodes_[old_idx].deleted = true;
        id_to_index_.erase(external_id);
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

    id_to_index_[external_id] = new_idx;

    // Ensure visited_gen_ is large enough
    if (visited_gen_.size() < nodes_.size()) {
        visited_gen_.resize(nodes_.size(), 0);
    }

    // First node: just set as entry point
    if (nodes_.size() == 1 || max_layer_ < 0) {
        entry_point_ = new_idx;
        max_layer_ = new_level;
        return;
    }

    uint32_t curr_entry = entry_point_;

    // Phase 1: Greedy descent from top layer to new_level + 1
    for (int layer = max_layer_; layer > new_level; layer--) {
        auto candidates = searchLayer(embedding, curr_entry, 1, layer);
        if (!candidates.empty()) {
            // Get closest
            std::vector<DistPair> sorted;
            while (!candidates.empty()) {
                sorted.push_back(candidates.top());
                candidates.pop();
            }
            std::sort(sorted.begin(), sorted.end());
            curr_entry = sorted[0].second;
        }
    }

    // Phase 2: Insert at layers min(new_level, max_layer) down to 0
    for (int layer = std::min(new_level, max_layer_); layer >= 0; layer--) {
        auto candidates = searchLayer(embedding, curr_entry, ef_construction_, layer);

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
}

bool HNSWIndex::remove(uint64_t external_id) {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    auto it = id_to_index_.find(external_id);
    if (it == id_to_index_.end()) return false;
    nodes_[it->second].deleted = true;
    id_to_index_.erase(it);
    return true;
}

std::vector<std::pair<uint64_t, float>> HNSWIndex::knnSearch(const float* query, int k, int ef) const {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    if (nodes_.empty() || max_layer_ < 0) return {};

    if (ef < 0) ef = ef_search_;
    if (ef < k) ef = k;

    uint32_t curr_entry = entry_point_;

    // Greedy descent from top to layer 1
    for (int layer = max_layer_; layer > 0; layer--) {
        auto candidates = searchLayer(query, curr_entry, 1, layer);
        if (!candidates.empty()) {
            std::vector<DistPair> sorted;
            while (!candidates.empty()) {
                sorted.push_back(candidates.top());
                candidates.pop();
            }
            std::sort(sorted.begin(), sorted.end());
            curr_entry = sorted[0].second;
        }
    }

    // Search at layer 0
    auto candidates = searchLayer(query, curr_entry, ef, 0);

    // Extract top-k
    std::vector<DistPair> sorted;
    while (!candidates.empty()) {
        sorted.push_back(candidates.top());
        candidates.pop();
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

void HNSWIndex::clear() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);
    nodes_.clear();
    embeddings_.clear();
    id_to_index_.clear();
    entry_point_ = 0;
    max_layer_ = -1;
    visit_counter_ = 0;
    visited_gen_.clear();
}

bool HNSWIndex::rebuildFromChunkStore(const ChunkStore& store) {
    clear();
    std::vector<uint64_t> ids = store.getAllIds();
    for (uint64_t id : ids) {
        ChunkNode chunk;
        if (store.get(id, chunk)) {
            insert(chunk.id, chunk.embedding);
        }
    }
    EVDB_LOG_INFO("HNSWIndex: Rebuilt with %zu nodes", size());
    return true;
}

bool HNSWIndex::save() {
    std::shared_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) return false;

    std::ofstream file(file_path_, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        EVDB_LOG_ERROR("HNSWIndex: Cannot open %s for writing", file_path_.c_str());
        return false;
    }

    // Magic
    const char magic[8] = {'E','V','D','B','H','N','S','\0'};
    file.write(magic, 8);

    // Version
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), 4);

    // Parameters
    file.write(reinterpret_cast<const char*>(&M_), 4);
    file.write(reinterpret_cast<const char*>(&M0_), 4);
    file.write(reinterpret_cast<const char*>(&ef_construction_), 4);

    // Node count (excluding deleted)
    uint32_t node_count = static_cast<uint32_t>(nodes_.size());
    file.write(reinterpret_cast<const char*>(&node_count), 4);

    // Entry point and max layer
    file.write(reinterpret_cast<const char*>(&entry_point_), 4);
    int32_t ml = static_cast<int32_t>(max_layer_);
    file.write(reinterpret_cast<const char*>(&ml), 4);

    // Per node: external_id, level, embedding, then per layer: count + neighbors
    std::vector<uint8_t> crc_data;

    for (uint32_t i = 0; i < node_count; i++) {
        const auto& node = nodes_[i];
        file.write(reinterpret_cast<const char*>(&node.external_id), 8);
        int32_t level = static_cast<int32_t>(node.level);
        file.write(reinterpret_cast<const char*>(&level), 4);
        uint8_t del = node.deleted ? 1 : 0;
        file.write(reinterpret_cast<const char*>(&del), 1);

        // Write embedding
        file.write(reinterpret_cast<const char*>(embeddings_[i].data()), EMBEDDING_DIM * sizeof(float));

        // Per layer
        for (int layer = 0; layer <= node.level; layer++) {
            int32_t count = static_cast<int32_t>(node.neighbors[layer].size());
            file.write(reinterpret_cast<const char*>(&count), 4);
            for (int32_t j = 0; j < count; j++) {
                uint32_t n = node.neighbors[layer][j];
                file.write(reinterpret_cast<const char*>(&n), 4);
            }
        }
    }

    EVDB_LOG_INFO("HNSWIndex: Saved %u nodes to %s", node_count, file_path_.c_str());
    return file.good();
}

bool HNSWIndex::open() {
    std::unique_lock<std::shared_mutex> lock(rw_mutex_);

    if (file_path_.empty()) return true;

    std::ifstream file(file_path_, std::ios::binary);
    if (!file.is_open()) {
        EVDB_LOG_INFO("HNSWIndex: No existing file, starting empty");
        return true;
    }

    char magic[8];
    file.read(magic, 8);
    const char expected_magic[8] = {'E','V','D','B','H','N','S','\0'};
    if (std::memcmp(magic, expected_magic, 8) != 0) {
        EVDB_LOG_ERROR("HNSWIndex: Invalid magic");
        return false;
    }

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), 4);

    file.read(reinterpret_cast<char*>(&M_), 4);
    file.read(reinterpret_cast<char*>(&M0_), 4);
    file.read(reinterpret_cast<char*>(&ef_construction_), 4);
    mL_ = 1.0 / std::log(static_cast<double>(M_));

    uint32_t node_count;
    file.read(reinterpret_cast<char*>(&node_count), 4);
    file.read(reinterpret_cast<char*>(&entry_point_), 4);
    int32_t ml;
    file.read(reinterpret_cast<char*>(&ml), 4);
    max_layer_ = ml;

    nodes_.clear();
    embeddings_.clear();
    id_to_index_.clear();
    nodes_.resize(node_count);
    embeddings_.resize(node_count);

    for (uint32_t i = 0; i < node_count; i++) {
        file.read(reinterpret_cast<char*>(&nodes_[i].external_id), 8);
        int32_t level;
        file.read(reinterpret_cast<char*>(&level), 4);
        nodes_[i].level = level;
        uint8_t del;
        file.read(reinterpret_cast<char*>(&del), 1);
        nodes_[i].deleted = (del != 0);

        // Read embedding
        embeddings_[i].resize(EMBEDDING_DIM);
        file.read(reinterpret_cast<char*>(embeddings_[i].data()), EMBEDDING_DIM * sizeof(float));

        // Read neighbors
        nodes_[i].neighbors.resize(level + 1);
        for (int layer = 0; layer <= level; layer++) {
            int32_t count;
            file.read(reinterpret_cast<char*>(&count), 4);
            nodes_[i].neighbors[layer].resize(count);
            for (int32_t j = 0; j < count; j++) {
                file.read(reinterpret_cast<char*>(&nodes_[i].neighbors[layer][j]), 4);
            }
        }

        if (!nodes_[i].deleted) {
            id_to_index_[nodes_[i].external_id] = i;
        }
    }

    visited_gen_.resize(nodes_.size(), 0);
    visit_counter_ = 0;

    EVDB_LOG_INFO("HNSWIndex: Loaded %u nodes from %s", node_count, file_path_.c_str());
    return true;
}

uint32_t HNSWIndex::computeCRC32(const uint8_t* data, size_t len) const {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; i++) {
        crc = (crc >> 8) ^ ((crc ^ data[i]) & 0xFF);
    }
    return crc ^ 0xFFFFFFFF;
}

} // namespace edgevdb
