#pragma once

// Shared CRC-32 (ISO 3309 / ITU-T V.42, reflected, poly 0xEDB88320).
// Single source of truth for all store checksums. The table is generated
// at compile time, so it cannot drift or be miscopied.

#include <array>
#include <cstdint>
#include <cstddef>

namespace edgevdb {

namespace detail {

constexpr std::array<uint32_t, 256> makeCrc32Table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) {
            c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        table[i] = c;
    }
    return table;
}

inline constexpr std::array<uint32_t, 256> kCrc32Table = makeCrc32Table();

} // namespace detail

// Incremental form: pass the previous return value as `state` to continue.
// Initial state must be 0xFFFFFFFF; finalize with crc32Finalize().
inline uint32_t crc32Update(uint32_t state, const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; i++) {
        state = detail::kCrc32Table[(state ^ p[i]) & 0xFF] ^ (state >> 8);
    }
    return state;
}

inline constexpr uint32_t CRC32_INIT = 0xFFFFFFFFu;

inline uint32_t crc32Finalize(uint32_t state) {
    return state ^ 0xFFFFFFFFu;
}

// One-shot convenience.
inline uint32_t crc32(const void* data, size_t len) {
    return crc32Finalize(crc32Update(CRC32_INIT, data, len));
}

} // namespace edgevdb
