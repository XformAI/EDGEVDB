#pragma once

#include "schema.hpp"
#include "log.hpp"

#include <string>
#include <vector>

namespace edgevdb {

class TokenBudget {
public:
    TokenBudget(int max_tokens = 3200, float chars_per_token = 3.5f);

    std::string assembleContext(const std::vector<QueryResult>& results,
                                 bool& truncated_out) const;

    void setMaxTokens(int max_tokens) { max_tokens_ = max_tokens; }
    void setCharsPerToken(float ratio) { chars_per_token_ = ratio; }

private:
    int max_tokens_;
    float chars_per_token_;

    int estimateTokens(const std::string& text) const;
};

// Inline implementation
inline TokenBudget::TokenBudget(int max_tokens, float chars_per_token)
    : max_tokens_(max_tokens), chars_per_token_(chars_per_token) {}

inline int TokenBudget::estimateTokens(const std::string& text) const {
    return static_cast<int>(static_cast<float>(text.size()) / chars_per_token_);
}

inline std::string TokenBudget::assembleContext(const std::vector<QueryResult>& results,
                                                  bool& truncated_out) const {
    truncated_out = false;
    std::string context;
    int total_tokens = 0;

    for (const auto& r : results) {
        std::string entry = "[Page " + std::to_string(r.page_number) + "] " +
                           std::string(r.text) + "\n\n";
        int entry_tokens = estimateTokens(entry);

        if (total_tokens + entry_tokens > max_tokens_) {
            truncated_out = true;
            // Try to fit partial
            int remaining_tokens = max_tokens_ - total_tokens;
            if (remaining_tokens > 10) {
                int remaining_chars = static_cast<int>(remaining_tokens * chars_per_token_);
                if (remaining_chars > 0 && static_cast<size_t>(remaining_chars) < entry.size()) {
                    context += entry.substr(0, remaining_chars);
                }
            }
            context += "\n[Context truncated for length]";
            break;
        }

        context += entry;
        total_tokens += entry_tokens;
    }

    return context;
}

} // namespace edgevdb
