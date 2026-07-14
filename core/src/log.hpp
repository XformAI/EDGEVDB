#pragma once

// EdgeVDB Logging macros
// On Android: uses __android_log_print
// On other platforms: uses fprintf(stderr, ...)
// All macros honor the runtime level set via edgevdb::setLogLevel().

#include <cstdio>
#include <cstdarg>

namespace edgevdb {

enum class LogLevel {
    Off = 0,
    Error = 1,
    Info = 2,
    Debug = 3
};

inline LogLevel& globalLogLevel() {
    static LogLevel level = LogLevel::Info;
    return level;
}

inline void setLogLevel(LogLevel level) {
    globalLogLevel() = level;
}

inline bool logEnabled(LogLevel level) {
    return static_cast<int>(level) <= static_cast<int>(globalLogLevel());
}

} // namespace edgevdb

#ifdef __ANDROID__
#include <android/log.h>
#define EVDB_LOG_TAG "EdgeVDB"
#define EVDB_LOG_ERROR(msg, ...) do { if (::edgevdb::logEnabled(::edgevdb::LogLevel::Error)) \
    __android_log_print(ANDROID_LOG_ERROR, EVDB_LOG_TAG, msg, ##__VA_ARGS__); } while (0)
#define EVDB_LOG_INFO(msg, ...)  do { if (::edgevdb::logEnabled(::edgevdb::LogLevel::Info)) \
    __android_log_print(ANDROID_LOG_INFO, EVDB_LOG_TAG, msg, ##__VA_ARGS__); } while (0)
#define EVDB_LOG_DEBUG(msg, ...) do { if (::edgevdb::logEnabled(::edgevdb::LogLevel::Debug)) \
    __android_log_print(ANDROID_LOG_DEBUG, EVDB_LOG_TAG, msg, ##__VA_ARGS__); } while (0)
#else
#define EVDB_LOG_ERROR(msg, ...) do { if (::edgevdb::logEnabled(::edgevdb::LogLevel::Error)) \
    fprintf(stderr, "[EdgeVDB ERROR] " msg "\n", ##__VA_ARGS__); } while (0)
#define EVDB_LOG_INFO(msg, ...)  do { if (::edgevdb::logEnabled(::edgevdb::LogLevel::Info)) \
    fprintf(stderr, "[EdgeVDB INFO]  " msg "\n", ##__VA_ARGS__); } while (0)
#define EVDB_LOG_DEBUG(msg, ...) do { if (::edgevdb::logEnabled(::edgevdb::LogLevel::Debug)) \
    fprintf(stderr, "[EdgeVDB DEBUG] " msg "\n", ##__VA_ARGS__); } while (0)
#endif
