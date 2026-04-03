#pragma once

// EdgeVDB Logging macros
// On Android: uses __android_log_print
// On other platforms: uses fprintf(stderr, ...)

#include <cstdio>
#include <cstdarg>

#ifdef __ANDROID__
#include <android/log.h>
#define EVDB_LOG_TAG "EdgeVDB"
#define EVDB_LOG_ERROR(msg, ...) __android_log_print(ANDROID_LOG_ERROR, EVDB_LOG_TAG, msg, ##__VA_ARGS__)
#define EVDB_LOG_INFO(msg, ...)  __android_log_print(ANDROID_LOG_INFO, EVDB_LOG_TAG, msg, ##__VA_ARGS__)
#define EVDB_LOG_DEBUG(msg, ...) __android_log_print(ANDROID_LOG_DEBUG, EVDB_LOG_TAG, msg, ##__VA_ARGS__)
#else
#define EVDB_LOG_ERROR(msg, ...) fprintf(stderr, "[EdgeVDB ERROR] " msg "\n", ##__VA_ARGS__)
#define EVDB_LOG_INFO(msg, ...)  fprintf(stderr, "[EdgeVDB INFO]  " msg "\n", ##__VA_ARGS__)
#define EVDB_LOG_DEBUG(msg, ...) fprintf(stderr, "[EdgeVDB DEBUG] " msg "\n", ##__VA_ARGS__)
#endif

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

} // namespace edgevdb
