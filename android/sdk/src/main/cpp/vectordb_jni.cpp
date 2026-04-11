/**
 * vectordb_jni.cpp
 *
 * JNI bridge between Kotlin EdgeVDB class and the EdgeVDB C API.
 * Method naming: Java_ai_edgevdb_EdgeVDB_<methodName>
 */

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>

#include "edgevdb/vectordb.h"  // Stable C API

#define LOG_TAG "EdgeVDB-JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ── Helper: jstring → std::string ────────────────────────────────────
static std::string jstringToStd(JNIEnv* env, jstring js) {
    if (!js) return {};
    const char* chars = env->GetStringUTFChars(js, nullptr);
    std::string result(chars ? chars : "");
    env->ReleaseStringUTFChars(js, chars);
    return result;
}

// ── Helper: throw a Java RuntimeException ────────────────────────────
static void throwJavaException(JNIEnv* env, const char* msg) {
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls) env->ThrowNew(cls, msg);
    env->DeleteLocalRef(cls);
}

// ── Helper: jfloatArray → std::vector<float> ─────────────────────────
static std::vector<float> jfloatArrayToVec(JNIEnv* env, jfloatArray arr) {
    jsize len = env->GetArrayLength(arr);
    std::vector<float> vec(len);
    env->GetFloatArrayRegion(arr, 0, len, vec.data());
    return vec;
}

// ── Helper: escape JSON string ───────────────────────────────────────
static std::string escapeJson(const char* s) {
    if (!s) return "";
    std::string out;
    for (char c : std::string(s)) {
        if (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────

extern "C" JNIEXPORT jlong JNICALL
Java_ai_edgevdb_EdgeVDB_nativeOpen(
        JNIEnv* env, jclass /*clazz*/,
        jstring jPath, jint dims,
        jboolean enableKG, jboolean enableOnnx,
        jstring jModelPath, jstring jVocabPath)
{
    EvdbConfig config;
    evdb_default_config(&config);

    std::string path      = jstringToStd(env, jPath);
    std::string modelPath = jstringToStd(env, jModelPath);
    std::string vocabPath = jstringToStd(env, jVocabPath);

    config.storage_dir           = path.c_str();
    config.hnsw_M                = 16;
    config.hnsw_ef_construction  = 200;
    config.hnsw_ef_search        = 64;
    config.enable_knowledge_graph = enableKG ? 1 : 0;

    EvdbHandle* handle = evdb_open(&config);
    if (!handle) {
        throwJavaException(env, "evdb_open() failed — check db_path and model files");
        return 0;
    }
    LOGI("Database opened at path: %s", path.c_str());
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_edgevdb_EdgeVDB_nativeClose(JNIEnv* /*env*/, jclass /*clazz*/, jlong handle)
{
    if (handle) evdb_close(reinterpret_cast<EvdbHandle*>(handle));
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_edgevdb_EdgeVDB_nativeSave(JNIEnv* env, jclass /*clazz*/, jlong handle)
{
    int rc = static_cast<int>(evdb_save(reinterpret_cast<EvdbHandle*>(handle)));
    if (rc != EVDB_OK) LOGE("evdb_save() failed: %d", rc);
    return rc;
}

// ─────────────────────────────────────────────────────────────────────
// Vector Store — Insert
// ─────────────────────────────────────────────────────────────────────

extern "C" JNIEXPORT jlong JNICALL
Java_ai_edgevdb_EdgeVDB_nativeInsertText(
        JNIEnv* env, jclass /*clazz*/, jlong handle,
        jstring jText, jstring jMeta)
{
    auto h    = reinterpret_cast<EvdbHandle*>(handle);
    std::string text = jstringToStd(env, jText);
    std::string meta = jstringToStd(env, jMeta);

    // Use the built-in embedder path if available
    EvdbEmbedder* embedder = nullptr;  // placeholder — C++ embedder not used in Kotlin path
    uint64_t chunkId = 0;
    EvdbError rc = evdb_insert_chunk(h, text.c_str(), nullptr, 0, 0, &chunkId);
    if (rc != EVDB_OK) {
        throwJavaException(env, ("evdb_insert_text failed: " + std::to_string(rc)).c_str());
        return -1;
    }
    return static_cast<jlong>(chunkId);
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_edgevdb_EdgeVDB_nativeInsertChunk(
        JNIEnv* env, jclass /*clazz*/, jlong handle,
        jstring jText, jfloatArray jEmbedding, jstring jMeta)
{
    auto h         = reinterpret_cast<EvdbHandle*>(handle);
    std::string text = jstringToStd(env, jText);
    std::string meta = jstringToStd(env, jMeta);
    auto embedding   = jfloatArrayToVec(env, jEmbedding);

    uint64_t chunkId = 0;
    EvdbError rc = evdb_insert_chunk(h, text.c_str(), embedding.data(), 0, 0, &chunkId);
    if (rc != EVDB_OK) {
        throwJavaException(env, ("evdb_insert_chunk failed: " + std::to_string(rc)).c_str());
        return -1;
    }
    return static_cast<jlong>(chunkId);
}

// ─────────────────────────────────────────────────────────────────────
// Vector Store — Query
// ─────────────────────────────────────────────────────────────────────

extern "C" JNIEXPORT jstring JNICALL
Java_ai_edgevdb_EdgeVDB_nativeQueryText(
        JNIEnv* env, jclass /*clazz*/, jlong handle,
        jstring jQuery, jint topK)
{
    auto h        = reinterpret_cast<EvdbHandle*>(handle);
    std::string query = jstringToStd(env, jQuery);

    // For text query, we need an embedder — return empty if not available
    // In the Kotlin pipeline path, queryVector is used instead
    return env->NewStringUTF("[]");
}

/**
 * Returns a JSON array string of ChunkResult objects.
 * Format: [{"id":1,"text":"...","score":0.87,"meta":"...","docId":"","page":0},...]
 */
extern "C" JNIEXPORT jstring JNICALL
Java_ai_edgevdb_EdgeVDB_nativeQueryVector(
        JNIEnv* env, jclass /*clazz*/, jlong handle,
        jfloatArray jEmbedding, jint topK)
{
    auto h         = reinterpret_cast<EvdbHandle*>(handle);
    auto embedding = jfloatArrayToVec(env, jEmbedding);

    EvdbQueryHandle* qh = evdb_query_vector(h, embedding.data(), "", topK);
    if (!qh) {
        return env->NewStringUTF("[]");
    }

    int count = evdb_result_count(qh);

    std::ostringstream json;
    json << "[";
    for (int i = 0; i < count; ++i) {
        const char* text = evdb_result_text(qh, i);
        float score = evdb_result_score(qh, i);
        uint64_t chunkId = evdb_result_chunk_id(qh, i);
        uint32_t page = evdb_result_page(qh, i);

        if (i > 0) json << ",";
        json << "{"
             << "\"id\":"     << chunkId  << ","
             << "\"text\":\"" << escapeJson(text) << "\","
             << "\"score\":"  << score     << ","
             << "\"meta\":\"" << "" << "\","
             << "\"docId\":\"" << "" << "\","
             << "\"page\":"   << page
             << "}";
    }
    json << "]";

    evdb_query_free(qh);
    return env->NewStringUTF(json.str().c_str());
}

// ─────────────────────────────────────────────────────────────────────
// Object Store
// ─────────────────────────────────────────────────────────────────────

extern "C" JNIEXPORT jint JNICALL
Java_ai_edgevdb_EdgeVDB_nativeObjectPut(
        JNIEnv* env, jclass /*clazz*/, jlong handle,
        jstring jCollection, jstring jId, jstring jJson)
{
    auto h      = reinterpret_cast<EvdbHandle*>(handle);
    std::string collection = jstringToStd(env, jCollection);
    std::string id         = jstringToStd(env, jId);
    std::string jsonStr    = jstringToStd(env, jJson);

    uint64_t outId = 0;
    EvdbError rc = evdb_object_put(h, collection.c_str(), jsonStr.c_str(), &outId);
    return static_cast<jint>(rc);
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_edgevdb_EdgeVDB_nativeObjectGet(
        JNIEnv* env, jclass /*clazz*/, jlong handle,
        jstring jCollection, jstring jId)
{
    auto h      = reinterpret_cast<EvdbHandle*>(handle);
    std::string collection = jstringToStd(env, jCollection);
    std::string id         = jstringToStd(env, jId);

    // Use object ID parsed from string
    uint64_t objectId = 0;
    try { objectId = std::stoull(id); } catch (...) { return env->NewStringUTF("{}"); }

    char buffer[4096] = {};
    EvdbError rc = evdb_object_get(h, objectId, buffer, sizeof(buffer));
    if (rc != EVDB_OK) return env->NewStringUTF("{}");
    return env->NewStringUTF(buffer);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_edgevdb_EdgeVDB_nativeRelationAdd(
        JNIEnv* env, jclass /*clazz*/, jlong handle,
        jstring jFromCol, jstring jFromId,
        jstring jRelType,
        jstring jToCol,   jstring jToId)
{
    auto h = reinterpret_cast<EvdbHandle*>(handle);
    std::string fromId   = jstringToStd(env, jFromId);
    std::string relType  = jstringToStd(env, jRelType);
    std::string toId     = jstringToStd(env, jToId);

    uint64_t from = 0, to = 0;
    try { from = std::stoull(fromId); } catch (...) { return -1; }
    try { to = std::stoull(toId); } catch (...) { return -1; }

    EvdbError rc = evdb_relation_add(h, relType.c_str(), from, to);
    return static_cast<jint>(rc);
}

// ─────────────────────────────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────────────────────────────

extern "C" JNIEXPORT jstring JNICALL
Java_ai_edgevdb_EdgeVDB_nativeGetStats(
        JNIEnv* env, jclass /*clazz*/, jlong handle)
{
    auto h = reinterpret_cast<EvdbHandle*>(handle);
    const char* ver = evdb_version_string();
    std::string stats = "{\"version\":\"";
    stats += (ver ? ver : "unknown");
    stats += "\"}";
    return env->NewStringUTF(stats.c_str());
}
