/**
 * vectordb_jni.cpp — JNI bridge from Android Kotlin to EdgeVDB C API
 *
 * JNI naming: Java_ai_edgevdb_EdgeVDB_<methodName>
 */

#include <jni.h>
#include <string>
#include <android/log.h>
#include "edgevdb/vectordb.h"

#define TAG "EdgeVDB"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

// Helper: convert jstring to std::string
static std::string jstringToString(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

extern "C" {

JNIEXPORT jlong JNICALL
Java_ai_edgevdb_EdgeVDB_nativeOpen(JNIEnv* env, jobject thiz,
                                    jstring storageDir,
                                    jint hnswM, jint efConstruction, jint efSearch,
                                    jfloat alpha, jfloat beta, jfloat gamma,
                                    jint tokenBudget) {
    try {
        EvdbConfig config;
        evdb_default_config(&config);
        std::string dir = jstringToString(env, storageDir);
        config.storage_dir = dir.c_str();
        config.hnsw_M = hnswM;
        config.hnsw_ef_construction = efConstruction;
        config.hnsw_ef_search = efSearch;
        config.ranker_alpha = alpha;
        config.ranker_beta = beta;
        config.ranker_gamma = gamma;
        config.token_budget = tokenBudget;

        EvdbHandle* h = evdb_open(&config);
        return reinterpret_cast<jlong>(h);
    } catch (const std::exception& e) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
        return 0;
    }
}

JNIEXPORT void JNICALL
Java_ai_edgevdb_EdgeVDB_nativeClose(JNIEnv* env, jobject thiz, jlong handle) {
    evdb_close(reinterpret_cast<EvdbHandle*>(handle));
}

JNIEXPORT jint JNICALL
Java_ai_edgevdb_EdgeVDB_nativeSave(JNIEnv* env, jobject thiz, jlong handle) {
    return static_cast<jint>(evdb_save(reinterpret_cast<EvdbHandle*>(handle)));
}

JNIEXPORT jlong JNICALL
Java_ai_edgevdb_EdgeVDB_nativeEmbedderCreate(JNIEnv* env, jobject thiz,
                                              jstring modelPath, jstring vocabPath, jint threads) {
    std::string mp = jstringToString(env, modelPath);
    std::string vp = jstringToString(env, vocabPath);
    EvdbEmbedder* e = evdb_embedder_create(mp.c_str(), vp.c_str(), threads);
    return reinterpret_cast<jlong>(e);
}

JNIEXPORT void JNICALL
Java_ai_edgevdb_EdgeVDB_nativeEmbedderDestroy(JNIEnv* env, jobject thiz, jlong embedderHandle) {
    evdb_embedder_destroy(reinterpret_cast<EvdbEmbedder*>(embedderHandle));
}

JNIEXPORT jlong JNICALL
Java_ai_edgevdb_EdgeVDB_nativeInsertText(JNIEnv* env, jobject thiz,
                                          jlong handle, jlong embedderHandle,
                                          jstring text, jint docId, jint pageNumber) {
    try {
        std::string t = jstringToString(env, text);
        uint64_t chunkId = 0;
        EvdbError err = evdb_insert_text(
            reinterpret_cast<EvdbHandle*>(handle),
            reinterpret_cast<EvdbEmbedder*>(embedderHandle),
            t.c_str(), static_cast<uint32_t>(docId),
            static_cast<uint32_t>(pageNumber), &chunkId);
        if (err != EVDB_OK) return -1;
        return static_cast<jlong>(chunkId);
    } catch (...) {
        return -1;
    }
}

JNIEXPORT jlong JNICALL
Java_ai_edgevdb_EdgeVDB_nativeInsertChunk(JNIEnv* env, jobject thiz,
                                           jlong handle, jfloatArray embedding,
                                           jstring text, jint docId, jint pageNumber) {
    try {
        std::string t = jstringToString(env, text);
        jfloat* emb = env->GetFloatArrayElements(embedding, nullptr);
        uint64_t chunkId = 0;
        EvdbError err = evdb_insert_chunk(
            reinterpret_cast<EvdbHandle*>(handle),
            t.c_str(), emb,
            static_cast<uint32_t>(docId),
            static_cast<uint32_t>(pageNumber), &chunkId);
        env->ReleaseFloatArrayElements(embedding, emb, 0);
        if (err != EVDB_OK) return -1;
        return static_cast<jlong>(chunkId);
    } catch (...) {
        return -1;
    }
}

JNIEXPORT jint JNICALL
Java_ai_edgevdb_EdgeVDB_nativeRemoveChunk(JNIEnv* env, jobject thiz,
                                           jlong handle, jlong chunkId) {
    return static_cast<jint>(evdb_remove_chunk(
        reinterpret_cast<EvdbHandle*>(handle), static_cast<uint64_t>(chunkId)));
}

JNIEXPORT jlong JNICALL
Java_ai_edgevdb_EdgeVDB_nativeQueryText(JNIEnv* env, jobject thiz,
                                         jlong handle, jlong embedderHandle,
                                         jstring queryText, jint topK,
                                         jboolean useKgExpansion) {
    std::string qt = jstringToString(env, queryText);
    EvdbQueryHandle* qh = evdb_query_text(
        reinterpret_cast<EvdbHandle*>(handle),
        reinterpret_cast<EvdbEmbedder*>(embedderHandle),
        qt.c_str(), topK, useKgExpansion ? 1 : 0);
    return reinterpret_cast<jlong>(qh);
}

JNIEXPORT jint JNICALL
Java_ai_edgevdb_EdgeVDB_nativeResultCount(JNIEnv* env, jobject thiz, jlong queryHandle) {
    return evdb_result_count(reinterpret_cast<EvdbQueryHandle*>(queryHandle));
}

JNIEXPORT jstring JNICALL
Java_ai_edgevdb_EdgeVDB_nativeResultText(JNIEnv* env, jobject thiz, jlong queryHandle, jint index) {
    const char* text = evdb_result_text(reinterpret_cast<EvdbQueryHandle*>(queryHandle), index);
    return env->NewStringUTF(text);
}

JNIEXPORT jfloat JNICALL
Java_ai_edgevdb_EdgeVDB_nativeResultScore(JNIEnv* env, jobject thiz, jlong queryHandle, jint index) {
    return evdb_result_score(reinterpret_cast<EvdbQueryHandle*>(queryHandle), index);
}

JNIEXPORT jint JNICALL
Java_ai_edgevdb_EdgeVDB_nativeResultPage(JNIEnv* env, jobject thiz, jlong queryHandle, jint index) {
    return static_cast<jint>(evdb_result_page(reinterpret_cast<EvdbQueryHandle*>(queryHandle), index));
}

JNIEXPORT jstring JNICALL
Java_ai_edgevdb_EdgeVDB_nativeResultContextString(JNIEnv* env, jobject thiz, jlong queryHandle) {
    const char* ctx = evdb_result_context_string(reinterpret_cast<EvdbQueryHandle*>(queryHandle));
    return env->NewStringUTF(ctx);
}

JNIEXPORT void JNICALL
Java_ai_edgevdb_EdgeVDB_nativeQueryFree(JNIEnv* env, jobject thiz, jlong queryHandle) {
    evdb_query_free(reinterpret_cast<EvdbQueryHandle*>(queryHandle));
}

JNIEXPORT jlong JNICALL
Java_ai_edgevdb_EdgeVDB_nativeObjectPut(JNIEnv* env, jobject thiz,
                                         jlong handle, jstring typeName,
                                         jstring jsonProperties) {
    std::string tn = jstringToString(env, typeName);
    std::string jp = jstringToString(env, jsonProperties);
    uint64_t id = 0;
    EvdbError err = evdb_object_put(reinterpret_cast<EvdbHandle*>(handle),
                                     tn.c_str(), jp.c_str(), &id);
    return (err == EVDB_OK) ? static_cast<jlong>(id) : -1;
}

JNIEXPORT jstring JNICALL
Java_ai_edgevdb_EdgeVDB_nativeObjectGet(JNIEnv* env, jobject thiz, jlong handle, jlong id) {
    char buf[4096];
    EvdbError err = evdb_object_get(reinterpret_cast<EvdbHandle*>(handle),
                                     static_cast<uint64_t>(id), buf, sizeof(buf));
    if (err != EVDB_OK) return nullptr;
    return env->NewStringUTF(buf);
}

JNIEXPORT jint JNICALL
Java_ai_edgevdb_EdgeVDB_nativeObjectRemove(JNIEnv* env, jobject thiz, jlong handle, jlong id) {
    return static_cast<jint>(evdb_object_remove(
        reinterpret_cast<EvdbHandle*>(handle), static_cast<uint64_t>(id)));
}

JNIEXPORT jstring JNICALL
Java_ai_edgevdb_EdgeVDB_nativeObjectQuery(JNIEnv* env, jobject thiz,
                                           jlong handle, jstring typeName,
                                           jstring property, jstring value, jint limit) {
    std::string tn = jstringToString(env, typeName);
    std::string prop = jstringToString(env, property);
    std::string val = jstringToString(env, value);
    char buf[32768];
    EvdbError err = evdb_object_query(reinterpret_cast<EvdbHandle*>(handle),
                                       tn.c_str(), prop.c_str(), val.c_str(),
                                       buf, sizeof(buf), limit);
    if (err != EVDB_OK) return nullptr;
    return env->NewStringUTF(buf);
}

JNIEXPORT jint JNICALL
Java_ai_edgevdb_EdgeVDB_nativeRelationAdd(JNIEnv* env, jobject thiz,
                                           jlong handle, jstring name,
                                           jlong fromId, jlong toId) {
    std::string n = jstringToString(env, name);
    return static_cast<jint>(evdb_relation_add(
        reinterpret_cast<EvdbHandle*>(handle), n.c_str(),
        static_cast<uint64_t>(fromId), static_cast<uint64_t>(toId)));
}

JNIEXPORT jstring JNICALL
Java_ai_edgevdb_EdgeVDB_nativeVersion(JNIEnv* env, jobject thiz) {
    return env->NewStringUTF(evdb_version_string());
}

} // extern "C"
