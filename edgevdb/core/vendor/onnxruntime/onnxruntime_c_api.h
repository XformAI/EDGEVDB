// ONNX Runtime C API Header Stub
// This is a minimal declaration stub for compilation.
// In production, replace with the official ONNX Runtime C API header from:
// https://github.com/microsoft/onnxruntime/releases
// License: MIT

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
typedef struct OrtEnv OrtEnv;
typedef struct OrtSession OrtSession;
typedef struct OrtSessionOptions OrtSessionOptions;
typedef struct OrtRunOptions OrtRunOptions;
typedef struct OrtValue OrtValue;
typedef struct OrtMemoryInfo OrtMemoryInfo;
typedef struct OrtAllocator OrtAllocator;
typedef struct OrtStatus OrtStatus;
typedef struct OrtApi OrtApi;
typedef struct OrtApiBase OrtApiBase;

// Enums
typedef enum OrtLoggingLevel {
    ORT_LOGGING_LEVEL_VERBOSE = 0,
    ORT_LOGGING_LEVEL_INFO = 1,
    ORT_LOGGING_LEVEL_WARNING = 2,
    ORT_LOGGING_LEVEL_ERROR = 3,
    ORT_LOGGING_LEVEL_FATAL = 4
} OrtLoggingLevel;

typedef enum ONNXTensorElementDataType {
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED = 0,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 = 2,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 = 3,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16 = 4,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16 = 5,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 = 6,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING = 8,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL = 9,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 = 10,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE = 11,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32 = 12,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64 = 13,
} ONNXTensorElementDataType;

typedef enum OrtAllocatorType {
    OrtInvalidAllocator = -1,
    OrtDeviceAllocator = 0,
    OrtArenaAllocator = 1
} OrtAllocatorType;

typedef enum OrtMemType {
    OrtMemTypeCPUInput = -2,
    OrtMemTypeCPUOutput = -1,
    OrtMemTypeCPU = OrtMemTypeCPUOutput,
    OrtMemTypeDefault = 0,
} OrtMemType;

// API structure with function pointers
struct OrtApi {
    // Environment
    OrtStatus* (*CreateEnv)(OrtLoggingLevel log_severity_level, const char* logid, OrtEnv** out);
    
    // Session
    OrtStatus* (*CreateSessionOptions)(OrtSessionOptions** options);
    OrtStatus* (*SetIntraOpNumThreads)(OrtSessionOptions* options, int intra_op_num_threads);
    OrtStatus* (*SetInterOpNumThreads)(OrtSessionOptions* options, int inter_op_num_threads);
    OrtStatus* (*CreateSession)(const OrtEnv* env, const char* model_path,
                                 const OrtSessionOptions* options, OrtSession** out);
    
    // Memory
    OrtStatus* (*CreateCpuMemoryInfo)(OrtAllocatorType type, OrtMemType mem_type,
                                       OrtMemoryInfo** out);
    
    // Tensor
    OrtStatus* (*CreateTensorWithDataAsOrtValue)(const OrtMemoryInfo* info,
                                                   void* p_data, size_t p_data_len,
                                                   const int64_t* shape, size_t shape_len,
                                                   ONNXTensorElementDataType type,
                                                   OrtValue** out);
    OrtStatus* (*GetTensorMutableData)(OrtValue* value, void** out);
    
    // Run
    OrtStatus* (*Run)(OrtSession* session, const OrtRunOptions* run_options,
                       const char* const* input_names, const OrtValue* const* inputs,
                       size_t input_len,
                       const char* const* output_names, size_t output_names_len,
                       OrtValue** outputs);
    
    // Cleanup
    void (*ReleaseEnv)(OrtEnv* input);
    void (*ReleaseSession)(OrtSession* input);
    void (*ReleaseSessionOptions)(OrtSessionOptions* input);
    void (*ReleaseMemoryInfo)(OrtMemoryInfo* input);
    void (*ReleaseValue)(OrtValue* input);
    void (*ReleaseStatus)(OrtStatus* status);
    
    // Error handling
    const char* (*GetErrorMessage)(const OrtStatus* status);
};

struct OrtApiBase {
    const OrtApi* (*GetApi)(uint32_t version);
    const char* (*GetVersionString)(void);
};

// Main entry point
#define ORT_API_VERSION 18
const OrtApiBase* OrtGetApiBase(void);

#ifdef __cplusplus
}
#endif
