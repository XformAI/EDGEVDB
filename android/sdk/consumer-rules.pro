# Keep EdgeVDB public API
-keep class ai.edgevdb.EdgeVDB { *; }
-keep class ai.edgevdb.ChunkResult { *; }
-keep class ai.edgevdb.QueryResult { *; }
-keep class ai.edgevdb.DocumentChunk { *; }
-keep class ai.edgevdb.OnnxEmbeddingPipeline { *; }
-keep class ai.edgevdb.SimpleVectorDB { *; }
-keep class ai.edgevdb.RagEngine { *; }
-keep class ai.edgevdb.EmbeddingPipeline { *; }

# Keep ONNX Runtime classes (JNI)
-keep class ai.onnxruntime.** { *; }
-keepclassmembers class ai.onnxruntime.** { *; }

# Keep JNI-called native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Kotlin serialisation
-keepattributes *Annotation*, InnerClasses
-dontnote kotlinx.serialization.AnnotationsKt
-keepclassmembers class kotlinx.serialization.json.** { *** Companion; }
-keep @kotlinx.serialization.Serializable class * { *; }
