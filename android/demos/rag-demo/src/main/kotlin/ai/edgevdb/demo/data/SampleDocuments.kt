package ai.edgevdb.demo.data

/** Sample documents pre-loaded on first launch to demonstrate RAG. */
object SampleDocuments {

    data class Doc(val id: String, val title: String, val content: String)

    val all: List<Doc> = listOf(
        Doc(
            id      = "doc_android_arch",
            title   = "Android Architecture Guide",
            content = """
                Android applications are structured around four main components:
                Activities, Services, Broadcast Receivers, and Content Providers.
                Activities represent single screens with a user interface.
                The Android lifecycle callbacks—onCreate, onStart, onResume, onPause,
                onStop, and onDestroy—allow apps to manage resources appropriately.
                
                Modern Android development encourages the MVVM (Model-View-ViewModel)
                architecture pattern. ViewModels survive configuration changes and expose
                data through StateFlow or LiveData to Composables or Views. Jetpack
                Compose replaces XML layouts with declarative Kotlin functions.
                
                Dependency injection with Hilt simplifies object graph management.
                Room provides a SQLite abstraction for local persistence. WorkManager
                handles background tasks with guaranteed execution constraints.
                
                The Navigation component manages screen transitions with a type-safe
                navigation graph. Deep links and back-stack behaviour are declared
                declaratively.
            """.trimIndent()
        ),
        Doc(
            id      = "doc_vector_search",
            title   = "Vector Search & Semantic Similarity",
            content = """
                Vector search enables semantic similarity retrieval beyond keyword matching.
                Text embeddings map sentences to dense numeric vectors in high-dimensional
                space. Similar sentences cluster together; dissimilar ones are distant.
                
                The all-MiniLM-L6-v2 model produces 384-dimensional float32 embeddings.
                Cosine similarity—equivalent to dot-product for L2-normalised vectors—
                measures the angle between vectors. A score of 1.0 indicates identical
                semantic meaning; 0.0 indicates orthogonality.
                
                HNSW (Hierarchical Navigable Small World) is the state-of-the-art
                approximate nearest-neighbour algorithm. It builds a multi-layer proximity
                graph during indexing (O(n log n)) and performs greedy traversal during
                queries (O(log n)). Parameters M (edges per node) and ef control the
                accuracy/speed trade-off.
                
                Hybrid search combines vector similarity with keyword matching and
                structural signals (page proximity, document recency) to improve recall.
            """.trimIndent()
        ),
        Doc(
            id      = "doc_rag",
            title   = "Retrieval-Augmented Generation",
            content = """
                Retrieval-Augmented Generation (RAG) augments large language models with
                a dynamic knowledge base. Instead of relying solely on parametric memory
                baked into model weights, RAG retrieves relevant passages at inference time
                and prepends them to the prompt as context.
                
                The RAG pipeline has two phases: indexing and retrieval. During indexing,
                documents are chunked, embedded, and stored in a vector database. During
                retrieval, the user query is embedded and the top-K most similar chunks are
                fetched. These chunks form the context window for the LLM.
                
                On-device RAG with EdgeVDB eliminates network calls and preserves user
                privacy. The ONNX Runtime executes the embedding model locally on CPU or
                NNAPI. The HNSW index runs query latency under 10ms for up to 10,000
                indexed chunks on a mid-range Android device.
                
                Chunking strategy affects quality. Chunks should be semantically coherent—
                typically 150–300 words—with 10–20% overlap to avoid boundary effects.
                Sentence-aware splitting is preferred over character-count splitting.
            """.trimIndent()
        ),
        Doc(
            id      = "doc_kotlin_coroutines",
            title   = "Kotlin Coroutines & Flow",
            content = """
                Kotlin coroutines provide structured concurrency for Android. Coroutines
                are lightweight threads managed by a scheduler (dispatcher). The main
                dispatchers are Main (UI thread), IO (disk/network), and Default (CPU).
                
                suspend functions can be called only from other suspend functions or
                coroutine builders (launch, async, runBlocking). They do not block the
                calling thread; instead, they suspend and resume when the underlying
                operation completes.
                
                StateFlow is a hot observable that emits values to collectors. ViewModels
                expose StateFlow<UiState>; Compose collects them via collectAsStateWithLifecycle().
                Cold flows (flow { emit(...) }) are lazily evaluated.
                
                Structured concurrency ensures coroutine hierarchies: cancellation
                propagates from parent to child, preventing leaks. viewModelScope is
                automatically cancelled when the ViewModel is cleared.
            """.trimIndent()
        )
    )
}
