plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    kotlin("plugin.serialization") version "1.9.22"
    `maven-publish`
}

android {
    namespace = "in.xformai.edgevdb"
    compileSdk = 35

    defaultConfig {
        minSdk = 26
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("../consumer-rules.pro")

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                cppFlags("-std=c++17")
                arguments(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_ARM_NEON=TRUE",
                    "-DCMAKE_BUILD_TYPE=Release"
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false  // consumers control minification
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

// Suppress duplicate source entries in the sources JAR produced by maven-publish
tasks.withType<Jar> {
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
}

val libVersion = findProperty("EDGEVDB_VERSION") as String? ?: "0.1.1"

publishing {
    publications {
        register<MavenPublication>("release") {
            groupId = "in.xformai"
            artifactId = "edgevdb-android"
            version = libVersion

            afterEvaluate {
                from(components["release"])
            }

            pom {
                name.set("EdgeVDB Android SDK")
                description.set("On-device vector database with HNSW, hybrid retrieval, knowledge graph, and CRDT sync")
                url.set("https://github.com/XformAI/EDGEVDB")

                licenses {
                    license {
                        name.set("Apache License 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0")
                    }
                }

                developers {
                    developer {
                        id.set("xformai")
                        name.set("XformAI")
                        email.set("amlan@xformai.in")
                    }
                }

                scm {
                    connection.set("scm:git:git://github.com/XformAI/EDGEVDB.git")
                    developerConnection.set("scm:git:ssh://github.com:XformAI/EDGEVDB.git")
                    url.set("https://github.com/XformAI/EDGEVDB")
                }
            }
        }
    }

    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/XformAI/EDGEVDB")
            credentials {
                username = System.getenv("GITHUB_ACTOR") ?: findProperty("gpr.user") as String? ?: ""
                password = System.getenv("GITHUB_TOKEN") ?: findProperty("gpr.token") as String? ?: ""
            }
        }
    }
}

dependencies {
    // ONNX Runtime — full Android package (includes GPU / NNAPI delegates)
    api(libs.onnxruntime.android)

    // Coroutines for async embedding
    api(libs.kotlinx.coroutines.android)

    // Serialization for chunk metadata
    implementation(libs.kotlinx.serialization.json)

    // Testing
    testImplementation(libs.junit)
    testImplementation(libs.mockk)
    testImplementation(libs.coroutines.test)
}
