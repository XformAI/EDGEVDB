plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.hilt.android)
    alias(libs.plugins.ksp)
    id("org.jetbrains.kotlin.plugin.serialization") version libs.versions.kotlin.get()
}

android {
    namespace  = "ai.edgevdb.demo"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.edgevdb.demo"
        minSdk        = 26
        targetSdk     = 35
        versionCode   = 1
        versionName   = "1.0"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.12"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation(project(":sdk"))

    // Compose
    val composeBom = platform(libs.compose.bom)
    implementation(composeBom)
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    debugImplementation(libs.compose.ui.tooling)

    // Navigation
    implementation(libs.navigation.compose)
    implementation(libs.activity.compose)

    // Lifecycle / ViewModel
    implementation(libs.lifecycle.viewmodel.compose)

    // Hilt
    implementation(libs.hilt.android)
    ksp(libs.hilt.compiler)

    // Hilt + Navigation Compose
    implementation("androidx.hilt:hilt-navigation-compose:1.2.0")

    // Coroutines
    implementation(libs.kotlinx.coroutines.android)

    // ONNX Runtime (required by SDK)
    implementation(libs.onnxruntime.android)

    // Kotlinx Serialization (required by SDK)
    implementation(libs.kotlinx.serialization.json)

    // Testing
    testImplementation(libs.junit)
    testImplementation(libs.coroutines.test)
    testImplementation(libs.mockk)
}
