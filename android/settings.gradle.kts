pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        flatDir {
            dirs("sdk/build/outputs/aar")
        }
    }
}

rootProject.name = "EDGEVDB"
include(":sdk")
include(":demos:rag-demo")
