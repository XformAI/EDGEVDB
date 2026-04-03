// doctest.h - single header testing framework stub
// This is a minimal implementation for EdgeVDB tests.
// In production, replace with the full doctest from:
// https://github.com/doctest/doctest/releases
// License: MIT

#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <functional>
#include <cmath>
#include <chrono>

namespace doctest {

struct TestCase {
    std::string name;
    std::function<void()> func;
    bool enabled = true;
};

inline std::vector<TestCase>& getTestCases() {
    static std::vector<TestCase> cases;
    return cases;
}

inline int& getFailCount() {
    static int count = 0;
    return count;
}

inline int& getPassCount() {
    static int count = 0;
    return count;
}

inline std::string& getCurrentSubcase() {
    static std::string s;
    return s;
}

struct TestRegistrar {
    TestRegistrar(const char* name, std::function<void()> func) {
        getTestCases().push_back({name, func, true});
    }
};

struct SubcaseSignature {
    const char* name;
    SubcaseSignature(const char* n) : name(n) {}
    operator bool() {
        getCurrentSubcase() = name;
        return true;
    }
};

} // namespace doctest

#define DOCTEST_CAT_IMPL(a, b) a##b
#define DOCTEST_CAT(a, b) DOCTEST_CAT_IMPL(a, b)

#define DOCTEST_TEST_CASE(name) \
    static void DOCTEST_CAT(DOCTEST_FUNC_, __LINE__)(); \
    static doctest::TestRegistrar DOCTEST_CAT(DOCTEST_REG_, __LINE__)(name, DOCTEST_CAT(DOCTEST_FUNC_, __LINE__)); \
    static void DOCTEST_CAT(DOCTEST_FUNC_, __LINE__)()

#define TEST_CASE(name) DOCTEST_TEST_CASE(name)

#define SUBCASE(name) if (doctest::SubcaseSignature DOCTEST_CAT(DOCTEST_SC_, __LINE__) = doctest::SubcaseSignature(name))


#define CHECK(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "CHECK FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            doctest::getFailCount()++; \
        } else { \
            doctest::getPassCount()++; \
        } \
    } while(0)

#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_FALSE(expr) CHECK(!(expr))
#define CHECK_UNARY(expr) CHECK(expr)
#define CHECK_UNARY_FALSE(expr) CHECK(!(expr))

#define REQUIRE(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "REQUIRE FAILED: " << #expr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            doctest::getFailCount()++; \
            return; \
        } else { \
            doctest::getPassCount()++; \
        } \
    } while(0)

#define REQUIRE_EQ(a, b) REQUIRE((a) == (b))
#define REQUIRE_NE(a, b) REQUIRE((a) != (b))
#define REQUIRE_FALSE(expr) REQUIRE(!(expr))

#define WARN(expr) \
    do { \
        if (!(expr)) { \
            std::cerr << "WARN: " << #expr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

#define MESSAGE(msg) std::cout << "  " << msg << std::endl;

#ifdef DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
inline int doctest_main(int argc, char** argv) {
    (void)argc; (void)argv;
    int total = 0, passed = 0, failed = 0;
    std::cout << "[doctest] Running " << doctest::getTestCases().size() << " test cases..." << std::endl;
    for (auto& tc : doctest::getTestCases()) {
        if (!tc.enabled) continue;
        std::cout << "  TEST: " << tc.name << std::endl;
        total++;
        doctest::getFailCount() = 0;
        doctest::getPassCount() = 0;
        try {
            tc.func();
        } catch (const std::exception& e) {
            std::cerr << "    EXCEPTION: " << e.what() << std::endl;
            doctest::getFailCount()++;
        }
        if (doctest::getFailCount() > 0) {
            std::cout << "    FAILED (" << doctest::getFailCount() << " failures)" << std::endl;
            failed++;
        } else {
            std::cout << "    PASSED (" << doctest::getPassCount() << " checks)" << std::endl;
            passed++;
        }
    }
    std::cout << "[doctest] " << passed << "/" << total << " tests passed";
    if (failed > 0) std::cout << ", " << failed << " FAILED";
    std::cout << std::endl;
    return failed > 0 ? 1 : 0;
}

int main(int argc, char** argv) { return doctest_main(argc, argv); }
#endif
