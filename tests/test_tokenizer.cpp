#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "tokenizer.hpp"
#include <fstream>
#include <cstdio>

using namespace edgevdb;

namespace {

const char* VOCAB_PATH = "test_vocab_tmp.txt";

void writeVocab(const std::vector<std::string>& lines) {
    std::ofstream f(VOCAB_PATH, std::ios::trunc);
    for (const auto& l : lines) f << l << "\n";
}

} // namespace

TEST_CASE("Special token ids resolved from vocab, not hardcoded positions") {
    // Non-standard ordering: [CLS] is NOT at index 101.
    writeVocab({"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world"});
    WordPieceTokenizer tok;
    REQUIRE(tok.loadVocab(VOCAB_PATH));

    CHECK(tok.padId() == 0);
    CHECK(tok.unkId() == 1);
    CHECK(tok.clsId() == 2);
    CHECK(tok.sepId() == 3);

    auto out = tok.encode("hello world", 16);
    CHECK(out.input_ids[0] == 2);              // [CLS]
    CHECK(out.input_ids[1] == 4);              // hello
    CHECK(out.input_ids[2] == 5);              // world
    CHECK(out.input_ids[3] == 3);              // [SEP]
    CHECK(out.input_ids[4] == 0);              // [PAD]
    std::remove(VOCAB_PATH);
}

TEST_CASE("Blank vocab lines keep line-number alignment") {
    // Line 2 (index 2) is blank; "foo" on line 4 must still get id 3.
    writeVocab({"[PAD]", "[UNK]", "", "foo"});
    WordPieceTokenizer tok;
    REQUIRE(tok.loadVocab(VOCAB_PATH));

    auto out = tok.encode("foo", 8);
    // [CLS] falls back to 101 (not in vocab); "foo" must be id 3.
    bool found_foo = false;
    for (int i = 0; i < out.actual_length; i++) {
        if (out.input_ids[i] == 3) found_foo = true;
    }
    CHECK(found_foo);
    std::remove(VOCAB_PATH);
}

TEST_CASE("UTF-8 text survives normalization") {
    writeVocab({"[PAD]", "[UNK]", "[CLS]", "[SEP]", "cafe", "naive"});
    WordPieceTokenizer tok;
    REQUIRE(tok.loadVocab(VOCAB_PATH));

    SUBCASE("Latin accents fold to base letters (correct codepoints)") {
        // café → cafe (é = U+00E9, encoded 0xC3 0xA9; must fold to 'e', not 'a')
        auto out = tok.encode("caf\xC3\xA9", 8);
        bool found_cafe = false;
        for (int i = 0; i < out.actual_length; i++) {
            if (out.input_ids[i] == 4) found_cafe = true;
        }
        CHECK(found_cafe);

        // naïve → naive (ï = U+00EF)
        auto out2 = tok.encode("na\xC3\xAFve", 8);
        bool found_naive = false;
        for (int i = 0; i < out2.actual_length; i++) {
            if (out2.input_ids[i] == 5) found_naive = true;
        }
        CHECK(found_naive);
    }

    SUBCASE("Non-Latin scripts become UNK, not empty") {
        // CJK characters are not in this vocab: expect [UNK] tokens instead
        // of the input silently vanishing.
        auto out = tok.encode("\xE4\xB8\xAD\xE6\x96\x87", 8); // 中文
        bool has_unk = false;
        for (int i = 0; i < out.actual_length; i++) {
            if (out.input_ids[i] == tok.unkId()) has_unk = true;
        }
        CHECK(has_unk);
        CHECK(out.actual_length > 2); // more than just [CLS][SEP]
    }
    std::remove(VOCAB_PATH);
}

TEST_CASE("WordPiece continuation pieces") {
    writeVocab({"[PAD]", "[UNK]", "[CLS]", "[SEP]", "un", "##able"});
    WordPieceTokenizer tok;
    REQUIRE(tok.loadVocab(VOCAB_PATH));

    auto out = tok.encode("unable", 8);
    CHECK(out.input_ids[1] == 4); // "un"
    CHECK(out.input_ids[2] == 5); // "##able"
    std::remove(VOCAB_PATH);
}
