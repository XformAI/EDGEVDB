#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "object_store.hpp"
#include <cstdio>

using namespace edgevdb;

TEST_CASE("Object Store CRUD") {
    ObjectStore store;

    SUBCASE("Insert and get") {
        nlohmann::json props;
        props["title"] = "Test Document";
        props["author"] = "Alice";
        props["page_count"] = 42;

        uint64_t id = store.put("Document", props);
        CHECK(id > 0);

        nlohmann::json out;
        CHECK(store.get(id, out));
        CHECK(out["title"].get<std::string>() == "Test Document");
        CHECK(out["author"].get<std::string>() == "Alice");
    }

    SUBCASE("Query by property") {
        for (int i = 0; i < 10; i++) {
            nlohmann::json p;
            p["name"] = (i < 5) ? "Alice" : "Bob";
            p["index"] = i;
            store.put("User", p);
        }

        auto results = store.query("User", "name", "Alice");
        CHECK(results.size() == 5);
    }

    SUBCASE("Remove sets soft delete") {
        nlohmann::json p;
        p["value"] = "test";
        uint64_t id = store.put("Item", p);
        CHECK(store.remove(id));

        nlohmann::json out;
        CHECK_FALSE(store.get(id, out));
    }

    SUBCASE("Count") {
        for (int i = 0; i < 20; i++) {
            nlohmann::json p;
            p["val"] = i;
            store.put("Counter", p);
        }
        CHECK(store.count("Counter") == 20);
    }
}

TEST_CASE("Object Store save/load") {
    std::string path = "test_objects_temp.bin";

    {
        ObjectStore store(path);
        for (int i = 0; i < 50; i++) {
            nlohmann::json p;
            p["name"] = "item_" + std::to_string(i);
            store.put("Item", p);
        }
        CHECK(store.save());
    }

    {
        ObjectStore loaded(path);
        CHECK(loaded.open());
        CHECK(loaded.totalCount() == 50);
    }

    std::remove(path.c_str());
}
