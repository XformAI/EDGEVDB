// nlohmann/json.hpp - Vendored single-header JSON library
// This is a minimal stub for compilation. In production, replace with the full
// nlohmann/json single-header from https://github.com/nlohmann/json/releases
// License: MIT

#pragma once

#include <string>
#include <map>
#include <vector>
#include <variant>
#include <sstream>
#include <stdexcept>
#include <initializer_list>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace nlohmann {

class json {
public:
    enum class value_t {
        null,
        object,
        array,
        string,
        boolean,
        number_integer,
        number_unsigned,
        number_float
    };

    using object_t = std::map<std::string, json>;
    using array_t = std::vector<json>;

private:
    value_t type_ = value_t::null;
    object_t object_val_;
    array_t array_val_;
    std::string string_val_;
    double number_val_ = 0.0;
    int64_t int_val_ = 0;
    uint64_t uint_val_ = 0;
    bool bool_val_ = false;

public:
    json() : type_(value_t::null) {}
    json(std::nullptr_t) : type_(value_t::null) {}
    json(bool v) : type_(value_t::boolean), bool_val_(v) {}
    json(int v) : type_(value_t::number_integer), int_val_(v), number_val_(v) {}
    json(int64_t v) : type_(value_t::number_integer), int_val_(v), number_val_(static_cast<double>(v)) {}
    json(uint64_t v) : type_(value_t::number_unsigned), uint_val_(v), number_val_(static_cast<double>(v)) {}
    json(double v) : type_(value_t::number_float), number_val_(v) {}
    json(const char* v) : type_(value_t::string), string_val_(v) {}
    json(const std::string& v) : type_(value_t::string), string_val_(v) {}
    json(std::initializer_list<std::pair<const std::string, json>> init) : type_(value_t::object) {
        for (const auto& p : init) {
            object_val_[p.first] = p.second;
        }
    }

    static json object() {
        json j;
        j.type_ = value_t::object;
        return j;
    }

    static json array() {
        json j;
        j.type_ = value_t::array;
        return j;
    }

    static json parse(const std::string& str) {
        json result;
        size_t pos = 0;
        result = parse_value(str, pos);
        return result;
    }

    bool is_null() const { return type_ == value_t::null; }
    bool is_object() const { return type_ == value_t::object; }
    bool is_array() const { return type_ == value_t::array; }
    bool is_string() const { return type_ == value_t::string; }
    bool is_number() const { return type_ == value_t::number_integer || type_ == value_t::number_unsigned || type_ == value_t::number_float; }
    bool is_number_integer() const { return type_ == value_t::number_integer; }
    bool is_number_unsigned() const { return type_ == value_t::number_unsigned; }
    bool is_number_float() const { return type_ == value_t::number_float; }
    bool is_boolean() const { return type_ == value_t::boolean; }

    value_t type() const { return type_; }

    // Access operators
    json& operator[](const std::string& key) {
        if (type_ == value_t::null) type_ = value_t::object;
        return object_val_[key];
    }

    const json& operator[](const std::string& key) const {
        static json null_json;
        auto it = object_val_.find(key);
        if (it != object_val_.end()) return it->second;
        return null_json;
    }

    json& operator[](size_t index) {
        if (type_ == value_t::null) type_ = value_t::array;
        if (index >= array_val_.size()) array_val_.resize(index + 1);
        return array_val_[index];
    }

    const json& operator[](size_t index) const {
        return array_val_[index];
    }

    void push_back(const json& val) {
        if (type_ == value_t::null) type_ = value_t::array;
        array_val_.push_back(val);
    }

    size_t size() const {
        if (type_ == value_t::object) return object_val_.size();
        if (type_ == value_t::array) return array_val_.size();
        return 0;
    }

    bool empty() const { return size() == 0; }

    bool contains(const std::string& key) const {
        if (type_ != value_t::object) return false;
        return object_val_.find(key) != object_val_.end();
    }

    size_t count(const std::string& key) const {
        if (type_ != value_t::object) return 0;
        return object_val_.count(key);
    }

    // Iterators for objects
    auto begin() { return object_val_.begin(); }
    auto end() { return object_val_.end(); }
    auto begin() const { return object_val_.begin(); }
    auto end() const { return object_val_.end(); }

    // Items for range-based iteration
    const object_t& items() const { return object_val_; }

    // Conversion
    template<typename T>
    T get() const {
        if constexpr (std::is_same_v<T, std::string>) {
            return string_val_;
        } else if constexpr (std::is_same_v<T, int>) {
            return static_cast<int>(int_val_);
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return int_val_;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            if (type_ == value_t::number_unsigned) return uint_val_;
            return static_cast<uint64_t>(int_val_);
        } else if constexpr (std::is_same_v<T, double>) {
            return number_val_;
        } else if constexpr (std::is_same_v<T, float>) {
            return static_cast<float>(number_val_);
        } else if constexpr (std::is_same_v<T, bool>) {
            return bool_val_;
        }
        return T{};
    }

    operator std::string() const { return string_val_; }

    // Dump to string
    std::string dump(int indent = -1) const {
        std::ostringstream oss;
        dump_impl(oss, indent, 0);
        return oss.str();
    }

    // Erase
    size_t erase(const std::string& key) {
        return object_val_.erase(key);
    }

    // Find
    auto find(const std::string& key) { return object_val_.find(key); }
    auto find(const std::string& key) const { return object_val_.find(key); }

    // Comparison
    bool operator==(const json& other) const {
        if (type_ != other.type_) return false;
        switch (type_) {
            case value_t::null: return true;
            case value_t::boolean: return bool_val_ == other.bool_val_;
            case value_t::number_integer: return int_val_ == other.int_val_;
            case value_t::number_unsigned: return uint_val_ == other.uint_val_;
            case value_t::number_float: return number_val_ == other.number_val_;
            case value_t::string: return string_val_ == other.string_val_;
            case value_t::object: return object_val_ == other.object_val_;
            case value_t::array: return array_val_ == other.array_val_;
        }
        return false;
    }

    bool operator!=(const json& other) const { return !(*this == other); }

    friend std::ostream& operator<<(std::ostream& os, const json& j) {
        os << j.dump();
        return os;
    }

private:
    void dump_impl(std::ostringstream& oss, int indent, int depth) const {
        std::string pad = (indent > 0) ? std::string(depth * indent, ' ') : "";
        std::string pad_inner = (indent > 0) ? std::string((depth + 1) * indent, ' ') : "";
        std::string nl = (indent > 0) ? "\n" : "";
        std::string sep = (indent > 0) ? ": " : ":";

        switch (type_) {
            case value_t::null: oss << "null"; break;
            case value_t::boolean: oss << (bool_val_ ? "true" : "false"); break;
            case value_t::number_integer: oss << int_val_; break;
            case value_t::number_unsigned: oss << uint_val_; break;
            case value_t::number_float: oss << number_val_; break;
            case value_t::string:
                oss << "\"";
                for (char c : string_val_) {
                    switch (c) {
                        case '"': oss << "\\\""; break;
                        case '\\': oss << "\\\\"; break;
                        case '\n': oss << "\\n"; break;
                        case '\r': oss << "\\r"; break;
                        case '\t': oss << "\\t"; break;
                        default: oss << c;
                    }
                }
                oss << "\"";
                break;
            case value_t::object: {
                oss << "{" << nl;
                bool first = true;
                for (const auto& [k, v] : object_val_) {
                    if (!first) oss << "," << nl;
                    first = false;
                    oss << pad_inner << "\"" << k << "\"" << sep;
                    v.dump_impl(oss, indent, depth + 1);
                }
                oss << nl << pad << "}";
                break;
            }
            case value_t::array: {
                oss << "[" << nl;
                bool first = true;
                for (const auto& v : array_val_) {
                    if (!first) oss << "," << nl;
                    first = false;
                    oss << pad_inner;
                    v.dump_impl(oss, indent, depth + 1);
                }
                oss << nl << pad << "]";
                break;
            }
        }
    }

    static void skip_whitespace(const std::string& s, size_t& pos) {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r'))
            pos++;
    }

    static json parse_value(const std::string& s, size_t& pos) {
        skip_whitespace(s, pos);
        if (pos >= s.size()) return json();

        char c = s[pos];
        if (c == '"') return parse_string(s, pos);
        if (c == '{') return parse_object(s, pos);
        if (c == '[') return parse_array(s, pos);
        if (c == 't' || c == 'f') return parse_bool(s, pos);
        if (c == 'n') return parse_null(s, pos);
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number(s, pos);
        return json();
    }

    static json parse_string(const std::string& s, size_t& pos) {
        pos++; // skip opening "
        std::string result;
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                pos++;
                switch (s[pos]) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    default: result += s[pos]; break;
                }
            } else {
                result += s[pos];
            }
            pos++;
        }
        if (pos < s.size()) pos++; // skip closing "
        return json(result);
    }

    static json parse_object(const std::string& s, size_t& pos) {
        json result = json::object();
        pos++; // skip {
        skip_whitespace(s, pos);
        if (pos < s.size() && s[pos] == '}') { pos++; return result; }

        while (pos < s.size()) {
            skip_whitespace(s, pos);
            if (pos >= s.size() || s[pos] != '"') break;
            json key = parse_string(s, pos);
            skip_whitespace(s, pos);
            if (pos < s.size() && s[pos] == ':') pos++;
            skip_whitespace(s, pos);
            json val = parse_value(s, pos);
            result[key.get<std::string>()] = val;
            skip_whitespace(s, pos);
            if (pos < s.size() && s[pos] == ',') pos++;
            else break;
        }
        if (pos < s.size() && s[pos] == '}') pos++;
        return result;
    }

    static json parse_array(const std::string& s, size_t& pos) {
        json result = json::array();
        pos++; // skip [
        skip_whitespace(s, pos);
        if (pos < s.size() && s[pos] == ']') { pos++; return result; }

        while (pos < s.size()) {
            skip_whitespace(s, pos);
            json val = parse_value(s, pos);
            result.push_back(val);
            skip_whitespace(s, pos);
            if (pos < s.size() && s[pos] == ',') pos++;
            else break;
        }
        if (pos < s.size() && s[pos] == ']') pos++;
        return result;
    }

    static json parse_bool(const std::string& s, size_t& pos) {
        if (s.substr(pos, 4) == "true") { pos += 4; return json(true); }
        if (s.substr(pos, 5) == "false") { pos += 5; return json(false); }
        return json();
    }

    static json parse_null(const std::string& s, size_t& pos) {
        if (s.substr(pos, 4) == "null") { pos += 4; }
        return json();
    }

    static json parse_number(const std::string& s, size_t& pos) {
        size_t start = pos;
        bool is_float = false;
        bool is_negative = false;
        if (s[pos] == '-') { is_negative = true; pos++; }
        while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++;
        if (pos < s.size() && s[pos] == '.') { is_float = true; pos++; while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++; }
        if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) { is_float = true; pos++; if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) pos++; while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') pos++; }
        std::string numstr = s.substr(start, pos - start);
        if (is_float) return json(std::stod(numstr));
        if (is_negative) return json(static_cast<int64_t>(std::stoll(numstr)));
        uint64_t val = std::stoull(numstr);
        if (val <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            return json(static_cast<int64_t>(val));
        return json(val);
    }
};

} // namespace nlohmann
