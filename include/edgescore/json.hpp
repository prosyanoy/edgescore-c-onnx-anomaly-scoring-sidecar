// Minimal, strict JSON parser (RFC 8259) that can also walk a stream of
// concatenated / newline-delimited values, which is what Splunk HEC accepts.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace edgescore::json {

class ParseError : public std::runtime_error {
 public:
  ParseError(const std::string& msg, std::size_t offset)
      : std::runtime_error(msg + " at offset " + std::to_string(offset)), offset_(offset) {}
  std::size_t offset() const noexcept { return offset_; }

 private:
  std::size_t offset_;
};

class Value {
 public:
  enum class Type { Null, Bool, Number, String, Array, Object };
  using Member = std::pair<std::string, Value>;

  Value() = default;
  static Value make_bool(bool b) { Value v; v.type_ = Type::Bool; v.bool_ = b; return v; }
  static Value make_number(double d) { Value v; v.type_ = Type::Number; v.num_ = d; return v; }
  static Value make_string(std::string s) { Value v; v.type_ = Type::String; v.str_ = std::move(s); return v; }
  static Value make_array() { Value v; v.type_ = Type::Array; return v; }
  static Value make_object() { Value v; v.type_ = Type::Object; return v; }

  Type type() const noexcept { return type_; }
  bool is_null() const noexcept { return type_ == Type::Null; }
  bool is_bool() const noexcept { return type_ == Type::Bool; }
  bool is_number() const noexcept { return type_ == Type::Number; }
  bool is_string() const noexcept { return type_ == Type::String; }
  bool is_array() const noexcept { return type_ == Type::Array; }
  bool is_object() const noexcept { return type_ == Type::Object; }

  bool as_bool() const { return bool_; }
  double as_number() const { return num_; }
  const std::string& as_string() const { return str_; }
  const std::vector<Value>& items() const { return arr_; }
  const std::vector<Member>& members() const { return obj_; }

  // Object lookup (last duplicate key wins, like most parsers). nullptr if absent.
  const Value* find(std::string_view key) const noexcept {
    for (auto it = obj_.rbegin(); it != obj_.rend(); ++it)
      if (it->first == key) return &it->second;
    return nullptr;
  }

  void push_back(Value v) { arr_.push_back(std::move(v)); }
  void set(std::string key, Value v) { obj_.emplace_back(std::move(key), std::move(v)); }

 private:
  Type type_ = Type::Null;
  bool bool_ = false;
  double num_ = 0.0;
  std::string str_;
  std::vector<Value> arr_;
  std::vector<Member> obj_;
};

// Parse exactly one document; trailing non-whitespace is an error.
Value parse(std::string_view text);

// Parse the next value starting at `pos`, skipping leading whitespace/commas-free
// separators. Returns false (and leaves pos at end) when only whitespace remains.
// Throws ParseError on malformed input. Values may be separated by any whitespace
// (or nothing, e.g. `{"a":1}{"b":2}`).
bool parse_next(std::string_view text, std::size_t& pos, Value& out);

// Escape a string for embedding in JSON output (adds no surrounding quotes).
std::string escape(std::string_view s);

}  // namespace edgescore::json
