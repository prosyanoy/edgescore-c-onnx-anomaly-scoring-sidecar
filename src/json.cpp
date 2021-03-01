#include "edgescore/json.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace edgescore::json {
namespace {

constexpr int kMaxDepth = 64;

class Parser {
 public:
  Parser(std::string_view t, std::size_t pos) : t_(t), p_(pos) {}

  std::size_t pos() const { return p_; }
  void skip_ws() {
    while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\t' || t_[p_] == '\n' || t_[p_] == '\r')) ++p_;
  }
  bool at_end() const { return p_ >= t_.size(); }

  Value value(int depth) {
    if (depth > kMaxDepth) fail("nesting too deep");
    skip_ws();
    if (at_end()) fail("unexpected end of input");
    switch (t_[p_]) {
      case '{': return object(depth);
      case '[': return array(depth);
      case '"': return Value::make_string(string());
      case 't': literal("true"); return Value::make_bool(true);
      case 'f': literal("false"); return Value::make_bool(false);
      case 'n': literal("null"); return Value();
      default: return number();
    }
  }

 private:
  [[noreturn]] void fail(const std::string& m) const { throw ParseError(m, p_); }

  void literal(const char* word) {
    std::size_t n = std::strlen(word);
    if (t_.substr(p_, n) != word) fail("invalid literal");
    p_ += n;
  }

  Value object(int depth) {
    ++p_;  // {
    Value v = Value::make_object();
    skip_ws();
    if (!at_end() && t_[p_] == '}') { ++p_; return v; }
    for (;;) {
      skip_ws();
      if (at_end() || t_[p_] != '"') fail("expected object key");
      std::string key = string();
      skip_ws();
      if (at_end() || t_[p_] != ':') fail("expected ':'");
      ++p_;
      v.set(std::move(key), value(depth + 1));
      skip_ws();
      if (at_end()) fail("unterminated object");
      if (t_[p_] == ',') { ++p_; continue; }
      if (t_[p_] == '}') { ++p_; return v; }
      fail("expected ',' or '}'");
    }
  }

  Value array(int depth) {
    ++p_;  // [
    Value v = Value::make_array();
    skip_ws();
    if (!at_end() && t_[p_] == ']') { ++p_; return v; }
    for (;;) {
      v.push_back(value(depth + 1));
      skip_ws();
      if (at_end()) fail("unterminated array");
      if (t_[p_] == ',') { ++p_; continue; }
      if (t_[p_] == ']') { ++p_; return v; }
      fail("expected ',' or ']'");
    }
  }

  unsigned hex4() {
    if (p_ + 4 > t_.size()) fail("truncated \\u escape");
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = t_[p_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad hex digit");
    }
    return v;
  }

  static void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  std::string string() {
    ++p_;  // opening quote
    std::string out;
    for (;;) {
      if (at_end()) fail("unterminated string");
      unsigned char c = static_cast<unsigned char>(t_[p_++]);
      if (c == '"') return out;
      if (c < 0x20) { --p_; fail("control character in string"); }
      if (c != '\\') { out += static_cast<char>(c); continue; }
      if (at_end()) fail("unterminated escape");
      char e = t_[p_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (t_.substr(p_, 2) != "\\u") fail("lone high surrogate");
            p_ += 2;
            unsigned lo = hex4();
            if (lo < 0xDC00 || lo > 0xDFFF) fail("invalid low surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            fail("lone low surrogate");
          }
          append_utf8(out, cp);
          break;
        }
        default: --p_; fail("invalid escape");
      }
    }
  }

  Value number() {
    std::size_t start = p_;
    if (p_ < t_.size() && t_[p_] == '-') ++p_;
    if (p_ >= t_.size()) fail("invalid number");
    if (t_[p_] == '0') ++p_;
    else if (t_[p_] >= '1' && t_[p_] <= '9') while (p_ < t_.size() && std::isdigit(static_cast<unsigned char>(t_[p_]))) ++p_;
    else { p_ = start; fail("unexpected character"); }
    if (p_ < t_.size() && t_[p_] == '.') {
      ++p_;
      std::size_t d = p_;
      while (p_ < t_.size() && std::isdigit(static_cast<unsigned char>(t_[p_]))) ++p_;
      if (d == p_) fail("digits expected after '.'");
    }
    if (p_ < t_.size() && (t_[p_] == 'e' || t_[p_] == 'E')) {
      ++p_;
      if (p_ < t_.size() && (t_[p_] == '+' || t_[p_] == '-')) ++p_;
      std::size_t d = p_;
      while (p_ < t_.size() && std::isdigit(static_cast<unsigned char>(t_[p_]))) ++p_;
      if (d == p_) fail("digits expected in exponent");
    }
    std::string tmp(t_.substr(start, p_ - start));  // strtod needs NUL termination
    double d = std::strtod(tmp.c_str(), nullptr);
    if (!std::isfinite(d)) { p_ = start; fail("number out of range"); }
    return Value::make_number(d);
  }

  std::string_view t_;
  std::size_t p_;
};

}  // namespace

Value parse(std::string_view text) {
  Parser p(text, 0);
  Value v = p.value(0);
  p.skip_ws();
  if (!p.at_end()) throw ParseError("trailing characters", p.pos());
  return v;
}

bool parse_next(std::string_view text, std::size_t& pos, Value& out) {
  Parser p(text, pos);
  p.skip_ws();
  if (p.at_end()) { pos = text.size(); return false; }
  out = p.value(0);
  pos = p.pos();
  return true;
}

std::string escape(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

}  // namespace edgescore::json
