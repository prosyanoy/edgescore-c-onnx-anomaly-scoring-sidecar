#include "edgescore/events.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>

#include "edgescore/json.hpp"

namespace edgescore {
namespace {

bool as_float(const json::Value& v, float& out) {
  double d;
  if (v.is_number()) {
    d = v.as_number();
  } else if (v.is_string() && !v.as_string().empty()) {
    const std::string& s = v.as_string();
    char* end = nullptr;
    errno = 0;
    d = std::strtod(s.c_str(), &end);
    if (end != s.c_str() + s.size() || errno == ERANGE) return false;
  } else {
    return false;
  }
  if (!std::isfinite(d) || std::fabs(d) > 3.0e38) return false;
  out = static_cast<float>(d);
  return true;
}

std::string str_field(const json::Value& obj, std::string_view key) {
  const json::Value* v = obj.find(key);
  return v && v->is_string() ? v->as_string() : std::string();
}

}  // namespace

ParseResult parse_events(std::string_view body, const std::vector<std::string>& features, std::size_t max_events) {
  ParseResult res;
  std::size_t pos = 0;
  json::Value doc;
  auto fail = [&](std::string msg) {
    res.ok = false;
    res.error = std::move(msg);
    res.error_index = res.events.size();
    res.events.clear();
    return res;
  };
  for (;;) {
    try {
      if (!json::parse_next(body, pos, doc)) break;
    } catch (const json::ParseError& e) {
      return fail(e.what());
    }
    if (!doc.is_object()) return fail("event must be a JSON object");
    if (res.events.size() >= max_events) return fail("too many events in one request");

    const json::Value* inner = doc.find("event");
    const json::Value* fields = doc.find("fields");
    if (inner && !inner->is_object()) {
      return fail("\"event\" must be an object containing the telemetry fields");
    }
    const json::Value* sources[3] = {inner, fields && fields->is_object() ? fields : nullptr, &doc};

    EventRecord rec;
    rec.features.reserve(features.size());
    for (const auto& name : features) {
      const json::Value* v = nullptr;
      for (const json::Value* src : sources) {
        if (src && (v = src->find(name)) != nullptr) break;
      }
      float f;
      if (!v) return fail("missing feature \"" + name + "\"");
      if (!as_float(*v, f)) return fail("feature \"" + name + "\" is not a finite number");
      rec.features.push_back(f);
    }
    rec.host = str_field(doc, "host");
    rec.source = str_field(doc, "source");
    rec.id = inner ? str_field(*inner, "id") : std::string();
    if (rec.id.empty()) rec.id = str_field(doc, "id");
    res.events.push_back(std::move(rec));
  }
  return res;
}

}  // namespace edgescore
