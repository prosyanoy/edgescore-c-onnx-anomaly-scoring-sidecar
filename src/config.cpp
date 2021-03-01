#include "edgescore/config.hpp"

#include <cerrno>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace edgescore {
namespace {

unsigned long long parse_uint(const char* name, const char* text, unsigned long long lo, unsigned long long hi) {
  char* end = nullptr;
  errno = 0;
  if (*text == '-' || *text == '\0') throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
  unsigned long long v = std::strtoull(text, &end, 10);
  if (*end != '\0' || errno == ERANGE) throw std::invalid_argument(std::string(name) + " must be a non-negative integer");
  if (v < lo || v > hi)
    throw std::invalid_argument(std::string(name) + " must be between " + std::to_string(lo) + " and " + std::to_string(hi));
  return v;
}

}  // namespace

Config Config::from_env(const Getter& get) {
  Config c;
  auto str = [&](const char* name, std::string& dst) {
    if (const char* v = get(name); v && *v) dst = v;
  };
  auto num = [&](const char* name, auto& dst, unsigned long long lo, unsigned long long hi) {
    if (const char* v = get(name); v && *v) dst = static_cast<std::decay_t<decltype(dst)>>(parse_uint(name, v, lo, hi));
  };
  str("EDGESCORE_MODEL", c.model_path);
  str("EDGESCORE_MODEL_META", c.meta_path);
  str("EDGESCORE_BIND", c.bind_address);
  str("EDGESCORE_HEC_TOKEN", c.hec_token);
  num("EDGESCORE_PORT", c.port, 0, 65535);
  num("EDGESCORE_WORKERS", c.workers, 1, 256);
  num("EDGESCORE_INTRA_OP_THREADS", c.intra_op_threads, 1, 256);
  num("EDGESCORE_QUEUE_CAPACITY", c.queue_capacity, 1, 100000000);
  num("EDGESCORE_MAX_BATCH", c.max_batch, 1, 65536);
  num("EDGESCORE_MAX_CONNECTIONS", c.max_connections, 1, 100000);
  num("EDGESCORE_MAX_BODY_BYTES", c.max_body_bytes, 1, 1ull << 30);
  if (const char* v = get("EDGESCORE_MAX_WAIT_US"); v && *v)
    c.max_wait = std::chrono::microseconds(parse_uint("EDGESCORE_MAX_WAIT_US", v, 0, 10000000));
  if (const char* v = get("EDGESCORE_SCORE_TIMEOUT_MS"); v && *v)
    c.score_timeout = std::chrono::milliseconds(parse_uint("EDGESCORE_SCORE_TIMEOUT_MS", v, 1, 600000));

  if (c.meta_path.empty()) {
    const std::string& m = c.model_path;
    auto slash = m.find_last_of('/');
    c.meta_path = (slash == std::string::npos ? std::string() : m.substr(0, slash + 1)) + "model.json";
  }
  if (c.max_batch > c.queue_capacity)
    throw std::invalid_argument("EDGESCORE_MAX_BATCH must not exceed EDGESCORE_QUEUE_CAPACITY");
  return c;
}

Config Config::from_env() {
  return from_env([](const char* n) { return std::getenv(n); });
}

}  // namespace edgescore
