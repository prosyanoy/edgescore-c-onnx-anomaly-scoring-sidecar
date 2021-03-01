#include "edgescore/service.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "edgescore/events.hpp"
#include "edgescore/json.hpp"

namespace edgescore {
namespace {

HttpResponse json_response(int status, std::string body) {
  HttpResponse r;
  r.status = status;
  r.body = std::move(body);
  return r;
}

// Splunk HEC style: {"text": "...", "code": N}
HttpResponse hec(int status, const char* text, int code, const std::string& extra = {}) {
  return json_response(status, std::string("{\"text\":\"") + text + "\",\"code\":" + std::to_string(code) + extra + "}");
}

std::string scored_json(const ScoredEvent& e) {
  char num[32];
  std::snprintf(num, sizeof num, "%.6f", static_cast<double>(e.score));
  std::string s = "{\"seq\":" + std::to_string(e.seq);
  if (!e.id.empty()) s += ",\"id\":\"" + json::escape(e.id) + "\"";
  if (!e.host.empty()) s += ",\"host\":\"" + json::escape(e.host) + "\"";
  s += e.failed ? ",\"error\":\"inference failed\"" : std::string(",\"score\":") + num + ",\"anomaly\":" + (e.anomaly ? "true" : "false");
  return s + "}";
}

}  // namespace

Service::Service(const ModelMeta& meta, Pipeline& pipeline, Metrics& metrics, ServiceOptions opts)
    : meta_(meta), pipeline_(pipeline), metrics_(metrics), opts_(std::move(opts)) {}

bool Service::authorized(const HttpRequest& req, HttpResponse& deny) const {
  if (opts_.hec_token.empty()) return true;
  std::string auth = req.header("authorization");
  if (auth.empty()) { deny = hec(401, "Token is required", 2); return false; }
  static const std::string kPrefix = "Splunk ";
  if (auth.compare(0, kPrefix.size(), kPrefix) != 0) { deny = hec(401, "Invalid authorization", 3); return false; }
  const std::string token = auth.substr(kPrefix.size());
  // Constant-time compare to avoid leaking token prefixes through timing.
  unsigned diff = token.size() ^ opts_.hec_token.size();
  for (std::size_t i = 0; i < token.size(); ++i)
    diff |= static_cast<unsigned char>(token[i]) ^ static_cast<unsigned char>(opts_.hec_token[i % std::max<std::size_t>(1, opts_.hec_token.size())]);
  if (diff != 0) { deny = hec(403, "Invalid token", 4); return false; }
  return true;
}

HttpResponse Service::handle(const HttpRequest& req) {
  metrics_.requests.fetch_add(1, std::memory_order_relaxed);
  const std::string& p = req.path;
  const bool get = req.method == "GET" || req.method == "HEAD";
  const bool post = req.method == "POST";

  if (p == "/healthz") return get ? json_response(200, "{\"status\":\"ok\"}") : hec(405, "Method not allowed", 6);
  if (p == "/readyz") {
    bool ready = pipeline_.running();
    return json_response(ready ? 200 : 503, ready ? "{\"status\":\"ready\"}" : "{\"status\":\"stopping\"}");
  }
  if (p == "/metrics") return get ? metrics_page() : hec(405, "Method not allowed", 6);
  if (p == "/v1/model") return get ? json_response(200, meta_.raw_json) : hec(405, "Method not allowed", 6);
  if (p == "/v1/results") return get ? results(req) : hec(405, "Method not allowed", 6);
  if (p == "/services/collector/health" || p == "/services/collector/health/1.0")
    return get ? hec(200, "HEC is healthy", 17) : hec(405, "Method not allowed", 6);

  if (p == "/services/collector/event" || p == "/services/collector" || p == "/services/collector/event/1.0")
    return post ? ingest_hec(req) : hec(405, "Method not allowed", 6);
  if (p == "/v1/events") return post ? ingest_async(req) : hec(405, "Method not allowed", 6);
  if (p == "/v1/score") return post ? score_sync(req) : hec(405, "Method not allowed", 6);
  return hec(404, "Not found", 6);
}

HttpResponse Service::metrics_page() {
  HttpResponse r;
  r.content_type = "text/plain; version=0.0.4; charset=utf-8";
  r.body = metrics_.render_prometheus(pipeline_.queue_depth(), pipeline_.queue_capacity());
  return r;
}

HttpResponse Service::ingest_hec(const HttpRequest& req) {
  HttpResponse deny;
  if (!authorized(req, deny)) return deny;
  ParseResult pr = parse_events(req.body, meta_.features, opts_.max_events_per_request);
  if (!pr.ok) {
    metrics_.rejected_invalid.fetch_add(1, std::memory_order_relaxed);
    return json_response(400, "{\"text\":\"Invalid data format\",\"code\":6,\"invalid-event-number\":" +
                                  std::to_string(pr.error_index) + ",\"detail\":\"" + json::escape(pr.error) + "\"}");
  }
  if (pr.events.empty()) return hec(400, "No data", 5);
  const std::size_t n = pr.events.size();
  if (!pipeline_.submit(pr.events)) {
    metrics_.rejected_backpressure.fetch_add(n, std::memory_order_relaxed);
    HttpResponse r = hec(503, "Server is busy", 9);
    r.headers.emplace_back("Retry-After", "1");
    return r;
  }
  return hec(200, "Success", 0);
}

HttpResponse Service::ingest_async(const HttpRequest& req) {
  HttpResponse deny;
  if (!authorized(req, deny)) return deny;
  ParseResult pr = parse_events(req.body, meta_.features, opts_.max_events_per_request);
  if (!pr.ok) {
    metrics_.rejected_invalid.fetch_add(1, std::memory_order_relaxed);
    return json_response(400, "{\"error\":\"" + json::escape(pr.error) + "\",\"event_index\":" + std::to_string(pr.error_index) + "}");
  }
  if (pr.events.empty()) return json_response(400, "{\"error\":\"no events in request body\"}");
  const std::size_t n = pr.events.size();
  if (!pipeline_.submit(pr.events)) {
    metrics_.rejected_backpressure.fetch_add(n, std::memory_order_relaxed);
    HttpResponse r = json_response(503, "{\"error\":\"queue full, retry later\"}");
    r.headers.emplace_back("Retry-After", "1");
    return r;
  }
  return json_response(202, "{\"accepted\":" + std::to_string(n) + "}");
}

HttpResponse Service::score_sync(const HttpRequest& req) {
  HttpResponse deny;
  if (!authorized(req, deny)) return deny;
  ParseResult pr = parse_events(req.body, meta_.features, opts_.max_events_per_request);
  if (!pr.ok) {
    metrics_.rejected_invalid.fetch_add(1, std::memory_order_relaxed);
    return json_response(400, "{\"error\":\"" + json::escape(pr.error) + "\",\"event_index\":" + std::to_string(pr.error_index) + "}");
  }
  if (pr.events.empty()) return json_response(400, "{\"error\":\"no events in request body\"}");
  const std::size_t n = pr.events.size();
  auto completion = std::make_shared<Completion>(n);
  if (!pipeline_.submit(pr.events, completion)) {
    metrics_.rejected_backpressure.fetch_add(n, std::memory_order_relaxed);
    HttpResponse r = json_response(503, "{\"error\":\"queue full, retry later\"}");
    r.headers.emplace_back("Retry-After", "1");
    return r;
  }
  if (!completion->wait_for(opts_.score_timeout)) return json_response(504, "{\"error\":\"scoring timed out\"}");
  HttpResponse r;
  r.content_type = "application/x-ndjson";
  for (const auto& e : completion->results()) r.body += scored_json(e) + "\n";
  return r;
}

HttpResponse Service::results(const HttpRequest& req) {
  std::size_t limit = 100;
  if (std::string l = req.query_param("limit"); !l.empty()) {
    char* end = nullptr;
    unsigned long v = std::strtoul(l.c_str(), &end, 10);
    if (*end != '\0' || v == 0) return json_response(400, "{\"error\":\"limit must be a positive integer\"}");
    limit = std::min<unsigned long>(v, 10000);
  }
  bool only_anomalies = req.query_param("anomalies") == "1" || req.query_param("anomalies") == "true";
  auto recent = pipeline_.recent(limit, only_anomalies);
  std::string body = "{\"threshold\":" + std::to_string(meta_.threshold) + ",\"results\":[";
  for (std::size_t i = 0; i < recent.size(); ++i) {
    if (i) body += ',';
    body += scored_json(recent[i]);
  }
  return json_response(200, body + "]}");
}

}  // namespace edgescore
