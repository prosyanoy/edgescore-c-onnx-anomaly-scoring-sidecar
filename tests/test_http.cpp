#include <gtest/gtest.h>

#include <future>
#include <sstream>
#include <thread>

#include "edgescore/http_server.hpp"
#include "edgescore/json.hpp"
#include "edgescore/service.hpp"
#include "test_support.hpp"

using namespace edgescore;
using namespace edgescore::testing;
using namespace std::chrono_literals;

namespace {

ModelMeta fake_meta() {
  return parse_meta(R"({"name":"fake","features":["cpu","mem"],"threshold":50.0})");
}

struct Harness {
  explicit Harness(std::string token = "", std::size_t queue = 1024, std::size_t max_body = 1 << 20)
      : scorer(std::make_shared<FakeScorer>()), meta(fake_meta()) {
    PipelineOptions po;
    po.workers = 2; po.queue_capacity = queue; po.max_batch = 16; po.max_wait = 200us; po.threshold = meta.threshold;
    pipeline = std::make_unique<Pipeline>(scorer, po, metrics);
    ServiceOptions so;
    so.hec_token = std::move(token);
    so.score_timeout = 5s;
    service = std::make_unique<Service>(meta, *pipeline, metrics, so);
    HttpServerOptions ho;
    ho.bind_address = "127.0.0.1"; ho.port = 0; ho.max_body_bytes = max_body;
    server = std::make_unique<HttpServer>(ho, [this](const HttpRequest& r) { return service->handle(r); });
    server->start();
  }
  ~Harness() { server->stop(); pipeline->stop(); }
  std::uint16_t port() const { return server->port(); }
  bool wait_scored(std::uint64_t n) {
    for (int i = 0; i < 500; ++i) { if (metrics.events_scored.load() >= n) return true; std::this_thread::sleep_for(10ms); }
    return false;
  }
  std::shared_ptr<FakeScorer> scorer;
  ModelMeta meta;
  Metrics metrics;
  std::unique_ptr<Pipeline> pipeline;
  std::unique_ptr<Service> service;
  std::unique_ptr<HttpServer> server;
};

bool has(const std::string& s, const std::string& needle) { return s.find(needle) != std::string::npos; }

}  // namespace

TEST(Http, HecEventEndpointAcceptsEnvelopesAndScoresAsynchronously) {
  Harness h;
  std::string body =
      R"({"time":1700000000,"host":"web-1","sourcetype":"telemetry","event":{"cpu":10,"mem":20}})"
      "\n"
      R"({"event":{"cpu":90,"mem":20}})";
  auto resp = http_roundtrip(h.port(), post("/services/collector/event", body));
  EXPECT_TRUE(has(resp, "HTTP/1.1 200 OK")) << resp;
  EXPECT_EQ(body_of(resp), R"({"text":"Success","code":0})");
  ASSERT_TRUE(h.wait_scored(2));
  EXPECT_EQ(h.metrics.anomalies.load(), 1u);  // cpu=90 > threshold 50
}

TEST(Http, HecRejectsInvalidAndEmptyPayloads) {
  Harness h;
  auto bad = http_roundtrip(h.port(), post("/services/collector/event", R"({"event":{"cpu":1,"mem":1}}{"event":{"cpu":1}})"));
  EXPECT_TRUE(has(bad, "400 Bad Request")) << bad;
  auto doc = json::parse(body_of(bad));
  EXPECT_EQ(doc.find("code")->as_number(), 6);
  EXPECT_EQ(doc.find("invalid-event-number")->as_number(), 1);
  auto empty = http_roundtrip(h.port(), post("/services/collector/event", "  \n"));
  EXPECT_TRUE(has(empty, "\"code\":5")) << empty;
  EXPECT_EQ(h.metrics.events_received.load(), 0u);  // all-or-nothing
}

TEST(Http, HecTokenAuthentication) {
  Harness h("s3cret");
  const std::string ev = R"({"event":{"cpu":1,"mem":1}})";
  EXPECT_TRUE(has(http_roundtrip(h.port(), post("/services/collector/event", ev)), "401"));
  EXPECT_TRUE(has(http_roundtrip(h.port(), post("/services/collector/event", ev, "Authorization: Splunk nope\r\n")), "403"));
  EXPECT_TRUE(has(http_roundtrip(h.port(), post("/services/collector/event", ev, "Authorization: Bearer s3cret\r\n")), "401"));
  auto ok = http_roundtrip(h.port(), post("/services/collector/event", ev, "Authorization: Splunk s3cret\r\n"));
  EXPECT_TRUE(has(ok, "200 OK")) << ok;
  // Probes and metrics stay open for the kubelet / Prometheus.
  EXPECT_TRUE(has(http_roundtrip(h.port(), get("/healthz")), "200 OK"));
  EXPECT_TRUE(has(http_roundtrip(h.port(), get("/metrics")), "200 OK"));
}

TEST(Http, SyncScoreReturnsOrderedNdjson) {
  Harness h;
  std::string body = "{\"cpu\":1,\"mem\":0,\"id\":\"a\"}\n{\"cpu\":75,\"mem\":0,\"id\":\"b\"}\n";
  auto resp = http_roundtrip(h.port(), post("/v1/score", body));
  ASSERT_TRUE(has(resp, "200 OK")) << resp;
  std::string out = body_of(resp);
  auto nl = out.find('\n');
  auto first = json::parse(out.substr(0, nl));
  auto second = json::parse(out.substr(nl + 1));
  EXPECT_EQ(first.find("id")->as_string(), "a");
  EXPECT_FALSE(first.find("anomaly")->as_bool());
  EXPECT_DOUBLE_EQ(second.find("score")->as_number(), 75.0);
  EXPECT_TRUE(second.find("anomaly")->as_bool());
}

TEST(Http, AsyncEndpointAndResultsListing) {
  Harness h;
  auto resp = http_roundtrip(h.port(), post("/v1/events", "{\"cpu\":99,\"mem\":1,\"id\":\"hot\"}\n{\"cpu\":1,\"mem\":1}\n"));
  EXPECT_TRUE(has(resp, "202 Accepted")) << resp;
  EXPECT_EQ(body_of(resp), R"({"accepted":2})");
  ASSERT_TRUE(h.wait_scored(2));
  auto all = json::parse(body_of(http_roundtrip(h.port(), get("/v1/results"))));
  EXPECT_EQ(all.find("results")->items().size(), 2u);
  auto bad = json::parse(body_of(http_roundtrip(h.port(), get("/v1/results?anomalies=1&limit=5"))));
  ASSERT_EQ(bad.find("results")->items().size(), 1u);
  EXPECT_EQ(bad.find("results")->items()[0].find("id")->as_string(), "hot");
  EXPECT_TRUE(has(http_roundtrip(h.port(), get("/v1/results?limit=abc")), "400"));
}

TEST(Http, BackpressureReturns503WithRetryAfter) {
  Harness h("", /*queue=*/8);
  h.scorer->block();
  auto blocker = std::async(std::launch::async, [&] {  // occupies a worker inside the model
    return http_roundtrip(h.port(), post("/v1/score", R"({"cpu":1,"mem":1})"));
  });
  h.scorer->wait_entered(1);
  std::string eight;
  for (int i = 0; i < 8; ++i) eight += R"({"cpu":1,"mem":1})" "\n";
  // Second worker takes some of these; keep posting until the queue is genuinely full.
  int rejected = 0;
  std::string last;
  for (int i = 0; i < 10 && !rejected; ++i) {
    last = http_roundtrip(h.port(), post("/v1/events", eight));
    if (has(last, "503")) ++rejected;
  }
  ASSERT_EQ(rejected, 1) << last;
  EXPECT_TRUE(has(last, "Retry-After: 1"));
  EXPECT_GE(h.metrics.rejected_backpressure.load(), 8u);
  auto hec503 = http_roundtrip(h.port(), post("/services/collector/event", R"({"event":{"cpu":1,"mem":1}})" "\n" + eight));
  EXPECT_TRUE(has(hec503, "\"code\":9")) << hec503;
  h.scorer->unblock();
  EXPECT_TRUE(has(blocker.get(), "200 OK"));
}

TEST(Http, ChunkedRequestBody) {
  Harness h;
  auto chunk = [](const std::string& data) {
    std::ostringstream len;
    len << std::hex << data.size() << "\r\n";
    return len.str() + data + "\r\n";
  };
  std::string raw = "POST /services/collector/event HTTP/1.1\r\nHost: x\r\nConnection: close\r\n"
                    "Transfer-Encoding: chunked\r\n\r\n";
  raw += chunk(R"({"event":{"cpu":1,)") + chunk(R"("mem":2}})") + "0\r\n\r\n";
  auto resp = http_roundtrip(h.port(), raw);
  EXPECT_TRUE(has(resp, "\"code\":0")) << resp;
  EXPECT_TRUE(h.wait_scored(1));
}

TEST(Http, KeepAliveServesPipelinedRequests) {
  Harness h;
  std::string raw = "GET /healthz HTTP/1.1\r\nHost: x\r\n\r\n" + get("/readyz");
  auto resp = http_roundtrip(h.port(), raw, /*until_close=*/true);
  std::size_t count = 0, pos = 0;
  while ((pos = resp.find("HTTP/1.1 200 OK", pos)) != std::string::npos) { ++count; ++pos; }
  EXPECT_EQ(count, 2u) << resp;
}

TEST(Http, ProtocolErrors) {
  Harness h("", 1024, /*max_body=*/64);
  EXPECT_TRUE(has(http_roundtrip(h.port(), get("/nope")), "404"));
  EXPECT_TRUE(has(http_roundtrip(h.port(), get("/services/collector/event")), "405"));
  EXPECT_TRUE(has(http_roundtrip(h.port(), "garbage\r\n\r\n"), "400"));
  EXPECT_TRUE(has(http_roundtrip(h.port(), post("/v1/events", std::string(200, ' '))), "413"));
  EXPECT_TRUE(has(http_roundtrip(h.port(), post("/v1/events", "{}", "Content-Encoding: gzip\r\n")), "415"));
  EXPECT_TRUE(has(http_roundtrip(h.port(), get("/services/collector/health")), "\"code\":17"));
}

TEST(Http, MetricsEndpointReflectsTraffic) {
  Harness h;
  http_roundtrip(h.port(), post("/v1/score", "{\"cpu\":1,\"mem\":1}\n{\"cpu\":80,\"mem\":1}\n"));
  auto resp = http_roundtrip(h.port(), get("/metrics"));
  EXPECT_TRUE(has(resp, "text/plain; version=0.0.4"));
  EXPECT_TRUE(has(resp, "edgescore_events_scored_total 2")) << resp;
  EXPECT_TRUE(has(resp, "edgescore_anomalies_total 1"));
  EXPECT_TRUE(has(resp, "edgescore_latency_p99_seconds"));
}

TEST(Http, StopIsPromptEvenWithIdleKeepAliveClients) {
  auto h = std::make_unique<Harness>();
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET; a.sin_port = htons(h->port());
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  ASSERT_EQ(::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a), 0);
  std::this_thread::sleep_for(50ms);
  auto t0 = std::chrono::steady_clock::now();
  h.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - t0, 3s);
  ::close(fd);
}
