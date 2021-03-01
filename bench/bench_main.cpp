// Micro-benchmarks: JSON/event parsing, raw ONNX inference by batch size, and the full
// queue -> worker -> model pipeline.
#include <benchmark/benchmark.h>

#include <filesystem>
#include <memory>
#include <random>
#include <string>

#include "edgescore/bounded_queue.hpp"
#include "edgescore/events.hpp"
#include "edgescore/metrics.hpp"
#include "edgescore/model.hpp"
#include "edgescore/pipeline.hpp"

using namespace edgescore;

namespace {

const std::string kDir = EDGESCORE_MODEL_DIR;

ModelMeta& meta() {
  static ModelMeta m = load_meta(kDir + "/model.json");
  return m;
}
std::shared_ptr<const Scorer> scorer() {
  static std::shared_ptr<const Scorer> s = make_onnx_scorer(kDir + "/model.onnx", meta(), 1);
  return s;
}

std::vector<float> random_rows(std::size_t n) {
  std::mt19937 rng(42);
  std::normal_distribution<float> d(0.f, 1.f);
  const float mean[6] = {50, 45, 52, 0.3f, 200, 800}, sd[6] = {4, 5, 10, 0.3f, 30, 120};
  std::vector<float> x(n * 6);
  for (std::size_t i = 0; i < n; ++i)
    for (int j = 0; j < 6; ++j) x[i * 6 + j] = mean[j] + sd[j] * d(rng);
  return x;
}

std::string hec_payload(std::size_t n) {
  auto x = random_rows(n);
  std::string body;
  char buf[256];
  for (std::size_t i = 0; i < n; ++i) {
    std::snprintf(buf, sizeof buf,
                  "{\"time\":1.7e9,\"host\":\"web-%zu\",\"event\":{\"cpu_pct\":%.3f,\"mem_pct\":%.3f,\"latency_ms\":%.3f,"
                  "\"error_rate\":%.3f,\"req_rate\":%.3f,\"bytes_out_kb\":%.3f}}\n",
                  i % 8, x[i * 6], x[i * 6 + 1], x[i * 6 + 2], x[i * 6 + 3], x[i * 6 + 4], x[i * 6 + 5]);
    body += buf;
  }
  return body;
}

void BM_ParseHecEvents(benchmark::State& st) {
  const std::size_t n = static_cast<std::size_t>(st.range(0));
  const std::string body = hec_payload(n);
  for (auto _ : st) {
    auto r = parse_events(body, meta().features);
    benchmark::DoNotOptimize(r.events.data());
  }
  st.SetItemsProcessed(st.iterations() * static_cast<int64_t>(n));
  st.SetBytesProcessed(st.iterations() * static_cast<int64_t>(body.size()));
}
BENCHMARK(BM_ParseHecEvents)->Arg(1)->Arg(100)->Arg(1000);

void BM_OnnxInference(benchmark::State& st) {
  const std::size_t n = static_cast<std::size_t>(st.range(0));
  auto x = random_rows(n);
  std::vector<float> out(n);
  auto s = scorer();
  for (auto _ : st) {
    s->score(x.data(), n, out.data());
    benchmark::DoNotOptimize(out.data());
  }
  st.SetItemsProcessed(st.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_OnnxInference)->Arg(1)->Arg(8)->Arg(32)->Arg(64)->Arg(256)->Arg(1024);

void BM_QueueRoundTrip(benchmark::State& st) {
  static BoundedQueue<int> q(1 << 16);
  for (auto _ : st) {
    std::vector<int> v(64, 1);
    q.try_push_all(v);
    auto b = q.pop_batch(64, std::chrono::microseconds(0));
    benchmark::DoNotOptimize(b.data());
  }
  st.SetItemsProcessed(st.iterations() * 64);
}
BENCHMARK(BM_QueueRoundTrip)->Threads(1)->Threads(4);

// End to end through Pipeline: submit request-sized groups, wait for all to be scored.
void BM_PipelineEndToEnd(benchmark::State& st) {
  const std::size_t workers = static_cast<std::size_t>(st.range(0));
  const std::size_t group = 100;
  Metrics metrics;
  PipelineOptions o;
  o.workers = workers; o.queue_capacity = 1 << 16; o.max_batch = 64; o.max_wait = std::chrono::microseconds(500);
  o.threshold = meta().threshold;
  Pipeline p(scorer(), o, metrics);
  auto x = random_rows(group);
  for (auto _ : st) {
    std::vector<EventRecord> ev(group);
    for (std::size_t i = 0; i < group; ++i) ev[i].features.assign(x.begin() + i * 6, x.begin() + (i + 1) * 6);
    auto c = std::make_shared<Completion>(group);
    while (!p.submit(ev, c)) {}
    c->wait_for(std::chrono::seconds(10));
  }
  st.SetItemsProcessed(st.iterations() * static_cast<int64_t>(group));
  p.stop();
  st.counters["p99_us"] = metrics.latency.quantile(0.99) * 1e6;
  st.counters["mean_batch"] = static_cast<double>(metrics.batch_size_sum.load()) / static_cast<double>(std::max<uint64_t>(1, metrics.batches.load()));
}
BENCHMARK(BM_PipelineEndToEnd)->Arg(1)->Arg(2)->Arg(4)->UseRealTime();

}  // namespace

int main(int argc, char** argv) {
  if (!std::filesystem::exists(kDir + "/model.onnx")) {
    std::fprintf(stderr, "model not found in %s; run `make model` first\n", kDir.c_str());
    return 1;
  }
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
