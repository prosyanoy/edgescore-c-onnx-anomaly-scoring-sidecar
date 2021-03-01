#include <gtest/gtest.h>

#include <thread>

#include "edgescore/metrics.hpp"

using namespace edgescore;

TEST(Histogram, QuantilesAreWithinBucketResolution) {
  LatencyHistogram h;
  for (int i = 1; i <= 1000; ++i) h.observe(i * 1e-3);  // 1 ms .. 1 s uniformly
  EXPECT_EQ(h.count(), 1000u);
  EXPECT_NEAR(h.quantile(0.5), 0.5, 0.5 * 0.2);
  EXPECT_NEAR(h.quantile(0.99), 0.99, 0.99 * 0.2);
  EXPECT_LE(h.quantile(0.5), h.quantile(0.99));
  EXPECT_NEAR(h.sum_seconds(), 500.5, 0.5);
}

TEST(Histogram, EmptyAndExtremes) {
  LatencyHistogram h;
  EXPECT_EQ(h.quantile(0.99), 0.0);
  h.observe(0.0);
  h.observe(-1.0);
  h.observe(1e6);  // clamps into the last bucket
  EXPECT_EQ(h.count(), 3u);
  EXPECT_GT(h.quantile(1.0), 10.0);
}

TEST(Histogram, ConcurrentObserversDoNotLoseCounts) {
  LatencyHistogram h;
  std::vector<std::thread> ts;
  for (int t = 0; t < 4; ++t) ts.emplace_back([&] { for (int i = 0; i < 10000; ++i) h.observe(1e-4 * (i % 50 + 1)); });
  for (auto& t : ts) t.join();
  EXPECT_EQ(h.count(), 40000u);
  std::uint64_t sum = 0;
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets; ++i) sum += h.bucket_count(i);
  EXPECT_EQ(sum, 40000u);
}

TEST(RateWindow, AveragesCompletedSeconds) {
  RateWindow w(10);
  using C = RateWindow::Clock;
  C::time_point t0{std::chrono::seconds(1000)};
  for (int s = 0; s < 5; ++s) w.add(100, t0 + std::chrono::seconds(s));
  EXPECT_DOUBLE_EQ(w.per_second(t0 + std::chrono::seconds(5)), 100.0);
  // Idle seconds drag the average down, old data ages out of the window.
  EXPECT_DOUBLE_EQ(w.per_second(t0 + std::chrono::seconds(10)), 50.0);
  EXPECT_DOUBLE_EQ(w.per_second(t0 + std::chrono::seconds(60)), 0.0);
}

TEST(Metrics, PrometheusExpositionContainsKeySeries) {
  Metrics m;
  m.events_scored = 42;
  m.rejected_backpressure = 3;
  m.latency.observe(0.002);
  std::string text = m.render_prometheus(7, 100);
  for (const char* needle : {"edgescore_events_scored_total 42", "edgescore_events_rejected_total{reason=\"backpressure\"} 3",
                             "edgescore_queue_depth 7", "edgescore_queue_capacity 100", "edgescore_latency_p50_seconds",
                             "edgescore_latency_p99_seconds", "edgescore_latency_seconds_bucket{le=\"+Inf\"} 1",
                             "edgescore_throughput_events_per_second", "# TYPE edgescore_latency_seconds histogram"})
    EXPECT_NE(text.find(needle), std::string::npos) << needle << "\n" << text;
}
