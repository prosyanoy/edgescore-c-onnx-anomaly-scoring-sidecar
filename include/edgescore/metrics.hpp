// Lock-free-on-the-hot-path metrics with Prometheus text exposition.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

namespace edgescore {

// Log-spaced latency histogram (1 us .. ~65 s, 4 buckets per octave-ish) with
// atomic counters; quantiles are interpolated within a bucket.
class LatencyHistogram {
 public:
  static constexpr std::size_t kBuckets = 96;

  void observe(double seconds) noexcept;
  std::uint64_t count() const noexcept { return count_.load(std::memory_order_relaxed); }
  double sum_seconds() const noexcept { return sum_ns_.load(std::memory_order_relaxed) * 1e-9; }
  // q in [0,1]; returns 0 when empty. Upper bound of bucket is returned
  // (linearly interpolated), so error is bounded by the bucket width (~19%).
  double quantile(double q) const noexcept;
  // Upper bound of bucket i in seconds.
  static double upper_bound(std::size_t i) noexcept;
  std::uint64_t bucket_count(std::size_t i) const noexcept { return buckets_[i].load(std::memory_order_relaxed); }

 private:
  static std::size_t index_for(double seconds) noexcept;
  std::array<std::atomic<std::uint64_t>, kBuckets> buckets_{};
  std::atomic<std::uint64_t> count_{0};
  std::atomic<std::uint64_t> sum_ns_{0};
};

// Events/second over a sliding window of 1 s slots.
class RateWindow {
 public:
  using Clock = std::chrono::steady_clock;
  explicit RateWindow(int window_seconds = 10) : window_(window_seconds) {}
  void add(std::uint64_t n, Clock::time_point now = Clock::now());
  double per_second(Clock::time_point now = Clock::now()) const;

 private:
  static constexpr int kMaxSlots = 64;
  int window_;
  mutable std::mutex mu_;
  std::array<std::uint64_t, kMaxSlots> counts_{};
  std::array<std::int64_t, kMaxSlots> stamps_{};  // second index each slot belongs to
  std::int64_t first_sec_ = -1;
};

struct Metrics {
  std::atomic<std::uint64_t> events_received{0};   // parsed and accepted into the queue
  std::atomic<std::uint64_t> events_scored{0};
  std::atomic<std::uint64_t> anomalies{0};
  std::atomic<std::uint64_t> batches{0};
  std::atomic<std::uint64_t> rejected_backpressure{0};  // events refused because the queue was full
  std::atomic<std::uint64_t> rejected_invalid{0};       // events refused as malformed
  std::atomic<std::uint64_t> requests{0};
  std::atomic<std::uint64_t> inference_errors{0};
  std::atomic<std::uint64_t> batch_size_sum{0};
  LatencyHistogram latency;            // enqueue -> scored, per event
  LatencyHistogram inference_latency;  // model run, per batch
  RateWindow throughput{10};

  // `queue_depth`/`queue_capacity` are sampled by the caller at scrape time.
  std::string render_prometheus(std::size_t queue_depth, std::size_t queue_capacity) const;
};

}  // namespace edgescore
