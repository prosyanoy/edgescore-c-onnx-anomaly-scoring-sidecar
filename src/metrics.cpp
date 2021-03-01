#include "edgescore/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace edgescore {
namespace {
constexpr double kBase = 1e-6;       // first bucket upper bound: 1 us
constexpr double kGrowth = 1.1892;   // 2^(1/4)
}  // namespace

double LatencyHistogram::upper_bound(std::size_t i) noexcept {
  return kBase * std::pow(kGrowth, static_cast<double>(i));
}

std::size_t LatencyHistogram::index_for(double seconds) noexcept {
  if (!(seconds > kBase)) return 0;
  double idx = std::ceil(std::log(seconds / kBase) / std::log(kGrowth));
  if (idx >= static_cast<double>(kBuckets - 1)) return kBuckets - 1;
  std::size_t i = static_cast<std::size_t>(idx);
  // Guard against floating point rounding at bucket edges.
  while (i > 0 && seconds <= upper_bound(i - 1)) --i;
  while (i < kBuckets - 1 && seconds > upper_bound(i)) ++i;
  return i;
}

void LatencyHistogram::observe(double seconds) noexcept {
  if (seconds < 0) seconds = 0;
  buckets_[index_for(seconds)].fetch_add(1, std::memory_order_relaxed);
  count_.fetch_add(1, std::memory_order_relaxed);
  sum_ns_.fetch_add(static_cast<std::uint64_t>(seconds * 1e9), std::memory_order_relaxed);
}

double LatencyHistogram::quantile(double q) const noexcept {
  std::array<std::uint64_t, kBuckets> snap{};
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < kBuckets; ++i) {
    snap[i] = buckets_[i].load(std::memory_order_relaxed);
    total += snap[i];
  }
  if (total == 0) return 0.0;
  q = std::clamp(q, 0.0, 1.0);
  double rank = q * static_cast<double>(total);
  std::uint64_t cum = 0;
  for (std::size_t i = 0; i < kBuckets; ++i) {
    if (snap[i] == 0) continue;
    if (static_cast<double>(cum + snap[i]) >= rank) {
      double lo = i == 0 ? 0.0 : upper_bound(i - 1);
      double hi = upper_bound(i);
      double frac = (rank - static_cast<double>(cum)) / static_cast<double>(snap[i]);
      return lo + (hi - lo) * std::clamp(frac, 0.0, 1.0);
    }
    cum += snap[i];
  }
  return upper_bound(kBuckets - 1);
}

void RateWindow::add(std::uint64_t n, Clock::time_point now) {
  std::int64_t sec = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  std::lock_guard<std::mutex> lk(mu_);
  if (first_sec_ < 0) first_sec_ = sec;
  std::size_t slot = static_cast<std::size_t>(sec % kMaxSlots);
  if (stamps_[slot] != sec) {
    stamps_[slot] = sec;
    counts_[slot] = 0;
  }
  counts_[slot] += n;
}

double RateWindow::per_second(Clock::time_point now) const {
  std::int64_t sec = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
  std::lock_guard<std::mutex> lk(mu_);
  if (first_sec_ < 0) return 0.0;
  // Average over the last `window_` *completed* seconds (current partial second excluded),
  // but never over more time than has elapsed since the first sample.
  std::uint64_t total = 0;
  std::int64_t oldest = std::max(sec - window_, first_sec_);
  for (std::int64_t s = oldest; s < sec; ++s) {
    std::size_t slot = static_cast<std::size_t>(s % kMaxSlots);
    if (stamps_[slot] == s) total += counts_[slot];
  }
  std::int64_t span = sec - oldest;
  if (span <= 0) {  // still inside the very first second
    std::size_t slot = static_cast<std::size_t>(sec % kMaxSlots);
    return stamps_[slot] == sec ? static_cast<double>(counts_[slot]) : 0.0;
  }
  return static_cast<double>(total) / static_cast<double>(span);
}

namespace {
void counter(std::ostringstream& o, const char* name, const char* help, std::uint64_t v) {
  o << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " counter\n" << name << ' ' << v << '\n';
}
void gauge(std::ostringstream& o, const char* name, const char* help, double v) {
  o << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " gauge\n" << name << ' ' << v << '\n';
}
void histogram(std::ostringstream& o, const char* name, const char* help, const LatencyHistogram& h) {
  o << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " histogram\n";
  std::uint64_t cum = 0;
  // Emit every 2nd bucket boundary to keep the exposition compact.
  for (std::size_t i = 0; i < LatencyHistogram::kBuckets; ++i) {
    cum += h.bucket_count(i);
    if (i % 2 == 1 || i == LatencyHistogram::kBuckets - 1) {
      char le[32];
      std::snprintf(le, sizeof le, "%.9g", LatencyHistogram::upper_bound(i));
      o << name << "_bucket{le=\"" << le << "\"} " << cum << '\n';
    }
  }
  o << name << "_bucket{le=\"+Inf\"} " << h.count() << '\n';
  o << name << "_sum " << h.sum_seconds() << '\n' << name << "_count " << h.count() << '\n';
}
}  // namespace

std::string Metrics::render_prometheus(std::size_t queue_depth, std::size_t queue_capacity) const {
  std::ostringstream o;
  o.precision(9);
  counter(o, "edgescore_http_requests_total", "HTTP requests handled.", requests.load());
  counter(o, "edgescore_events_received_total", "Events accepted into the inference queue.", events_received.load());
  counter(o, "edgescore_events_scored_total", "Events scored by the model.", events_scored.load());
  counter(o, "edgescore_anomalies_total", "Events scored above the anomaly threshold.", anomalies.load());
  counter(o, "edgescore_batches_total", "Inference batches executed.", batches.load());
  counter(o, "edgescore_inference_errors_total", "Batches that failed in the model runtime.", inference_errors.load());
  o << "# HELP edgescore_events_rejected_total Events refused at ingest.\n"
       "# TYPE edgescore_events_rejected_total counter\n"
    << "edgescore_events_rejected_total{reason=\"backpressure\"} " << rejected_backpressure.load() << '\n'
    << "edgescore_events_rejected_total{reason=\"invalid\"} " << rejected_invalid.load() << '\n';
  gauge(o, "edgescore_queue_depth", "Events waiting for inference.", static_cast<double>(queue_depth));
  gauge(o, "edgescore_queue_capacity", "Configured queue capacity.", static_cast<double>(queue_capacity));
  gauge(o, "edgescore_throughput_events_per_second", "Scored events/s over the last 10 s.", throughput.per_second());
  auto nb = batches.load();
  gauge(o, "edgescore_batch_size_mean", "Mean events per inference batch.",
        nb ? static_cast<double>(batch_size_sum.load()) / static_cast<double>(nb) : 0.0);
  gauge(o, "edgescore_latency_p50_seconds", "Median enqueue-to-scored latency.", latency.quantile(0.50));
  gauge(o, "edgescore_latency_p99_seconds", "99th percentile enqueue-to-scored latency.", latency.quantile(0.99));
  histogram(o, "edgescore_latency_seconds", "Per-event latency from enqueue to scored.", latency);
  histogram(o, "edgescore_inference_seconds", "Model execution time per batch.", inference_latency);
  return o.str();
}

}  // namespace edgescore
