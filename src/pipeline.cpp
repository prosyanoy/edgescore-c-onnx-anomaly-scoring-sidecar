#include "edgescore/pipeline.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>

namespace edgescore {

void Completion::set(std::size_t idx, ScoredEvent ev) {
  bool done;
  {
    std::lock_guard<std::mutex> lk(mu_);
    results_[idx] = std::move(ev);
    done = --remaining_ == 0;
  }
  if (done) cv_.notify_all();
}

bool Completion::wait_for(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lk(mu_);
  return cv_.wait_for(lk, timeout, [&] { return remaining_ == 0; });
}

Pipeline::Pipeline(std::shared_ptr<const Scorer> scorer, PipelineOptions opts, Metrics& metrics)
    : scorer_(std::move(scorer)), opts_(opts), metrics_(metrics), queue_(opts.queue_capacity) {
  opts_.workers = std::max<std::size_t>(1, opts_.workers);
  opts_.max_batch = std::max<std::size_t>(1, opts_.max_batch);
  for (std::size_t i = 0; i < opts_.workers; ++i) workers_.emplace_back([this] { worker_loop(); });
}

Pipeline::~Pipeline() { stop(); }

void Pipeline::stop() {
  std::lock_guard<std::mutex> lk(stop_mu_);
  if (stopped_.exchange(true)) return;
  queue_.close();
  for (auto& t : workers_) t.join();
  workers_.clear();
}

bool Pipeline::submit(std::vector<EventRecord>& events, std::shared_ptr<Completion> completion) {
  if (events.empty()) return true;
  const auto now = std::chrono::steady_clock::now();
  const std::uint64_t base = next_seq_.fetch_add(events.size(), std::memory_order_relaxed);
  std::vector<WorkItem> items;
  items.reserve(events.size());
  for (std::size_t i = 0; i < events.size(); ++i)
    items.push_back(WorkItem{std::move(events[i]), base + i, now, completion, i});
  if (!queue_.try_push_all(items)) {
    // Rejected groups are left untouched by the queue: hand the events back to the caller.
    for (std::size_t i = 0; i < items.size(); ++i) events[i] = std::move(items[i].ev);
    return false;
  }
  metrics_.events_received.fetch_add(events.size(), std::memory_order_relaxed);
  return true;
}

void Pipeline::worker_loop() {
  const std::size_t f = scorer_->num_features();
  std::vector<float> x, scores;
  for (;;) {
    std::vector<WorkItem> batch = queue_.pop_batch(opts_.max_batch, opts_.max_wait);
    if (batch.empty()) return;  // closed and drained
    const std::size_t n = batch.size();
    x.resize(n * f);
    scores.assign(n, 0.f);
    for (std::size_t i = 0; i < n; ++i) std::copy(batch[i].ev.features.begin(), batch[i].ev.features.end(), x.begin() + i * f);

    bool failed = false;
    auto t0 = std::chrono::steady_clock::now();
    try {
      scorer_->score(x.data(), n, scores.data());
    } catch (const std::exception& e) {
      failed = true;
      std::fprintf(stderr, "edgescore: inference failed for batch of %zu: %s\n", n, e.what());
    }
    auto t1 = std::chrono::steady_clock::now();

    std::vector<ScoredEvent> done;
    done.reserve(n);
    std::uint64_t anomalies = 0;
    for (std::size_t i = 0; i < n; ++i) {
      ScoredEvent se;
      se.seq = batch[i].seq;
      se.id = std::move(batch[i].ev.id);
      se.host = std::move(batch[i].ev.host);
      se.failed = failed;
      se.score = scores[i];
      se.anomaly = !failed && static_cast<double>(scores[i]) > opts_.threshold;
      anomalies += se.anomaly ? 1 : 0;
      if (batch[i].completion) batch[i].completion->set(batch[i].index, se);
      metrics_.latency.observe(std::chrono::duration<double>(std::chrono::steady_clock::now() - batch[i].enqueued).count());
      if (!failed) done.push_back(std::move(se));
    }
    metrics_.inference_latency.observe(std::chrono::duration<double>(t1 - t0).count());
    metrics_.batches.fetch_add(1, std::memory_order_relaxed);
    metrics_.batch_size_sum.fetch_add(n, std::memory_order_relaxed);
    if (failed) {
      metrics_.inference_errors.fetch_add(1, std::memory_order_relaxed);
    } else {
      metrics_.events_scored.fetch_add(n, std::memory_order_relaxed);
      metrics_.anomalies.fetch_add(anomalies, std::memory_order_relaxed);
      metrics_.throughput.add(n);
      remember(std::move(done));
    }
  }
}

void Pipeline::remember(std::vector<ScoredEvent>&& batch) {
  if (opts_.recent_capacity == 0) return;
  std::lock_guard<std::mutex> lk(recent_mu_);
  for (auto& e : batch) {
    if (recent_.size() >= opts_.recent_capacity) recent_.pop_front();
    recent_.push_back(std::move(e));
  }
}

std::vector<ScoredEvent> Pipeline::recent(std::size_t limit, bool only_anomalies) const {
  std::vector<ScoredEvent> out;
  std::lock_guard<std::mutex> lk(recent_mu_);
  for (auto it = recent_.rbegin(); it != recent_.rend() && out.size() < limit; ++it)
    if (!only_anomalies || it->anomaly) out.push_back(*it);
  return out;
}

}  // namespace edgescore
