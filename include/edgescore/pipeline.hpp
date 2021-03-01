// Multi-threaded batching inference pipeline: bounded queue -> N workers -> Scorer.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "edgescore/bounded_queue.hpp"
#include "edgescore/events.hpp"
#include "edgescore/metrics.hpp"
#include "edgescore/model.hpp"

namespace edgescore {

struct ScoredEvent {
  std::uint64_t seq = 0;
  std::string id;
  std::string host;
  float score = 0.f;      // higher == more anomalous
  bool anomaly = false;
  bool failed = false;    // model run failed for this event's batch
};

// Lets a request thread wait for the scores of the events it submitted.
class Completion {
 public:
  explicit Completion(std::size_t n) : remaining_(n), results_(n) {}
  void set(std::size_t idx, ScoredEvent ev);
  // Returns true if all results arrived before the timeout.
  bool wait_for(std::chrono::milliseconds timeout);
  const std::vector<ScoredEvent>& results() const { return results_; }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  std::size_t remaining_;
  std::vector<ScoredEvent> results_;
};

struct PipelineOptions {
  std::size_t workers = 2;
  std::size_t queue_capacity = 8192;
  std::size_t max_batch = 64;
  std::chrono::microseconds max_wait{2000};
  double threshold = 0.0;
  std::size_t recent_capacity = 1024;
};

class Pipeline {
 public:
  Pipeline(std::shared_ptr<const Scorer> scorer, PipelineOptions opts, Metrics& metrics);
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  // All-or-nothing enqueue. Returns false when the queue lacks room (backpressure)
  // or the pipeline is stopping; in that case `events` is left intact.
  // If `completion` is given it must have been created with events.size().
  bool submit(std::vector<EventRecord>& events, std::shared_ptr<Completion> completion = nullptr);

  // Close intake, drain queued work, join workers. Idempotent.
  void stop();

  std::size_t queue_depth() const { return queue_.size(); }
  std::size_t queue_capacity() const { return queue_.capacity(); }
  bool running() const { return !stopped_.load(); }

  // Most recent scored events, newest first.
  std::vector<ScoredEvent> recent(std::size_t limit, bool only_anomalies) const;

 private:
  struct WorkItem {
    EventRecord ev;
    std::uint64_t seq;
    std::chrono::steady_clock::time_point enqueued;
    std::shared_ptr<Completion> completion;
    std::size_t index;
  };
  void worker_loop();
  void remember(std::vector<ScoredEvent>&& batch);

  std::shared_ptr<const Scorer> scorer_;
  PipelineOptions opts_;
  Metrics& metrics_;
  BoundedQueue<WorkItem> queue_;
  std::vector<std::thread> workers_;
  std::atomic<std::uint64_t> next_seq_{1};
  std::atomic<bool> stopped_{false};
  std::mutex stop_mu_;

  mutable std::mutex recent_mu_;
  std::deque<ScoredEvent> recent_;
};

}  // namespace edgescore
