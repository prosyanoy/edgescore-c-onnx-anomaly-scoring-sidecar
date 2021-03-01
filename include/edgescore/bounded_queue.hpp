// Bounded multi-producer / multi-consumer queue with explicit backpressure.
//
// Producers never block: try_push_all() either enqueues the *whole* group or
// rejects it, so an HTTP request is accepted atomically or answered with 503.
// Consumers block in pop_batch() until work arrives, a batch fills, or the
// batching deadline expires. A mutex + condvar is deliberate here: contention
// is one lock acquisition per request (producers) / per batch (consumers),
// which is far below the cost of inference, and it keeps the code TSan-clean.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>

namespace edgescore {

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

  BoundedQueue(const BoundedQueue&) = delete;
  BoundedQueue& operator=(const BoundedQueue&) = delete;

  // Atomically enqueue all items. Returns false (and leaves `items` untouched)
  // if the queue is closed or lacks room for the whole group.
  bool try_push_all(std::vector<T>& items) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (closed_ || q_.size() + items.size() > capacity_) return false;
      for (auto& it : items) q_.push_back(std::move(it));
    }
    items.clear();
    not_empty_.notify_all();
    return true;
  }

  // Wait for at least one item, then keep collecting until `max_batch` items are
  // available or `max_wait` has elapsed since the first item was seen.
  // Returns an empty vector only when the queue is closed and drained.
  std::vector<T> pop_batch(std::size_t max_batch, std::chrono::microseconds max_wait) {
    std::vector<T> out;
    std::unique_lock<std::mutex> lk(mu_);
    not_empty_.wait(lk, [&] { return closed_ || !q_.empty(); });
    if (q_.empty()) return out;  // closed and drained
    auto deadline = std::chrono::steady_clock::now() + max_wait;
    while (!closed_ && q_.size() < max_batch) {
      if (not_empty_.wait_until(lk, deadline) == std::cv_status::timeout) break;
    }
    std::size_t n = q_.size() < max_batch ? q_.size() : max_batch;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      out.push_back(std::move(q_.front()));
      q_.pop_front();
    }
    // Leftover work may be waiting for another consumer.
    if (!q_.empty()) not_empty_.notify_one();
    return out;
  }

  // Reject further pushes; consumers drain what remains and then see empty batches.
  void close() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      closed_ = true;
    }
    not_empty_.notify_all();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.size();
  }
  std::size_t capacity() const noexcept { return capacity_; }

 private:
  const std::size_t capacity_;
  mutable std::mutex mu_;
  std::condition_variable not_empty_;
  std::deque<T> q_;
  bool closed_ = false;
};

}  // namespace edgescore
