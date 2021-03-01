#include <gtest/gtest.h>

#include <atomic>
#include <numeric>
#include <thread>

#include "edgescore/bounded_queue.hpp"

using namespace edgescore;
using namespace std::chrono_literals;

TEST(BoundedQueue, PushAllIsAtomicAndBounded) {
  BoundedQueue<int> q(5);
  std::vector<int> a{1, 2, 3};
  EXPECT_TRUE(q.try_push_all(a));
  EXPECT_TRUE(a.empty());
  std::vector<int> b{4, 5, 6};  // would exceed capacity: nothing may be enqueued
  EXPECT_FALSE(q.try_push_all(b));
  EXPECT_EQ(b.size(), 3u);
  EXPECT_EQ(q.size(), 3u);
  std::vector<int> c{4, 5};
  EXPECT_TRUE(q.try_push_all(c));
  EXPECT_EQ(q.size(), 5u);
}

TEST(BoundedQueue, PopBatchHonoursMaxBatchAndOrder) {
  BoundedQueue<int> q(100);
  std::vector<int> v(10);
  std::iota(v.begin(), v.end(), 0);
  ASSERT_TRUE(q.try_push_all(v));
  auto b1 = q.pop_batch(4, 1ms);
  EXPECT_EQ(b1, (std::vector<int>{0, 1, 2, 3}));
  auto b2 = q.pop_batch(100, 1ms);
  EXPECT_EQ(b2.size(), 6u);
  EXPECT_EQ(b2.front(), 4);
}

TEST(BoundedQueue, PartialBatchReturnsAfterDeadline) {
  BoundedQueue<int> q(10);
  std::vector<int> v{1};
  q.try_push_all(v);
  auto t0 = std::chrono::steady_clock::now();
  auto b = q.pop_batch(8, 20ms);
  auto dt = std::chrono::steady_clock::now() - t0;
  EXPECT_EQ(b.size(), 1u);
  EXPECT_GE(dt, 15ms);
  EXPECT_LT(dt, 2s);
}

TEST(BoundedQueue, CloseWakesConsumersAfterDrain) {
  BoundedQueue<int> q(10);
  std::vector<int> v{1, 2};
  q.try_push_all(v);
  std::thread t([&] {
    EXPECT_EQ(q.pop_batch(8, 1s).size(), 2u);  // closed => does not wait for the full second
    EXPECT_TRUE(q.pop_batch(8, 1s).empty());
  });
  q.close();
  t.join();
  std::vector<int> late{3};
  EXPECT_FALSE(q.try_push_all(late));
}

TEST(BoundedQueue, ManyProducersManyConsumersLoseNothing) {
  constexpr int kProducers = 4, kPerProducer = 5000;
  BoundedQueue<int> q(256);
  std::atomic<long long> sum{0};
  std::atomic<int> count{0};
  std::vector<std::thread> consumers, producers;
  for (int i = 0; i < 3; ++i)
    consumers.emplace_back([&] {
      for (;;) {
        auto b = q.pop_batch(16, 200us);
        if (b.empty()) return;
        for (int x : b) { sum += x; ++count; }
      }
    });
  for (int p = 0; p < kProducers; ++p)
    producers.emplace_back([&, p] {
      for (int i = 0; i < kPerProducer; ++i) {
        std::vector<int> one{p * kPerProducer + i};
        while (!q.try_push_all(one)) std::this_thread::yield();  // backpressure: retry
      }
    });
  for (auto& t : producers) t.join();
  q.close();
  for (auto& t : consumers) t.join();
  const long long n = kProducers * kPerProducer;
  EXPECT_EQ(count.load(), n);
  EXPECT_EQ(sum.load(), n * (n - 1) / 2);
}
