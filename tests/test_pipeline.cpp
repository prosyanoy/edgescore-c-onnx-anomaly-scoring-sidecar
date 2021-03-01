#include <gtest/gtest.h>

#include <set>
#include <thread>

#include "edgescore/pipeline.hpp"
#include "test_support.hpp"

using namespace edgescore;
using namespace edgescore::testing;
using namespace std::chrono_literals;

static std::vector<EventRecord> make_events(std::size_t n, float base = 0.f) {
  std::vector<EventRecord> v(n);
  for (std::size_t i = 0; i < n; ++i) { v[i].features = {base + static_cast<float>(i), 0.f}; v[i].id = "e" + std::to_string(i); }
  return v;
}

TEST(Pipeline, ScoresAndFlagsAnomaliesAgainstThreshold) {
  auto scorer = std::make_shared<FakeScorer>();
  Metrics m;
  PipelineOptions o;
  o.workers = 2; o.max_batch = 8; o.max_wait = 500us; o.threshold = 4.5;
  Pipeline p(scorer, o, m);
  auto ev = make_events(10);
  auto c = std::make_shared<Completion>(ev.size());
  ASSERT_TRUE(p.submit(ev, c));
  ASSERT_TRUE(c->wait_for(5s));
  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(c->results()[i].score, static_cast<float>(i));
    EXPECT_EQ(c->results()[i].anomaly, i >= 5) << i;
    EXPECT_EQ(c->results()[i].id, "e" + std::to_string(i));
  }
  p.stop();
  EXPECT_EQ(m.events_scored.load(), 10u);
  EXPECT_EQ(m.anomalies.load(), 5u);
  EXPECT_EQ(m.latency.count(), 10u);
  EXPECT_EQ(p.recent(100, true).size(), 5u);
  EXPECT_EQ(p.recent(3, false).size(), 3u);
}

TEST(Pipeline, BackpressureRejectsWholeGroupWhenQueueIsFull) {
  auto scorer = std::make_shared<FakeScorer>();
  scorer->block();
  Metrics m;
  PipelineOptions o;
  o.workers = 1; o.queue_capacity = 10; o.max_batch = 4; o.max_wait = 100us;
  Pipeline p(scorer, o, m);

  auto first = make_events(4);
  ASSERT_TRUE(p.submit(first));
  scorer->wait_entered(1);          // worker is now stuck inside the model holding 4 events
  auto fill = make_events(10);
  ASSERT_TRUE(p.submit(fill));      // queue now exactly full
  auto extra = make_events(1);
  EXPECT_FALSE(p.submit(extra));    // backpressure
  EXPECT_EQ(extra.size(), 1u);      // caller keeps its events
  EXPECT_EQ(extra[0].id, "e0");
  EXPECT_EQ(p.queue_depth(), 10u);

  scorer->unblock();
  p.stop();                         // drains
  EXPECT_EQ(m.events_scored.load(), 14u);
  EXPECT_EQ(scorer->rows(), 14u);
}

TEST(Pipeline, StopDrainsQueuedWorkAndRefusesNewWork) {
  auto scorer = std::make_shared<FakeScorer>();
  Metrics m;
  PipelineOptions o;
  o.workers = 2; o.max_wait = 50ms; o.max_batch = 1000;
  Pipeline p(scorer, o, m);
  for (int i = 0; i < 20; ++i) { auto ev = make_events(5); ASSERT_TRUE(p.submit(ev)); }
  p.stop();
  EXPECT_EQ(m.events_scored.load(), 100u);
  auto late = make_events(1);
  EXPECT_FALSE(p.submit(late));
  p.stop();  // idempotent
}

TEST(Pipeline, ModelFailureCompletesRequestsWithError) {
  auto scorer = std::make_shared<FakeScorer>();
  scorer->set_fail(true);
  Metrics m;
  PipelineOptions o;
  Pipeline p(scorer, o, m);
  auto ev = make_events(3);
  auto c = std::make_shared<Completion>(3);
  ASSERT_TRUE(p.submit(ev, c));
  ASSERT_TRUE(c->wait_for(5s));
  for (auto& r : c->results()) { EXPECT_TRUE(r.failed); EXPECT_FALSE(r.anomaly); }
  p.stop();
  EXPECT_GE(m.inference_errors.load(), 1u);
  EXPECT_EQ(m.events_scored.load(), 0u);
}

TEST(Pipeline, ConcurrentSubmittersAllGetScored) {
  auto scorer = std::make_shared<FakeScorer>();
  Metrics m;
  PipelineOptions o;
  o.workers = 4; o.queue_capacity = 512; o.max_batch = 32; o.max_wait = 200us;
  Pipeline p(scorer, o, m);
  constexpr int kThreads = 6, kReqs = 200, kPer = 5;
  std::atomic<int> accepted{0};
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t)
    ts.emplace_back([&] {
      for (int r = 0; r < kReqs; ++r) {
        auto ev = make_events(kPer);
        auto c = std::make_shared<Completion>(kPer);
        while (!p.submit(ev, c)) std::this_thread::yield();
        ++accepted;
        ASSERT_TRUE(c->wait_for(10s));
      }
    });
  for (auto& t : ts) t.join();
  p.stop();
  EXPECT_EQ(accepted.load(), kThreads * kReqs);
  EXPECT_EQ(m.events_scored.load(), static_cast<std::uint64_t>(kThreads * kReqs * kPer));
  EXPECT_EQ(m.events_received.load(), m.events_scored.load());
}

TEST(Pipeline, SequenceNumbersAreUnique) {
  auto scorer = std::make_shared<FakeScorer>();
  Metrics m;
  Pipeline p(scorer, PipelineOptions{}, m);
  auto a = make_events(3), b = make_events(3);
  auto ca = std::make_shared<Completion>(3), cb = std::make_shared<Completion>(3);
  p.submit(a, ca); p.submit(b, cb);
  ASSERT_TRUE(ca->wait_for(5s)); ASSERT_TRUE(cb->wait_for(5s));
  std::set<std::uint64_t> seen;
  for (auto& r : ca->results()) seen.insert(r.seq);
  for (auto& r : cb->results()) seen.insert(r.seq);
  EXPECT_EQ(seen.size(), 6u);
}
