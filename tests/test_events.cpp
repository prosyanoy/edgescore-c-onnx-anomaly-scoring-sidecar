#include <gtest/gtest.h>

#include "edgescore/events.hpp"

using namespace edgescore;

static const std::vector<std::string> kFeat{"cpu", "mem"};

TEST(Events, BareNdjson) {
  auto r = parse_events("{\"cpu\":1,\"mem\":2}\n{\"cpu\":3.5,\"mem\":4,\"id\":\"abc\"}\n", kFeat);
  ASSERT_TRUE(r.ok) << r.error;
  ASSERT_EQ(r.events.size(), 2u);
  EXPECT_EQ(r.events[0].features, (std::vector<float>{1, 2}));
  EXPECT_EQ(r.events[1].id, "abc");
}

TEST(Events, HecEnvelopeWithEventAndFields) {
  auto r = parse_events(
      R"({"time":1,"host":"h1","source":"s","event":{"cpu":10,"id":"e1"},"fields":{"mem":"20.5"}}{"event":{"cpu":1,"mem":2}})",
      kFeat);
  ASSERT_TRUE(r.ok) << r.error;
  ASSERT_EQ(r.events.size(), 2u);
  EXPECT_EQ(r.events[0].features, (std::vector<float>{10, 20.5f}));
  EXPECT_EQ(r.events[0].host, "h1");
  EXPECT_EQ(r.events[0].source, "s");
  EXPECT_EQ(r.events[0].id, "e1");
}

TEST(Events, EmptyBodyYieldsNoEvents) {
  auto r = parse_events("  \n ", kFeat);
  EXPECT_TRUE(r.ok);
  EXPECT_TRUE(r.events.empty());
}

TEST(Events, ErrorsReportIndexAndDropEverything) {
  auto r = parse_events("{\"cpu\":1,\"mem\":2}\n{\"cpu\":1}\n", kFeat);
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.error_index, 1u);
  EXPECT_TRUE(r.events.empty());
  EXPECT_NE(r.error.find("mem"), std::string::npos);
}

TEST(Events, RejectsBadTypesAndShapes) {
  EXPECT_FALSE(parse_events(R"({"cpu":"abc","mem":1})", kFeat).ok);
  EXPECT_FALSE(parse_events(R"({"cpu":null,"mem":1})", kFeat).ok);
  EXPECT_FALSE(parse_events(R"({"cpu":1e39,"mem":1})", kFeat).ok);
  EXPECT_FALSE(parse_events(R"({"event":"just a string"})", kFeat).ok);
  EXPECT_FALSE(parse_events("[1,2]", kFeat).ok);
  EXPECT_FALSE(parse_events("{\"cpu\":1,", kFeat).ok);
}

TEST(Events, EnforcesMaxEvents) {
  EXPECT_FALSE(parse_events("{\"cpu\":1,\"mem\":1}{\"cpu\":1,\"mem\":1}", kFeat, 1).ok);
}
