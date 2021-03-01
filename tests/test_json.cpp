#include <gtest/gtest.h>

#include "edgescore/json.hpp"

using namespace edgescore::json;

TEST(Json, ParsesScalarsAndContainers) {
  Value v = parse(R"({"a":1.5,"b":[true,false,null],"c":"x\ny","d":{"e":-2e2}})");
  ASSERT_TRUE(v.is_object());
  EXPECT_DOUBLE_EQ(v.find("a")->as_number(), 1.5);
  EXPECT_EQ(v.find("b")->items().size(), 3u);
  EXPECT_TRUE(v.find("b")->items()[0].as_bool());
  EXPECT_TRUE(v.find("b")->items()[2].is_null());
  EXPECT_EQ(v.find("c")->as_string(), "x\ny");
  EXPECT_DOUBLE_EQ(v.find("d")->find("e")->as_number(), -200.0);
  EXPECT_EQ(v.find("missing"), nullptr);
}

TEST(Json, UnicodeEscapesIncludingSurrogatePairs) {
  EXPECT_EQ(parse(R"("é")").as_string(), "\xC3\xA9");
  EXPECT_EQ(parse(R"("😀")").as_string(), "\xF0\x9F\x98\x80");
  EXPECT_THROW(parse(R"("\ud83d")"), ParseError);
  EXPECT_THROW(parse(R"("\ude00")"), ParseError);
}

TEST(Json, RejectsMalformedInput) {
  for (const char* bad : {"", "{", "[1,]", R"({"a":})", R"({"a" 1})", "01", "1.", "1e", "+1", "nul", "\"abc",
                          "{'a':1}", "[1 2]", "NaN", "1e999", "\"a\tb\""}) {
    EXPECT_THROW(parse(bad), ParseError) << bad;
  }
  EXPECT_THROW(parse("1 2"), ParseError);  // trailing garbage
}

TEST(Json, DepthLimitPreventsStackOverflow) {
  std::string deep(10000, '[');
  EXPECT_THROW(parse(deep), ParseError);
}

TEST(Json, StreamOfConcatenatedValues) {
  std::string body = "{\"a\":1}{\"a\":2}\n\n  {\"a\":3}\r\n";
  std::size_t pos = 0;
  Value v;
  std::vector<double> got;
  while (parse_next(body, pos, v)) got.push_back(v.find("a")->as_number());
  EXPECT_EQ(got, (std::vector<double>{1, 2, 3}));
}

TEST(Json, EscapeRoundTrips) {
  std::string raw = "quote\" back\\ nl\n tab\t ctl\x01";
  EXPECT_EQ(parse("\"" + escape(raw) + "\"").as_string(), raw);
}

TEST(Json, DuplicateKeyLastWins) {
  EXPECT_DOUBLE_EQ(parse(R"({"a":1,"a":2})").find("a")->as_number(), 2.0);
}
