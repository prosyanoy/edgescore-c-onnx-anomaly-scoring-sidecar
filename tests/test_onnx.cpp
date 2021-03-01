// Exercises the real ONNX model produced by training/train.py (make model).
#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <thread>
#include <vector>

#include "edgescore/model.hpp"

using namespace edgescore;

namespace {
const std::string kDir = EDGESCORE_MODEL_DIR;
bool model_present() { return std::filesystem::exists(kDir + "/model.onnx") && std::filesystem::exists(kDir + "/model.json"); }
}  // namespace

TEST(Onnx, MetadataParsing) {
  auto m = parse_meta(R"({"name":"x","features":["a","b"],"threshold":0.25,"score_sign":-1})");
  EXPECT_EQ(m.features.size(), 2u);
  EXPECT_DOUBLE_EQ(m.threshold, 0.25);
  EXPECT_DOUBLE_EQ(m.score_sign, -1.0);
  EXPECT_THROW(parse_meta("{"), std::runtime_error);
  EXPECT_THROW(parse_meta(R"({"features":[],"threshold":1})"), std::runtime_error);
  EXPECT_THROW(parse_meta(R"({"features":["a"]})"), std::runtime_error);
}

TEST(Onnx, MissingFilesGiveClearErrors) {
  EXPECT_THROW(load_meta("/definitely/not/here.json"), std::runtime_error);
  ModelMeta m = parse_meta(R"({"features":["a"],"threshold":1})");
  EXPECT_THROW(make_onnx_scorer("/definitely/not/here.onnx", m, 1), std::runtime_error);
}

TEST(Onnx, SeparatesNormalFromAnomalousTelemetry) {
  if (!model_present()) GTEST_SKIP() << "run `make model` first";
  ModelMeta meta = load_meta(kDir + "/model.json");
  auto scorer = make_onnx_scorer(kDir + "/model.onnx", meta, 1);
  ASSERT_EQ(scorer->num_features(), 6u);
  // cpu, mem, latency_ms, error_rate, req_rate, bytes_out_kb
  const std::vector<float> rows = {
      50, 45, 52, 0.2f, 200, 800,    // typical
      52, 47, 55, 0.3f, 190, 780,    // typical
      98, 45, 160, 0.2f, 200, 800,   // CPU saturation
      50, 45, 900, 25.f, 200, 800,   // downstream failure
      20, 45, 50, 0.2f, 5, 30,       // traffic collapse
      50, 98, 75, 0.2f, 200, 800,    // memory leak (with the GC latency it causes)
  };
  std::vector<float> s(6);
  scorer->score(rows.data(), 6, s.data());
  EXPECT_LE(s[0], meta.threshold);
  EXPECT_LE(s[1], meta.threshold);
  for (int i = 2; i < 6; ++i) EXPECT_GT(s[i], meta.threshold) << "row " << i << " score " << s[i];
  // Row results must not depend on batch composition.
  std::vector<float> single(1);
  scorer->score(rows.data() + 2 * 6, 1, single.data());
  EXPECT_NEAR(single[0], s[2], 1e-6);
}

TEST(Onnx, ConcurrentCallsAreSafe) {
  if (!model_present()) GTEST_SKIP() << "run `make model` first";
  ModelMeta meta = load_meta(kDir + "/model.json");
  auto scorer = make_onnx_scorer(kDir + "/model.onnx", meta, 1);
  const std::vector<float> row = {50, 45, 52, 0.2f, 200, 800};
  float expected;
  scorer->score(row.data(), 1, &expected);
  std::vector<std::thread> ts;
  std::atomic<int> bad{0};
  for (int t = 0; t < 4; ++t)
    ts.emplace_back([&] {
      for (int i = 0; i < 200; ++i) { float o; scorer->score(row.data(), 1, &o); if (o != expected) ++bad; }
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(bad.load(), 0);
}
