#include <gtest/gtest.h>

#include <map>

#include "edgescore/config.hpp"

using namespace edgescore;

static Config from(std::map<std::string, std::string> env) {
  return Config::from_env([&](const char* k) -> const char* {
    auto it = env.find(k);
    return it == env.end() ? nullptr : it->second.c_str();
  });
}

TEST(Config, DefaultsAreSafeAndLocal) {
  Config c = from({});
  EXPECT_EQ(c.port, 8088);
  EXPECT_TRUE(c.hec_token.empty());
  EXPECT_EQ(c.model_path, "models/model.onnx");
  EXPECT_EQ(c.meta_path, "models/model.json");
}

TEST(Config, ReadsOverridesAndDerivesMetaPath) {
  Config c = from({{"EDGESCORE_PORT", "9000"}, {"EDGESCORE_MODEL", "/opt/m/x.onnx"}, {"EDGESCORE_WORKERS", "8"},
                   {"EDGESCORE_MAX_WAIT_US", "500"}, {"EDGESCORE_HEC_TOKEN", "t"}});
  EXPECT_EQ(c.port, 9000);
  EXPECT_EQ(c.workers, 8u);
  EXPECT_EQ(c.max_wait.count(), 500);
  EXPECT_EQ(c.meta_path, "/opt/m/model.json");
  EXPECT_EQ(c.hec_token, "t");
}

TEST(Config, RejectsGarbage) {
  EXPECT_THROW(from({{"EDGESCORE_PORT", "70000"}}), std::invalid_argument);
  EXPECT_THROW(from({{"EDGESCORE_PORT", "abc"}}), std::invalid_argument);
  EXPECT_THROW(from({{"EDGESCORE_WORKERS", "0"}}), std::invalid_argument);
  EXPECT_THROW(from({{"EDGESCORE_WORKERS", "-1"}}), std::invalid_argument);
  EXPECT_THROW(from({{"EDGESCORE_MAX_BATCH", "100"}, {"EDGESCORE_QUEUE_CAPACITY", "10"}}), std::invalid_argument);
}
