// Runtime configuration, read from environment variables with safe local defaults.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace edgescore {

struct Config {
  std::string model_path = "models/model.onnx";  // EDGESCORE_MODEL ; metadata is <same path>.json minus .onnx
  std::string meta_path;                         // EDGESCORE_MODEL_META (default: model.json beside the model)
  std::string bind_address = "0.0.0.0";          // EDGESCORE_BIND
  std::uint16_t port = 8088;                     // EDGESCORE_PORT
  std::size_t workers = 2;                       // EDGESCORE_WORKERS (inference threads)
  int intra_op_threads = 1;                      // EDGESCORE_INTRA_OP_THREADS
  std::size_t queue_capacity = 8192;             // EDGESCORE_QUEUE_CAPACITY
  std::size_t max_batch = 64;                    // EDGESCORE_MAX_BATCH
  std::chrono::microseconds max_wait{2000};      // EDGESCORE_MAX_WAIT_US
  std::size_t max_connections = 256;             // EDGESCORE_MAX_CONNECTIONS
  std::size_t max_body_bytes = 8u << 20;         // EDGESCORE_MAX_BODY_BYTES
  std::chrono::milliseconds score_timeout{5000}; // EDGESCORE_SCORE_TIMEOUT_MS
  std::string hec_token;                         // EDGESCORE_HEC_TOKEN (empty => auth disabled)

  using Getter = std::function<const char*(const char*)>;
  // Throws std::invalid_argument with a descriptive message on malformed values.
  static Config from_env(const Getter& get);
  static Config from_env();  // uses std::getenv
};

}  // namespace edgescore
