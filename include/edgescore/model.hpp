// Model abstraction: a Scorer maps feature rows to anomaly scores (higher == more anomalous).
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace edgescore {

struct ModelMeta {
  std::string name;
  std::vector<std::string> features;  // input column order
  std::string input_name = "X";
  std::string score_output = "scores";
  double score_sign = 1.0;            // multiplied into the raw model output
  double threshold = 0.0;             // anomaly iff score > threshold
  std::string raw_json;               // verbatim metadata, served at /v1/model
};

// Loads the metadata sidecar written by training/train.py. Throws std::runtime_error.
ModelMeta load_meta(const std::string& path);
ModelMeta parse_meta(const std::string& json_text);

class Scorer {
 public:
  virtual ~Scorer() = default;
  virtual std::size_t num_features() const = 0;
  // `x` is row-major [n, num_features()]; writes n scores to `out`.
  // Must be safe to call concurrently from multiple threads. Throws on failure.
  virtual void score(const float* x, std::size_t n, float* out) const = 0;
};

// ONNX Runtime (CPU) backed scorer. A single Ort::Session is shared by all workers
// (Session::Run is thread-safe); intra-op parallelism is configurable.
std::unique_ptr<Scorer> make_onnx_scorer(const std::string& onnx_path, const ModelMeta& meta,
                                         int intra_op_threads);

}  // namespace edgescore
