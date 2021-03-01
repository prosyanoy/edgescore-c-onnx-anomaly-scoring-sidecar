#include "edgescore/model.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "edgescore/json.hpp"

namespace edgescore {

ModelMeta parse_meta(const std::string& text) {
  json::Value doc;
  try {
    doc = json::parse(text);
  } catch (const json::ParseError& e) {
    throw std::runtime_error(std::string("model metadata is not valid JSON: ") + e.what());
  }
  if (!doc.is_object()) throw std::runtime_error("model metadata must be a JSON object");
  ModelMeta m;
  m.raw_json = text;
  if (auto* v = doc.find("name"); v && v->is_string()) m.name = v->as_string();
  auto* feats = doc.find("features");
  if (!feats || !feats->is_array() || feats->items().empty())
    throw std::runtime_error("model metadata needs a non-empty \"features\" array");
  for (const auto& f : feats->items()) {
    if (!f.is_string()) throw std::runtime_error("\"features\" must contain strings");
    m.features.push_back(f.as_string());
  }
  if (auto* v = doc.find("input_name"); v && v->is_string()) m.input_name = v->as_string();
  if (auto* v = doc.find("score_output"); v && v->is_string()) m.score_output = v->as_string();
  if (auto* v = doc.find("score_sign"); v && v->is_number()) m.score_sign = v->as_number();
  auto* th = doc.find("threshold");
  if (!th || !th->is_number()) throw std::runtime_error("model metadata needs a numeric \"threshold\"");
  m.threshold = th->as_number();
  return m;
}

ModelMeta load_meta(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open model metadata: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return parse_meta(ss.str());
}

namespace {

class OnnxScorer final : public Scorer {
 public:
  OnnxScorer(const std::string& path, const ModelMeta& meta, int intra_threads)
      : env_(ORT_LOGGING_LEVEL_WARNING, "edgescore"),
        meta_(meta),
        mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(intra_threads);
    so.SetInterOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), so);

    Ort::AllocatorWithDefaultOptions alloc;
    if (session_->GetInputCount() != 1) throw std::runtime_error("model must have exactly one input");
    auto in_name = session_->GetInputNameAllocated(0, alloc);
    if (meta_.input_name != in_name.get())
      throw std::runtime_error(std::string("model input is '") + in_name.get() + "', metadata says '" +
                               meta_.input_name + "'");
    auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2) throw std::runtime_error("model input must be rank 2 [batch, features]");
    if (shape[1] > 0 && static_cast<std::size_t>(shape[1]) != meta_.features.size())
      throw std::runtime_error("model expects " + std::to_string(shape[1]) + " features but metadata lists " +
                               std::to_string(meta_.features.size()));
    bool found = false;
    for (std::size_t i = 0; i < session_->GetOutputCount(); ++i)
      if (meta_.score_output == session_->GetOutputNameAllocated(i, alloc).get()) found = true;
    if (!found) throw std::runtime_error("model has no output named '" + meta_.score_output + "'");
  }

  std::size_t num_features() const override { return meta_.features.size(); }

  void score(const float* x, std::size_t n, float* out) const override {
    if (n == 0) return;
    const std::size_t f = meta_.features.size();
    std::array<int64_t, 2> dims{static_cast<int64_t>(n), static_cast<int64_t>(f)};
    Ort::Value input = Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(x), n * f, dims.data(), dims.size());
    const char* in_names[] = {meta_.input_name.c_str()};
    const char* out_names[] = {meta_.score_output.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    auto info = outputs[0].GetTensorTypeAndShapeInfo();
    if (info.GetElementCount() != n) throw std::runtime_error("unexpected score output size");
    const float* raw = outputs[0].GetTensorData<float>();
    const float sign = static_cast<float>(meta_.score_sign);
    for (std::size_t i = 0; i < n; ++i) out[i] = sign * raw[i];
  }

 private:
  Ort::Env env_;
  ModelMeta meta_;
  Ort::MemoryInfo mem_;
  std::unique_ptr<Ort::Session> session_;
};

}  // namespace

std::unique_ptr<Scorer> make_onnx_scorer(const std::string& path, const ModelMeta& meta, int intra_threads) {
  try {
    return std::make_unique<OnnxScorer>(path, meta, intra_threads);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error(std::string("onnxruntime: ") + e.what());
  }
}

}  // namespace edgescore
