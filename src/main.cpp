#include <signal.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>

#include "edgescore/config.hpp"
#include "edgescore/http_server.hpp"
#include "edgescore/metrics.hpp"
#include "edgescore/model.hpp"
#include "edgescore/pipeline.hpp"
#include "edgescore/service.hpp"

int main(int argc, char** argv) {
  using namespace edgescore;
  if (argc > 1) {
    std::fprintf(stderr,
                 "edgescore: ONNX anomaly scoring sidecar\n"
                 "Configured via environment variables (see .env.example / README).\n");
    return std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0 ? 0 : 2;
  }

  // Block termination signals before any thread is created so every thread inherits the
  // mask and main() is the only one that receives them (via sigwait).
  sigset_t sigs;
  sigemptyset(&sigs);
  sigaddset(&sigs, SIGINT);
  sigaddset(&sigs, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &sigs, nullptr);

  try {
    Config cfg = Config::from_env();
    ModelMeta meta = load_meta(cfg.meta_path);
    std::shared_ptr<const Scorer> scorer = make_onnx_scorer(cfg.model_path, meta, cfg.intra_op_threads);

    Metrics metrics;
    PipelineOptions po;
    po.workers = cfg.workers;
    po.queue_capacity = cfg.queue_capacity;
    po.max_batch = cfg.max_batch;
    po.max_wait = cfg.max_wait;
    po.threshold = meta.threshold;
    Pipeline pipeline(scorer, po, metrics);

    ServiceOptions so;
    so.hec_token = cfg.hec_token;
    so.score_timeout = cfg.score_timeout;
    Service service(meta, pipeline, metrics, so);

    HttpServerOptions ho;
    ho.bind_address = cfg.bind_address;
    ho.port = cfg.port;
    ho.max_connections = cfg.max_connections;
    ho.max_body_bytes = cfg.max_body_bytes;
    HttpServer server(ho, [&](const HttpRequest& r) { return service.handle(r); });
    server.start();
    std::fprintf(stderr,
                 "edgescore: model '%s' (%zu features, threshold %.4f) listening on %s:%u, %zu workers, "
                 "queue %zu, batch<=%zu, hec auth %s\n",
                 meta.name.c_str(), meta.features.size(), meta.threshold, cfg.bind_address.c_str(),
                 static_cast<unsigned>(server.port()), cfg.workers, cfg.queue_capacity, cfg.max_batch,
                 cfg.hec_token.empty() ? "off" : "on");

    int sig = 0;
    sigwait(&sigs, &sig);
    std::fprintf(stderr, "edgescore: signal %d, draining...\n", sig);
    server.stop();    // no new requests; in-flight ones finish
    pipeline.stop();  // score everything already queued
    std::fprintf(stderr, "edgescore: scored %llu events, bye\n",
                 static_cast<unsigned long long>(metrics.events_scored.load()));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "edgescore: fatal: %s\n", e.what());
    return 1;
  }
  return 0;
}
