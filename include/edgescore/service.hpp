// HTTP routing and request handling on top of the Pipeline.
#pragma once

#include <chrono>
#include <string>

#include "edgescore/http_server.hpp"
#include "edgescore/metrics.hpp"
#include "edgescore/model.hpp"
#include "edgescore/pipeline.hpp"

namespace edgescore {

struct ServiceOptions {
  std::string hec_token;  // empty => no auth
  std::chrono::milliseconds score_timeout{5000};
  std::size_t max_events_per_request = 100000;
};

class Service {
 public:
  Service(const ModelMeta& meta, Pipeline& pipeline, Metrics& metrics, ServiceOptions opts);
  HttpResponse handle(const HttpRequest& req);

 private:
  HttpResponse ingest_hec(const HttpRequest& req);
  HttpResponse ingest_async(const HttpRequest& req);
  HttpResponse score_sync(const HttpRequest& req);
  HttpResponse results(const HttpRequest& req);
  HttpResponse metrics_page();
  bool authorized(const HttpRequest& req, HttpResponse& deny) const;

  const ModelMeta& meta_;
  Pipeline& pipeline_;
  Metrics& metrics_;
  ServiceOptions opts_;
};

}  // namespace edgescore
