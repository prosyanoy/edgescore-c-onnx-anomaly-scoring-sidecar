// Small thread-per-connection HTTP/1.1 server (keep-alive, Content-Length and chunked bodies).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace edgescore {

struct HttpRequest {
  std::string method;
  std::string path;
  std::string query;                           // raw, without '?'
  std::map<std::string, std::string> headers;  // lower-cased names
  std::string body;

  std::string header(const std::string& name) const {
    auto it = headers.find(name);
    return it == headers.end() ? std::string() : it->second;
  }
  // Value of a query parameter (no percent-decoding beyond '+', parameters are numeric here).
  std::string query_param(const std::string& key) const;
};

struct HttpResponse {
  int status = 200;
  std::string content_type = "application/json";
  std::string body;
  std::vector<std::pair<std::string, std::string>> headers;
};

struct HttpServerOptions {
  std::string bind_address = "0.0.0.0";
  std::uint16_t port = 8088;  // 0 picks a free port (see HttpServer::port())
  std::size_t max_connections = 256;
  std::size_t max_body_bytes = 8u << 20;
  int idle_timeout_seconds = 30;
};

class HttpServer {
 public:
  using Handler = std::function<HttpResponse(const HttpRequest&)>;
  HttpServer(HttpServerOptions opts, Handler handler);
  ~HttpServer();
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  void start();  // throws std::runtime_error if the socket cannot be bound
  void stop();   // idempotent; waits for in-flight connections
  std::uint16_t port() const { return bound_port_; }

 private:
  void accept_loop();
  void serve(int fd);

  HttpServerOptions opts_;
  Handler handler_;
  int listen_fd_ = -1;
  std::uint16_t bound_port_ = 0;
  std::atomic<bool> stopping_{false};
  std::thread acceptor_;
  std::mutex mu_;
  std::condition_variable idle_cv_;
  std::set<int> clients_;  // open client fds, guarded by mu_
  std::size_t active_ = 0; // running connection threads, guarded by mu_
};

}  // namespace edgescore
