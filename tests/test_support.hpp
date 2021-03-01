// Shared helpers for the test suite: a configurable fake model and a tiny raw-socket HTTP client.
#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>

#include "edgescore/model.hpp"

namespace edgescore::testing {

// Score of a row == value of its first feature. Can be gated to simulate a slow model.
class FakeScorer final : public Scorer {
 public:
  explicit FakeScorer(std::size_t features = 2) : features_(features) {}
  std::size_t num_features() const override { return features_; }
  void score(const float* x, std::size_t n, float* out) const override {
    {
      std::unique_lock<std::mutex> lk(mu_);
      ++entered_;
      cv_.notify_all();
      cv_.wait(lk, [&] { return !blocked_; });
    }
    batches_.fetch_add(1);
    if (fail_.load()) throw std::runtime_error("boom");
    for (std::size_t i = 0; i < n; ++i) out[i] = x[i * features_];
    rows_.fetch_add(n);
  }
  void block() { std::lock_guard<std::mutex> lk(mu_); blocked_ = true; }
  void unblock() { { std::lock_guard<std::mutex> lk(mu_); blocked_ = false; } cv_.notify_all(); }
  void wait_entered(int n) const {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return entered_ >= n; });
  }
  void set_fail(bool f) { fail_.store(f); }
  std::size_t rows() const { return rows_.load(); }
  std::size_t batches() const { return batches_.load(); }

 private:
  std::size_t features_;
  mutable std::mutex mu_;
  mutable std::condition_variable cv_;
  mutable int entered_ = 0;
  bool blocked_ = false;
  std::atomic<bool> fail_{false};
  mutable std::atomic<std::size_t> rows_{0}, batches_{0};
};

// Sends `raw` to 127.0.0.1:port and reads until the peer closes, or (unless `until_close`)
// until the first complete Content-Length response has arrived. Returns the raw response text.
inline std::string http_roundtrip(std::uint16_t port, const std::string& raw, bool until_close = false) {
  int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0) { ::close(fd); return {}; }
  timeval tv{10, 0};
  ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  ::send(fd, raw.data(), raw.size(), MSG_NOSIGNAL);
  std::string out;
  char buf[4096];
  for (;;) {
    ssize_t n = ::recv(fd, buf, sizeof buf, 0);
    if (n <= 0) break;
    out.append(buf, static_cast<std::size_t>(n));
    auto hdr = out.find("\r\n\r\n");
    if (!until_close && hdr != std::string::npos) {
      auto cl = out.find("Content-Length: ");
      if (cl != std::string::npos) {
        std::size_t len = std::stoul(out.substr(cl + 16));
        if (out.size() >= hdr + 4 + len) break;
      }
    }
  }
  ::close(fd);
  return out;
}

inline std::string post(const std::string& path, const std::string& body, const std::string& extra_headers = "") {
  return "POST " + path + " HTTP/1.1\r\nHost: x\r\nConnection: close\r\n" + extra_headers +
         "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
}
inline std::string get(const std::string& path) {
  return "GET " + path + " HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n";
}
inline std::string body_of(const std::string& resp) {
  auto p = resp.find("\r\n\r\n");
  return p == std::string::npos ? std::string() : resp.substr(p + 4);
}

}  // namespace edgescore::testing
