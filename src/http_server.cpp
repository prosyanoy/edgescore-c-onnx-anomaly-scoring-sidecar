#include "edgescore/http_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <system_error>

namespace edgescore {
namespace {

constexpr std::size_t kMaxHeaderBytes = 16 * 1024;

const char* reason(int status) {
  switch (status) {
    case 100: return "Continue";
    case 200: return "OK";
    case 202: return "Accepted";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 429: return "Too Many Requests";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout";
    default: return "Unknown";
  }
}

bool send_all(int fd, const char* data, std::size_t len) {
  while (len > 0) {
    ssize_t n = ::send(fd, data, len, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    data += n;
    len -= static_cast<std::size_t>(n);
  }
  return true;
}

bool send_response(int fd, const HttpResponse& r, bool keep_alive) {
  std::string out = "HTTP/1.1 " + std::to_string(r.status) + " " + reason(r.status) + "\r\n";
  out += "Content-Type: " + r.content_type + "\r\n";
  out += "Content-Length: " + std::to_string(r.body.size()) + "\r\n";
  out += keep_alive ? "Connection: keep-alive\r\n" : "Connection: close\r\n";
  for (const auto& h : r.headers) out += h.first + ": " + h.second + "\r\n";
  out += "\r\n";
  out += r.body;
  return send_all(fd, out.data(), out.size());
}

std::string lower(std::string s) {
  for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}
std::string trim(const std::string& s) {
  std::size_t a = s.find_first_not_of(" \t");
  if (a == std::string::npos) return {};
  std::size_t b = s.find_last_not_of(" \t");
  return s.substr(a, b - a + 1);
}

// Buffered reader over a socket.
class Reader {
 public:
  explicit Reader(int fd) : fd_(fd) {}
  // Ensure at least one unread byte is buffered. false on EOF/timeout/error.
  bool fill() {
    if (pos_ < buf_.size()) return true;
    buf_.resize(16384);
    pos_ = 0;
    ssize_t n;
    do { n = ::recv(fd_, buf_.data(), buf_.size(), 0); } while (n < 0 && errno == EINTR);
    if (n <= 0) { buf_.clear(); timed_out = n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK); return false; }
    buf_.resize(static_cast<std::size_t>(n));
    return true;
  }
  bool has_buffered() const { return pos_ < buf_.size(); }
  int get() { return fill() ? static_cast<unsigned char>(buf_[pos_++]) : -1; }
  // Read exactly n bytes appending to out.
  bool read_n(std::string& out, std::size_t n) {
    while (n > 0) {
      if (!fill()) return false;
      std::size_t take = std::min(n, buf_.size() - pos_);
      out.append(buf_.data() + pos_, take);
      pos_ += take;
      n -= take;
    }
    return true;
  }
  // Read a line terminated by LF (CR stripped). false on EOF or if longer than `limit`.
  bool read_line(std::string& line, std::size_t limit) {
    line.clear();
    for (;;) {
      int c = get();
      if (c < 0) return false;
      if (c == '\n') {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return true;
      }
      line += static_cast<char>(c);
      if (line.size() > limit) return false;
    }
  }
  bool timed_out = false;

 private:
  int fd_;
  std::string buf_;
  std::size_t pos_ = 0;
};

enum class ReadStatus { Ok, Closed, BadRequest, TooLarge, HeaderTooLarge, Unsupported, Timeout };

ReadStatus read_request(Reader& rd, int fd, HttpRequest& req, std::size_t max_body, bool& keep_alive) {
  std::string line;
  // Skip stray blank lines between pipelined requests.
  do {
    if (!rd.read_line(line, kMaxHeaderBytes)) return rd.timed_out ? ReadStatus::Timeout : ReadStatus::Closed;
  } while (line.empty());

  auto sp1 = line.find(' ');
  auto sp2 = line.rfind(' ');
  if (sp1 == std::string::npos || sp2 == sp1) return ReadStatus::BadRequest;
  req.method = line.substr(0, sp1);
  std::string target = line.substr(sp1 + 1, sp2 - sp1 - 1);
  std::string version = line.substr(sp2 + 1);
  if (version != "HTTP/1.1" && version != "HTTP/1.0") return ReadStatus::BadRequest;
  if (target.empty() || target[0] != '/') return ReadStatus::BadRequest;
  auto q = target.find('?');
  req.path = target.substr(0, q);
  req.query = q == std::string::npos ? "" : target.substr(q + 1);
  keep_alive = version == "HTTP/1.1";

  std::size_t header_bytes = line.size();
  for (;;) {
    if (!rd.read_line(line, kMaxHeaderBytes)) return rd.timed_out ? ReadStatus::Timeout : ReadStatus::Closed;
    if (line.empty()) break;
    header_bytes += line.size();
    if (header_bytes > kMaxHeaderBytes) return ReadStatus::HeaderTooLarge;
    auto colon = line.find(':');
    if (colon == std::string::npos || colon == 0) return ReadStatus::BadRequest;
    req.headers[lower(line.substr(0, colon))] = trim(line.substr(colon + 1));
  }

  std::string conn = lower(req.header("connection"));
  if (conn == "close") keep_alive = false;
  else if (conn == "keep-alive") keep_alive = true;

  if (!req.header("content-encoding").empty() && lower(req.header("content-encoding")) != "identity")
    return ReadStatus::Unsupported;

  if (lower(req.header("expect")) == "100-continue") {
    static const char kContinue[] = "HTTP/1.1 100 Continue\r\n\r\n";
    if (!send_all(fd, kContinue, sizeof kContinue - 1)) return ReadStatus::Closed;
  }

  std::string te = lower(req.header("transfer-encoding"));
  if (!te.empty()) {
    if (te != "chunked") return ReadStatus::Unsupported;
    for (;;) {
      if (!rd.read_line(line, 128)) return ReadStatus::BadRequest;
      auto semi = line.find(';');
      std::string hex = trim(line.substr(0, semi));
      if (hex.empty() || hex.size() > 8 || hex.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
        return ReadStatus::BadRequest;
      std::size_t size = std::strtoul(hex.c_str(), nullptr, 16);
      if (size == 0) {
        // Trailers until blank line.
        do { if (!rd.read_line(line, kMaxHeaderBytes)) return ReadStatus::BadRequest; } while (!line.empty());
        return ReadStatus::Ok;
      }
      if (req.body.size() + size > max_body) return ReadStatus::TooLarge;
      if (!rd.read_n(req.body, size)) return ReadStatus::BadRequest;
      if (!rd.read_line(line, 8) || !line.empty()) return ReadStatus::BadRequest;
    }
  }

  std::string cl = req.header("content-length");
  if (!cl.empty()) {
    if (cl.find_first_not_of("0123456789") != std::string::npos || cl.size() > 12) return ReadStatus::BadRequest;
    std::size_t n = std::strtoull(cl.c_str(), nullptr, 10);
    if (n > max_body) return ReadStatus::TooLarge;
    req.body.reserve(n);
    if (!rd.read_n(req.body, n)) return ReadStatus::BadRequest;
  }
  return ReadStatus::Ok;
}

}  // namespace

std::string HttpRequest::query_param(const std::string& key) const {
  std::size_t pos = 0;
  while (pos <= query.size()) {
    std::size_t amp = query.find('&', pos);
    std::string pair = query.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
    auto eq = pair.find('=');
    if (pair.substr(0, eq) == key) return eq == std::string::npos ? "" : pair.substr(eq + 1);
    if (amp == std::string::npos) break;
    pos = amp + 1;
  }
  return {};
}

HttpServer::HttpServer(HttpServerOptions opts, Handler handler) : opts_(std::move(opts)), handler_(std::move(handler)) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::start() {
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listen_fd_ < 0) throw std::runtime_error(std::string("socket: ") + std::strerror(errno));
  int one = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(opts_.port);
  if (::inet_pton(AF_INET, opts_.bind_address.c_str(), &addr.sin_addr) != 1) {
    ::close(listen_fd_);
    listen_fd_ = -1;
    throw std::runtime_error("invalid bind address: " + opts_.bind_address);
  }
  if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0 || ::listen(listen_fd_, 512) < 0) {
    std::string err = std::strerror(errno);
    ::close(listen_fd_);
    listen_fd_ = -1;
    throw std::runtime_error("cannot listen on " + opts_.bind_address + ":" + std::to_string(opts_.port) + ": " + err);
  }
  socklen_t len = sizeof addr;
  ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len);
  bound_port_ = ntohs(addr.sin_port);
  acceptor_ = std::thread([this] { accept_loop(); });
}

void HttpServer::stop() {
  if (stopping_.exchange(true)) return;
  if (acceptor_.joinable()) acceptor_.join();  // poll() wakes every 100 ms and sees stopping_
  if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
  std::unique_lock<std::mutex> lk(mu_);
  for (int fd : clients_) ::shutdown(fd, SHUT_RDWR);  // wakes blocked recv(); owners close
  idle_cv_.wait(lk, [&] { return active_ == 0; });
}

void HttpServer::accept_loop() {
  while (!stopping_.load()) {
    pollfd p{listen_fd_, POLLIN, 0};
    int r = ::poll(&p, 1, 100);
    if (r <= 0) continue;
    int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd < 0) continue;
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    timeval tv{opts_.idle_timeout_seconds, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    bool admit;
    {
      std::lock_guard<std::mutex> lk(mu_);
      admit = active_ < opts_.max_connections && !stopping_.load();
      if (admit) { ++active_; clients_.insert(fd); }
    }
    if (!admit) {
      HttpResponse busy;
      busy.status = 503;
      busy.body = "{\"text\":\"Too many connections\",\"code\":9}";
      busy.headers.emplace_back("Retry-After", "1");
      send_response(fd, busy, false);
      ::close(fd);
      continue;
    }
    try {
      std::thread([this, fd] { serve(fd); }).detach();
    } catch (const std::system_error&) {
      std::lock_guard<std::mutex> lk(mu_);
      clients_.erase(fd);
      --active_;
      ::close(fd);
      idle_cv_.notify_all();
    }
  }
}

void HttpServer::serve(int fd) {
  Reader rd(fd);
  for (;;) {
    HttpRequest req;
    bool keep_alive = true;
    ReadStatus st = read_request(rd, fd, req, opts_.max_body_bytes, keep_alive);
    if (st != ReadStatus::Ok) {
      if (st != ReadStatus::Closed) {
        HttpResponse err;
        err.content_type = "application/json";
        switch (st) {
          case ReadStatus::TooLarge: err.status = 413; err.body = "{\"text\":\"Payload too large\",\"code\":6}"; break;
          case ReadStatus::HeaderTooLarge: err.status = 431; err.body = "{\"text\":\"Headers too large\",\"code\":6}"; break;
          case ReadStatus::Unsupported: err.status = 415; err.body = "{\"text\":\"Unsupported content or transfer encoding (send identity)\",\"code\":6}"; break;
          case ReadStatus::Timeout: err.status = 408; err.body = "{\"text\":\"Request timeout\",\"code\":6}"; break;
          default: err.status = 400; err.body = "{\"text\":\"Malformed HTTP request\",\"code\":6}"; break;
        }
        send_response(fd, err, false);
      }
      break;
    }
    HttpResponse resp;
    try {
      resp = handler_(req);
    } catch (const std::exception& e) {
      resp = HttpResponse{};
      resp.status = 500;
      resp.body = "{\"text\":\"Internal error\",\"code\":8}";
    }
    if (stopping_.load()) keep_alive = false;
    if (!send_response(fd, resp, keep_alive) || !keep_alive) break;
  }
  {
    std::lock_guard<std::mutex> lk(mu_);
    clients_.erase(fd);
    ::close(fd);
    --active_;
    idle_cv_.notify_all();  // under the lock: stop() may destroy *this once active_ hits 0
  }
}

}  // namespace edgescore
