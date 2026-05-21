//===- RpcServer.cpp - Run a cosim server ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the cosim RPC server over the WebSocket + JSON protocol.
// The wire protocol is documented in cosim-protocol.md.
//
//===----------------------------------------------------------------------===//

#include "esi/backends/RpcServer.h"
#include "esi/Context.h"
#include "esi/Utils.h"
#include "esi/backends/RpcClient.h" // for ChannelDirection

#include "RpcWire.h"

#include <ixwebsocket/IXBase64.h>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXWebSocketServer.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace esi;
using namespace esi::cosim;
using json = nlohmann::json;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Write the bound port number to a file so callers (typically esi-cosim) can
/// discover it. Necessary when the OS picks the port, since the simulator's
/// stdout/stderr buffering is undefined.
static void writePortFile(uint16_t port) {
  FILE *fd = fopen("cosim.cfg", "w");
  if (!fd)
    return;
  fprintf(fd, "port: %u\n", static_cast<unsigned int>(port));
  fclose(fd);
}

/// Pre-bind a temporary loopback socket to port 0 to discover an OS-assigned
/// ephemeral port. IXWebSocket exposes the port a user passed in but does not
/// query `getsockname` after binding, so for the "let the OS pick" path we have
/// to find a free port ourselves and hand it to IX. SO_REUSEADDR + immediate
/// close keep the race window minimal in practice.
static int pickEphemeralPort() {
#ifdef _WIN32
  SOCKET fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd == INVALID_SOCKET)
    return -1;
#else
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0)
    return -1;
#endif
  int enable = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,
             reinterpret_cast<const char *>(&enable), sizeof(enable));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
    return -1;
  }
  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  int rc = getsockname(fd, reinterpret_cast<sockaddr *>(&bound), &len);
#ifdef _WIN32
  closesocket(fd);
#else
  close(fd);
#endif
  if (rc < 0)
    return -1;
  return ntohs(bound.sin_port);
}

class RpcServerReadPort;
class RpcServerWritePort;

} // namespace

//===----------------------------------------------------------------------===//
// RpcServer::Impl - private implementation
//===----------------------------------------------------------------------===//

class esi::cosim::RpcServer::Impl {
public:
  Impl(Context &ctxt, int port);
  ~Impl();

  Context &getContext() { return ctxt; }

  void setManifest(int esiVersion,
                   const std::vector<uint8_t> &compressedManifest);

  ReadChannelPort &registerReadPort(const std::string &name,
                                    const std::string &type);
  WriteChannelPort &registerWritePort(const std::string &name,
                                      const std::string &type);

  void stop(uint32_t timeoutMS);
  int getPort() const { return boundPort; }

  // Dirty-set doorbell for the transport thread. Public so each port's
  // TSQueue can pass `[&impl, id]{ impl.readyIds.markReady(id); }` as its
  // push-notifier; also called directly from `handleSubscribe` to flush
  // anything queued before the subscription arrived.
  utils::ReadyIdSet<uint16_t> readyIds;

private:
  // Reuse the public direction enum from the client side rather than
  // duplicating it; both ends of the cosim transport agree on the same two
  // values.
  using ChannelDirection = backends::cosim::RpcClient::ChannelDirection;
  struct ChannelInfo {
    uint16_t id;
    std::string name;
    std::string typeId;
    ChannelDirection direction;
    RpcServerReadPort *readPort = nullptr;
    RpcServerWritePort *writePort = nullptr;
  };

  struct ClientSession {
    ix::WebSocket *ws;
    // The set of `to_client` channel ids the client subscribed to.
    std::mutex subscribedMutex;
    std::unordered_set<uint16_t> subscribed;
    // True once `hello` has been answered.
    bool helloDone = false;
  };

  Context &ctxt;

  // Manifest state. setManifest() flips manifestReady and broadcasts the CV;
  // any in-flight `hello` handler blocks on this until it is set.
  std::mutex manifestMutex;
  std::condition_variable manifestReadyCV;
  int esiVersion = -1;
  std::vector<uint8_t> compressedManifest;
  bool manifestReady = false;

  // Channel table is keyed by name in the public API and by id on the wire.
  // Ports are owned here; ChannelInfo holds non-owning pointers.
  std::mutex channelsMutex;
  std::map<std::string, std::unique_ptr<RpcServerReadPort>> readPorts;
  std::map<std::string, std::unique_ptr<RpcServerWritePort>> writePorts;
  std::unordered_map<uint16_t, ChannelInfo> channelById;
  std::unordered_map<std::string, uint16_t> idByName;
  uint16_t nextChannelId = 0;

  // Session state. v3 of the protocol allows a single concurrent client.
  std::mutex sessionMutex;
  std::unique_ptr<ClientSession> session;

  // Transport thread; drains `readyIds` and dispatches per channel id.
  std::thread transportThread;

  // The IXWebSocket server.
  std::unique_ptr<ix::WebSocketServer> server;
  int boundPort = -1;

  // Connection callbacks.
  void onClientMessage(std::shared_ptr<ix::ConnectionState> state,
                       ix::WebSocket &ws, const ix::WebSocketMessagePtr &msg);
  void onOpen(ix::WebSocket &ws);
  void onClose();
  void handleBinaryFrame(const std::string &data);
  void handleControlFrame(ix::WebSocket &ws, const std::string &text);
  void handleHello(ix::WebSocket &ws, uint64_t requestId, const json &params);
  void handleSubscribe(ix::WebSocket &ws, uint64_t requestId,
                       const json &params);
  void handleUnsubscribe(ix::WebSocket &ws, uint64_t requestId,
                         const json &params);
  void sendResult(ix::WebSocket &ws, uint64_t requestId, const json &result);
  void sendError(ix::WebSocket &ws, uint64_t requestId, const std::string &code,
                 const std::string &message);

  void transportLoop();

  // Cross-thread fault propagation out of the IX network thread. See
  // FaultStash docs in RpcWire.h.
  ::esi::cosim::FaultStash faultStash;
};

using Impl = esi::cosim::RpcServer::Impl;

//===----------------------------------------------------------------------===//
// Port implementations
//
// Read and write ports are simple queues; the RPC server pushes/pops as
// appropriate. These mirror the previous gRPC implementation: they are
// transport-agnostic except that write port writes ring the sender doorbell.
//===----------------------------------------------------------------------===//

namespace {
/// Read port for "to server" channels.
///
/// The cosim transport hands inbound frames synchronously from the IX
/// network thread, which is shared across every channel on a connection.
/// Any back-pressure on a per-port consumer would stall that thread and risk
/// a cross-channel deadlock when an accelerator's flow control requires
/// ordering across channels. So we *force* polling-mode `connect` to use an
/// unbounded internal queue regardless of what the caller passes --
/// `ReadChannelPort::pollingState` then acts as our buffer and
/// `invokeCallback` is guaranteed non-blocking. The unbounded queue mirrors
/// the existing `to_client` write queue and is acceptable for a cosim
/// driver.
class RpcServerReadPort : public ReadChannelPort {
public:
  using ReadChannelPort::ReadChannelPort;

  /// Polling-mode connect: force the internal queue to be unbounded
  /// (`bufferSize == 0`) regardless of what the caller asked for.
  void connect(const ConnectOptions &options = {}) override {
    ConnectOptions forced = options;
    forced.bufferSize = 0;
    ReadChannelPort::connect(forced);
  }

  /// Hand one inbound frame to the user callback. Returns `false` only when
  /// the port is disconnected (typically during shutdown), in which case the
  /// caller should drop the frame.
  bool deliver(const MessageData &data) {
    std::unique_ptr<SegmentedMessageData> msg =
        std::make_unique<MessageData>(data);
    return invokeCallback(msg);
  }
};

/// Write queue for "to client" channels. Writes go into the per-port TSQueue;
/// the queue's notifier hook rings the transport doorbell so the transport
/// thread wakes up and drains across the WebSocket. DPI threads must never
/// block on I/O, so this path is strictly non-blocking.
class RpcServerWritePort : public WriteChannelPort {
public:
  RpcServerWritePort(Type *type, Impl &impl, uint16_t channelId)
      : WriteChannelPort(type), channelId(channelId),
        writeQueue([&impl, channelId] { impl.readyIds.markReady(channelId); }) {
  }

  uint16_t getChannelId() const { return channelId; }
  uint16_t channelId;
  utils::TSQueue<MessageData> writeQueue;

protected:
  // TODO: TSQueue is unbounded so if there's no client subscibed it'll fill up
  // memory. We should add some backpressure mechanism here to avoid that.
  void writeImpl(const MessageData &data) override { writeQueue.push(data); }
  bool tryWriteImpl(const MessageData &data) override {
    writeImpl(data);
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Impl - server lifecycle
//===----------------------------------------------------------------------===//

Impl::Impl(Context &ctxt, int port) : ctxt(ctxt) {
  // On Windows, `ix::initNetSystem()` calls `WSAStartup` and returns false if
  // that fails. On other platforms it's a no-op that always returns true.
  if (!ix::initNetSystem())
    throw std::runtime_error(
        "RpcServer: ix::initNetSystem() failed (WSAStartup)");

  // Resolve port 0 / negative request to an OS-assigned ephemeral port,
  // since IXWebSocket does not expose the bound port after the fact.
  int requestedPort = port;
  if (requestedPort <= 0) {
    requestedPort = pickEphemeralPort();
    if (requestedPort <= 0)
      throw std::runtime_error(
          "RpcServer: failed to obtain an ephemeral TCP port");
  }

  const std::string host = "127.0.0.1";
  server = std::make_unique<ix::WebSocketServer>(requestedPort, host);
  server->disablePerMessageDeflate();

  server->setOnClientMessageCallback(
      [this](std::shared_ptr<ix::ConnectionState> state, ix::WebSocket &ws,
             const ix::WebSocketMessagePtr &msg) {
        onClientMessage(std::move(state), ws, msg);
      });

  auto res = server->listen();
  if (!res.first)
    throw std::runtime_error("RpcServer: listen failed: " + res.second);

  server->start();
  boundPort = requestedPort;
  writePortFile(static_cast<uint16_t>(boundPort));
  ctxt.getLogger().info("cosim", std::format("Server listening on {}:{}", host,
                                             static_cast<unsigned>(boundPort)));

  // Start the always-on transport thread that drains every port's queue. It
  // lives the entire lifetime of the server and sleeps on `readyIds`'s
  // internal CV when there is nothing to do regardless of whether a client
  // is connected.
  transportThread = std::thread([this] { transportLoop(); });
}

Impl::~Impl() {
  if (server) {
    // A pending fault from the network thread is interesting, but throwing
    // out of a destructor is worse than swallowing it. The user can still
    // observe the fault by calling stop() explicitly before letting the
    // server destruct; if they didn't, log and continue.
    try {
      stop(0);
    } catch (const std::exception &e) {
      ctxt.getLogger().error(
          "cosim",
          std::string("Suppressed exception during ~RpcServer::Impl: ") +
              e.what());
    } catch (...) {
      ctxt.getLogger().error(
          "cosim", "Suppressed non-std::exception during ~RpcServer::Impl");
    }
  }
}

void Impl::setManifest(int esiVersion,
                       const std::vector<uint8_t> &compressedManifest) {
  faultStash.check();
  {
    std::lock_guard<std::mutex> lock(manifestMutex);
    this->esiVersion = esiVersion;
    this->compressedManifest = compressedManifest;
    manifestReady = true;
  }
  manifestReadyCV.notify_all();
}

ReadChannelPort &Impl::registerReadPort(const std::string &name,
                                        const std::string &type) {
  faultStash.check();
  std::lock_guard<std::mutex> lock(channelsMutex);
  uint16_t id = nextChannelId++;
  auto port = std::make_unique<RpcServerReadPort>(new Type(type));
  RpcServerReadPort *raw = port.get();
  readPorts.emplace(name, std::move(port));

  ChannelInfo info{id, name, type, ChannelDirection::ToServer, raw, nullptr};
  channelById.emplace(id, std::move(info));
  idByName.emplace(name, id);
  return *raw;
}

WriteChannelPort &Impl::registerWritePort(const std::string &name,
                                          const std::string &type) {
  faultStash.check();
  std::lock_guard<std::mutex> lock(channelsMutex);
  uint16_t id = nextChannelId++;
  auto port = std::make_unique<RpcServerWritePort>(new Type(type), *this, id);
  RpcServerWritePort *raw = port.get();
  writePorts.emplace(name, std::move(port));

  ChannelInfo info{id, name, type, ChannelDirection::ToClient, nullptr, raw};
  channelById.emplace(id, std::move(info));
  idByName.emplace(name, id);
  return *raw;
}

void Impl::stop(uint32_t /*timeoutMS*/) {
  // Disconnect ports first so any in-flight DPI writes see a closed channel.
  {
    std::lock_guard<std::mutex> lock(channelsMutex);
    for (auto &[name, port] : readPorts)
      port->disconnect();
    for (auto &[name, port] : writePorts)
      port->disconnect();
  }

  // Retire the transport thread.
  if (transportThread.joinable()) {
    readyIds.requestShutdown();
    transportThread.join();
    // NB: there's no explicit "clear readyIds" step: each port's queue
    // contents and the corresponding dirty markers belong to the ports,
    // not to the transport thread, and the server doesn't restart in-place.
  }

  if (server) {
    server->stop();
    server.reset();
  }

  {
    std::lock_guard<std::mutex> lock(sessionMutex);
    session.reset();
  }

  // Surface any fault the IX network thread caught while we were running.
  // Done last so the rest of the shutdown sequence completes regardless.
  faultStash.check();
}

//===----------------------------------------------------------------------===//
// Impl - WebSocket message dispatch
//===----------------------------------------------------------------------===//

void Impl::onClientMessage(std::shared_ptr<ix::ConnectionState> /*state*/,
                           ix::WebSocket &ws,
                           const ix::WebSocketMessagePtr &msg) {
  switch (msg->type) {
  case ix::WebSocketMessageType::Open: {
    // Single-client model: if a session is already active, send an unsolicited
    // JSON error frame so the new client gets an actionable, application-level
    // reason, then close with 1013 ("Try Again Later") rather than 1011
    // ("Internal Error") -- the latter falsely implies a server-side bug.
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (session) {
      ctxt.getLogger().warning(
          "cosim", "Rejecting additional client; one already connected");
      json err;
      err["type"] = "error";
      err["error"] = {{"code", "server_busy"},
                      {"message", "cosim server allows only one client at a "
                                  "time; another client is already connected"}};
      ws.sendUtf8Text(err.dump());
      // 1013 = Try Again Later (RFC 6455 §7.4).
      ws.close(1013, "cosim server busy: another client is already connected");
      return;
    }
    session = std::make_unique<ClientSession>();
    session->ws = &ws;
    ctxt.getLogger().debug("cosim", "Client connected");
    return;
  }
  case ix::WebSocketMessageType::Close:
  case ix::WebSocketMessageType::Error: {
    ctxt.getLogger().debug(
        "cosim", std::format("Client disconnected: {}",
                             msg->type == ix::WebSocketMessageType::Error
                                 ? msg->errorInfo.reason
                                 : msg->closeInfo.reason));
    onClose();
    return;
  }
  case ix::WebSocketMessageType::Message:
    // An exception escaping this callback would kill IX's network thread.
    // Stash the first one so the next public RpcServer method rethrows it
    // on the user's thread.
    try {
      if (msg->binary)
        handleBinaryFrame(msg->str);
      else
        handleControlFrame(ws, msg->str);
    } catch (...) {
      faultStash.record(std::current_exception());
    }
    return;
  default:
    return;
  }
}

void Impl::onClose() {
  std::lock_guard<std::mutex> lock(sessionMutex);
  session.reset();
}

void Impl::handleBinaryFrame(const std::string &data) {
  if (data.size() < 2) {
    ctxt.getLogger().error("cosim",
                           "Received binary frame shorter than 2-byte header");
    return;
  }
  uint16_t channelId =
      static_cast<uint8_t>(data[0]) |
      (static_cast<uint16_t>(static_cast<uint8_t>(data[1])) << 8);

  RpcServerReadPort *port = nullptr;
  {
    std::lock_guard<std::mutex> lock(channelsMutex);
    auto it = channelById.find(channelId);
    if (it == channelById.end() ||
        it->second.direction != ChannelDirection::ToServer)
      port = nullptr;
    else
      port = it->second.readPort;
  }
  if (!port) {
    ctxt.getLogger().error(
        "cosim", std::format("Binary frame for unknown to-server channel id {}",
                             static_cast<unsigned>(channelId)));
    return;
  }

  MessageData payload(reinterpret_cast<const uint8_t *>(data.data() + 2),
                      data.size() - 2);
  // RpcServerReadPort always has an unbounded internal queue so `deliver` is
  // non-blocking: it enqueues into `ReadChannelPort`'s internal polling buffer
  // and returns. A `false` return means the port has been disconnected
  // (shutdown); just drop the frame in that case.
  if (!port->deliver(payload))
    ctxt.getLogger().debug(
        "cosim",
        std::format("Dropped {} bytes for channel id {}: port not connected",
                    payload.getSize(), static_cast<unsigned>(channelId)));
}

void Impl::handleControlFrame(ix::WebSocket &ws, const std::string &text) {
  json req;
  try {
    req = json::parse(text);
  } catch (const std::exception &e) {
    ctxt.getLogger().error(
        "cosim", std::format("Failed to parse control frame: {}", e.what()));
    sendError(ws, 0, "protocol_error",
              std::string("Failed to parse JSON: ") + e.what());
    faultStash.record(std::current_exception());
    return;
  }

  auto typeIt = req.find("type");
  if (typeIt == req.end() || !typeIt->is_string() ||
      typeIt->get<std::string>() != "request") {
    sendError(ws, 0, "protocol_error",
              "Control frame missing \"type\":\"request\"");
    return;
  }
  uint64_t requestId = 0;
  if (auto idIt = req.find("request_id"); idIt != req.end()) {
    // Accept only unsigned integers.
    if (!idIt->is_number_unsigned()) {
      sendError(ws, 0, "protocol_error",
                "\"request_id\" must be an unsigned integer");
      return;
    }
    try {
      requestId = idIt->get<uint64_t>();
    } catch (const std::exception &e) {
      sendError(ws, 0, "protocol_error",
                std::string("Invalid \"request_id\": ") + e.what());
      return;
    }
  }
  auto methodIt = req.find("method");
  if (methodIt == req.end() || !methodIt->is_string()) {
    sendError(ws, requestId, "protocol_error", "Missing \"method\"");
    return;
  }
  std::string method = methodIt->get<std::string>();
  json params = req.value("params", json::object());

  if (method == "hello")
    handleHello(ws, requestId, params);
  else if (method == "subscribe")
    handleSubscribe(ws, requestId, params);
  else if (method == "unsubscribe")
    handleUnsubscribe(ws, requestId, params);
  else
    sendError(ws, requestId, "protocol_error", "Unknown method: " + method);
}

//===----------------------------------------------------------------------===//
// Impl - control methods
//===----------------------------------------------------------------------===//

void Impl::handleHello(ix::WebSocket &ws, uint64_t requestId,
                       const json & /*params*/) {
  // Block until the manifest has been set. This replaces the gRPC-era poll
  // loop on the client side.
  {
    std::unique_lock<std::mutex> lock(manifestMutex);
    manifestReadyCV.wait(lock, [&] { return manifestReady; });
  }

  json result;
  result["protocol_version"] = 3;
  {
    std::lock_guard<std::mutex> lock(manifestMutex);
    result["esi_version"] = esiVersion;
    result["compressed_manifest_b64"] = macaron::Base64::Encode(
        std::string(reinterpret_cast<const char *>(compressedManifest.data()),
                    compressedManifest.size()));
  }

  json channelsJson = json::array();
  {
    std::lock_guard<std::mutex> lock(channelsMutex);
    // Emit in id order for deterministic output.
    for (uint16_t i = 0; i < nextChannelId; ++i) {
      auto it = channelById.find(i);
      if (it == channelById.end())
        continue;
      const ChannelInfo &info = it->second;
      json c;
      c["channel_id"] = info.id;
      c["name"] = info.name;
      c["type"] = info.typeId;
      c["direction"] = info.direction == ChannelDirection::ToServer
                           ? "to_server"
                           : "to_client";
      channelsJson.push_back(std::move(c));
    }
  }
  result["channels"] = std::move(channelsJson);

  {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (session)
      session->helloDone = true;
  }

  sendResult(ws, requestId, result);
}

void Impl::handleSubscribe(ix::WebSocket &ws, uint64_t requestId,
                           const json &params) {
  auto chIdIt = params.find("channel_id");
  if (chIdIt == params.end() || !chIdIt->is_number_unsigned()) {
    sendError(ws, requestId, "protocol_error",
              "subscribe requires unsigned \"channel_id\"");
    return;
  }
  uint64_t rawId = chIdIt->get<uint64_t>();
  if (rawId > std::numeric_limits<uint16_t>::max()) {
    sendError(ws, requestId, "protocol_error",
              std::format("channel_id {} exceeds uint16_t range", rawId));
    return;
  }
  uint16_t channelId = static_cast<uint16_t>(rawId);

  {
    std::lock_guard<std::mutex> chLock(channelsMutex);
    auto it = channelById.find(channelId);
    if (it == channelById.end()) {
      sendError(ws, requestId, "unknown_channel",
                std::format("No channel with id {}",
                            static_cast<unsigned>(channelId)));
      return;
    }
    if (it->second.direction != ChannelDirection::ToClient) {
      sendError(ws, requestId, "wrong_direction",
                std::format("Channel id {} is not a to-client channel",
                            static_cast<unsigned>(channelId)));
      return;
    }

    std::lock_guard<std::mutex> sLock(sessionMutex);
    if (!session) {
      sendError(ws, requestId, "internal", "No active session");
      return;
    }
    std::lock_guard<std::mutex> subLock(session->subscribedMutex);
    session->subscribed.insert(channelId);
  }

  // Send the subscribe-ack BEFORE waking the transport thread. IXWebSocket
  // queues sends in FIFO order on a given connection, so as long as the ack
  // is enqueued first, any data frames the transport thread emits next will
  // arrive after it. This spares clients from having to tolerate data on a
  // channel before they've seen the ack confirming the subscription.
  sendResult(ws, requestId, json::object());

  // Now kick the transport thread: if the port already has queued data
  // (typical for accelerator-startup writes that landed before the client
  // subscribed), this is what flushes it; if the queue is empty, the
  // transport thread will just see nothing to drain and go back to sleep.
  // The dirty-set semantics of `readyIds` dedupe any concurrent doorbell
  // from a DPI write.
  readyIds.markReady(channelId);
}

void Impl::handleUnsubscribe(ix::WebSocket &ws, uint64_t requestId,
                             const json &params) {
  auto chIdIt = params.find("channel_id");
  if (chIdIt == params.end() || !chIdIt->is_number_unsigned()) {
    sendError(ws, requestId, "protocol_error",
              "unsubscribe requires unsigned \"channel_id\"");
    return;
  }
  uint64_t rawId = chIdIt->get<uint64_t>();
  if (rawId > std::numeric_limits<uint16_t>::max()) {
    sendError(ws, requestId, "protocol_error",
              std::format("channel_id {} exceeds uint16_t range", rawId));
    return;
  }
  uint16_t channelId = static_cast<uint16_t>(rawId);

  std::lock_guard<std::mutex> sLock(sessionMutex);
  if (!session) {
    sendError(ws, requestId, "internal", "No active session");
    return;
  }
  std::lock_guard<std::mutex> subLock(session->subscribedMutex);
  auto removed = session->subscribed.erase(channelId);
  if (!removed) {
    sendError(ws, requestId, "not_subscribed",
              std::format("Channel id {} is not subscribed",
                          static_cast<unsigned>(channelId)));
    return;
  }
  sendResult(ws, requestId, json::object());
}

void Impl::sendResult(ix::WebSocket &ws, uint64_t requestId,
                      const json &result) {
  json resp;
  resp["type"] = "response";
  resp["request_id"] = requestId;
  resp["result"] = result;
  // IXWebSocket serializes concurrent send calls internally; we don't need
  // sessionMutex here, and taking it would deadlock with handlers that hold
  // sessionMutex while replying (e.g. handleUnsubscribe).
  ws.sendUtf8Text(resp.dump());
}

void Impl::sendError(ix::WebSocket &ws, uint64_t requestId,
                     const std::string &code, const std::string &message) {
  json resp;
  resp["type"] = "response";
  resp["request_id"] = requestId;
  resp["error"] = {{"code", code}, {"message", message}};
  ws.sendUtf8Text(resp.dump());
}

//===----------------------------------------------------------------------===//
// Impl - transport thread (push model, `to_client` only)
//===----------------------------------------------------------------------===//

void Impl::transportLoop() {
  while (true) {
    std::unordered_set<uint16_t> ids;
    if (!readyIds.waitDrain(ids))
      return;
    for (uint16_t id : ids) {
      RpcServerWritePort *writePort = nullptr;
      {
        std::lock_guard<std::mutex> chLock(channelsMutex);
        auto it = channelById.find(id);
        if (it == channelById.end() ||
            it->second.direction != ChannelDirection::ToClient)
          continue;
        writePort = it->second.writePort;
      }
      if (!writePort)
        continue;

      // Only drain if a client is subscribed; otherwise data stays in the
      // queue until subscription (handleSubscribe fires markReady). We
      // re-acquire `sessionMutex` + `subscribedMutex` on every iteration so
      // `handleUnsubscribe` can interpose between frames: once it erases the
      // subscription and sends the unsubscribe-ack, the next iteration here
      // sees the channel as not subscribed and breaks, guaranteeing the ack
      // is the last `to_client` frame on that channel (as the protocol spec
      // requires). Holding the locks across the actual `sendBinary` also
      // keeps `session->ws` valid against a concurrent `onClose()`.
      while (true) {
        if (readyIds.isShutdown())
          return;
        std::lock_guard<std::mutex> sLock(sessionMutex);
        if (!session)
          break;
        std::lock_guard<std::mutex> subLock(session->subscribedMutex);
        if (!session->subscribed.count(id))
          break;
        std::optional<MessageData> msg = writePort->writeQueue.pop();
        if (!msg)
          break;
        std::string frame = buildDataFrame(id, msg->getBytes(), msg->getSize());
        // IXWebSocket serializes concurrent send calls internally and queues
        // them in FIFO order on the WS; no extra mutex is needed for that,
        // but we keep `sessionMutex` to guard the `ws` pointer's lifetime.
        session->ws->sendBinary(frame);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// RpcServer pass-throughs
//===----------------------------------------------------------------------===//

RpcServer::RpcServer(Context &ctxt) : ctxt(ctxt) {}
RpcServer::~RpcServer() = default;

void RpcServer::setManifest(int esiVersion,
                            const std::vector<uint8_t> &compressedManifest) {
  if (!impl)
    throw std::runtime_error("Server not running");
  impl->setManifest(esiVersion, compressedManifest);
}

ReadChannelPort &RpcServer::registerReadPort(const std::string &name,
                                             const std::string &type) {
  if (!impl)
    throw std::runtime_error("Server not running");
  return impl->registerReadPort(name, type);
}

WriteChannelPort &RpcServer::registerWritePort(const std::string &name,
                                               const std::string &type) {
  if (!impl)
    throw std::runtime_error("Server not running");
  return impl->registerWritePort(name, type);
}

void RpcServer::run(int port) {
  if (impl)
    throw std::runtime_error("Server already running");
  impl = std::make_unique<Impl>(ctxt, port);
}

void RpcServer::stop(uint32_t timeoutMS) {
  if (!impl)
    throw std::runtime_error("Server not running");
  impl->stop(timeoutMS);
}

int RpcServer::getPort() {
  if (!impl)
    throw std::runtime_error("Server not running");
  return impl->getPort();
}
