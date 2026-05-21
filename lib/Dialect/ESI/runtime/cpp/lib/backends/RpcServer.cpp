//===- RpcServer.cpp - Run a cosim server ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the cosim RPC server over the WebSocket + JSON protocol,
// backed by libwebsockets. The wire protocol is documented in
// cosim-protocol.md.
//
// All WebSocket I/O happens on the single LWS service thread inside
// `serviceLoop()`. Other threads (DPI write paths, the public `setManifest`
// caller, etc.) interact with the server by enqueuing onto thread-safe
// structures and calling `lws_cancel_service()`, which wakes the service
// thread which then services `LWS_CALLBACK_EVENT_WAIT_CANCELLED`.
//
//===----------------------------------------------------------------------===//

#include "esi/backends/RpcServer.h"
#include "esi/Context.h"
#include "esi/Utils.h"
#include "esi/backends/RpcClient.h" // for ChannelDirection

#include "Base64.h"
#include "RpcWire.h"

#include <libwebsockets.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <format>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace esi;
using namespace esi::cosim;
using json = nlohmann::json;

namespace {

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

class RpcServerReadPort;
class RpcServerWritePort;

/// Per-WebSocket-session state. LWS allocates `per_session_data_size` bytes
/// of raw memory per connection and hands a pointer to it as the `user`
/// parameter of every callback for that connection. We placement-new this
/// type into that buffer in `LWS_CALLBACK_ESTABLISHED` and destruct it in
/// `LWS_CALLBACK_CLOSED`. All access is from the LWS service thread, so no
/// locking is required.
struct CosimSessionPss {
  /// FIFO of outbound frames awaiting `lws_write`. Always drained from
  /// `LWS_CALLBACK_SERVER_WRITEABLE`.
  std::deque<WireFrame> outbound;
  /// Accumulator for fragmented inbound WebSocket frames. Cleared on each
  /// final fragment.
  std::string rxBuffer;
  /// Channels (to-client) this session is subscribed to.
  std::unordered_set<uint16_t> subscribed;
  /// True after we have answered the `hello` request.
  bool helloDone = false;
  /// `hello` arrived before `setManifest()` was called; the response is
  /// deferred until the manifest is ready. `helloRequestId` holds the
  /// request id to use for the deferred reply.
  bool helloPending = false;
  uint64_t helloRequestId = 0;
  /// Singleton-rejection path: this connection should be closed after its
  /// outbound queue drains.
  bool closeAfterWrite = false;
};

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

  /// Dirty-set doorbell for cross-thread writes. WritePort TSQueues notify
  /// this set whenever a DPI thread enqueues data, then we wake the LWS
  /// service thread.
  utils::ReadyIdSet<uint16_t> readyIds;

  /// Called by WritePort queue notifiers (from arbitrary threads).
  void ringDoorbell(uint16_t channelId) {
    readyIds.markReady(channelId);
    if (context)
      lws_cancel_service(context);
  }

  /// LWS protocol callback dispatch (service thread only).
  int callback(struct lws *wsi, enum lws_callback_reasons reason, void *user,
               void *in, size_t len);

private:
  using ChannelDirection = backends::cosim::RpcClient::ChannelDirection;
  struct ChannelInfo {
    uint16_t id;
    std::string name;
    std::string typeId;
    ChannelDirection direction;
    RpcServerReadPort *readPort = nullptr;
    RpcServerWritePort *writePort = nullptr;
  };

  Context &ctxt;

  // Manifest state. `setManifest()` flips `manifestReady` then wakes the LWS
  // service thread so any deferred `hello` reply can fire.
  std::mutex manifestMutex;
  int esiVersion = -1;
  std::vector<uint8_t> compressedManifest;
  std::atomic<bool> manifestReady{false};

  // Channel table is keyed by name in the public API and by id on the wire.
  std::mutex channelsMutex;
  std::map<std::string, std::unique_ptr<RpcServerReadPort>> readPorts;
  std::map<std::string, std::unique_ptr<RpcServerWritePort>> writePorts;
  std::unordered_map<uint16_t, ChannelInfo> channelById;
  std::unordered_map<std::string, uint16_t> idByName;
  uint16_t nextChannelId = 0;

  // Singleton session. v3 of the protocol allows a single concurrent client;
  // `clientWsi` is the live connection (only accessed from the service
  // thread or with `sessionMutex` held when the service thread is paused).
  std::mutex sessionMutex;
  struct lws *clientWsi = nullptr;

  // LWS plumbing.
  struct lws_context *context = nullptr;
  struct lws_vhost *vhost = nullptr;
  std::thread serviceThread;
  std::atomic<bool> shouldStop{false};
  int boundPort = -1;

  // Service-thread internals.
  void serviceLoop();
  void onEventCanceled();
  void drainWritePort(uint16_t channelId);

  // Per-message handlers (all run on service thread).
  void handleBinaryFrame(const std::string &data);
  void handleControlFrame(struct lws *wsi, CosimSessionPss *pss,
                          const std::string &text);
  void handleHello(struct lws *wsi, CosimSessionPss *pss, uint64_t requestId,
                   const json &params);
  void handleSubscribe(struct lws *wsi, CosimSessionPss *pss,
                       uint64_t requestId, const json &params);
  void handleUnsubscribe(struct lws *wsi, CosimSessionPss *pss,
                         uint64_t requestId, const json &params);

  // Frame enqueue helpers (service thread only). They push onto `pss`'s
  // outbound queue and schedule a writable callback.
  void enqueueResult(struct lws *wsi, CosimSessionPss *pss, uint64_t requestId,
                     const json &result);
  void enqueueError(struct lws *wsi, CosimSessionPss *pss, uint64_t requestId,
                    const std::string &code, const std::string &message);
  void enqueueFrame(struct lws *wsi, CosimSessionPss *pss, WireFrame frame);

  // Build the full `hello` reply JSON from current manifest + channel table.
  json buildHelloResult();
};

using Impl = esi::cosim::RpcServer::Impl;

//===----------------------------------------------------------------------===//
// Port implementations
//===----------------------------------------------------------------------===//

namespace {
/// Read port for "to server" channels.
///
/// Inbound binary frames are dispatched synchronously from the LWS service
/// thread. Any back-pressure on a per-port consumer would stall every
/// channel, so we force polling-mode `connect` to use an unbounded internal
/// queue (`bufferSize == 0`) regardless of what the caller passes.
class RpcServerReadPort : public ReadChannelPort {
public:
  using ReadChannelPort::ReadChannelPort;

  void connect(const ConnectOptions &options = {}) override {
    ConnectOptions forced = options;
    forced.bufferSize = 0;
    ReadChannelPort::connect(forced);
  }

  /// Hand one inbound frame to the user callback. Returns `false` if the
  /// port has been disconnected; the caller should then drop the frame.
  bool deliver(const MessageData &data) {
    std::unique_ptr<SegmentedMessageData> msg =
        std::make_unique<MessageData>(data);
    return invokeCallback(msg);
  }
};

/// Write queue for "to client" channels. Writes go into a per-port TSQueue;
/// the queue's notifier rings the server doorbell, which wakes the LWS
/// service thread. DPI threads must never block on I/O, so this path is
/// strictly non-blocking.
class RpcServerWritePort : public WriteChannelPort {
public:
  RpcServerWritePort(Type *type, Impl &impl, uint16_t channelId)
      : WriteChannelPort(type), channelId(channelId),
        writeQueue([&impl, channelId] { impl.ringDoorbell(channelId); }) {}

  uint16_t getChannelId() const { return channelId; }
  uint16_t channelId;
  utils::TSQueue<MessageData> writeQueue;

protected:
  // TODO: TSQueue is unbounded so if there's no client subscribed it'll fill
  // up memory. We should add some backpressure mechanism here to avoid that.
  void writeImpl(const MessageData &data) override { writeQueue.push(data); }
  bool tryWriteImpl(const MessageData &data) override {
    writeImpl(data);
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// LWS protocol table
//===----------------------------------------------------------------------===//

namespace {
static int cosimCallback(struct lws *wsi, enum lws_callback_reasons reason,
                         void *user, void *in, size_t len) {
  struct lws_context *ctx = lws_get_context(wsi);
  if (!ctx)
    return 0;
  void *impl = lws_context_user(ctx);
  if (!impl)
    return 0;
  return static_cast<Impl *>(impl)->callback(wsi, reason, user, in, len);
}

// One protocol entry; no per-protocol user-data of our own (we route through
// `lws_context_user`).
static const struct lws_protocols kProtocols[] = {
    {"esi-cosim-v3", cosimCallback, sizeof(CosimSessionPss),
     /*rx_buffer_size=*/0,
     /*id=*/0, /*user=*/nullptr,
     /*tx_packet_size=*/0},
    LWS_PROTOCOL_LIST_TERM};
} // namespace

//===----------------------------------------------------------------------===//
// Impl - server lifecycle
//===----------------------------------------------------------------------===//

Impl::Impl(Context &ctxt, int port) : ctxt(ctxt) {
  // Quiet libwebsockets' own log spam; keep warnings and errors.
  lws_set_log_level(LLL_ERR | LLL_WARN, nullptr);

  // We use explicit vhosts so we can capture the vhost pointer and query the
  // OS-assigned listen port via `lws_get_vhost_listen_port`. With implicit
  // vhost creation we'd have no handle to the default vhost.
  struct lws_context_creation_info ctxInfo = {};
  ctxInfo.options =
      LWS_SERVER_OPTION_EXPLICIT_VHOSTS | LWS_SERVER_OPTION_DISABLE_IPV6;
  ctxInfo.user = this;
  ctxInfo.gid = -1;
  ctxInfo.uid = -1;

  context = lws_create_context(&ctxInfo);
  if (!context)
    throw std::runtime_error("RpcServer: lws_create_context failed");

  struct lws_context_creation_info vhInfo = {};
  // Port 0 (or negative) tells the OS to pick an ephemeral port; LWS exposes
  // the chosen value via `lws_get_vhost_listen_port`.
  vhInfo.port = port < 0 ? 0 : port;
  vhInfo.iface = "127.0.0.1";
  vhInfo.protocols = kProtocols;
  vhInfo.options = LWS_SERVER_OPTION_DISABLE_IPV6;

  vhost = lws_create_vhost(context, &vhInfo);
  if (!vhost) {
    lws_context_destroy(context);
    context = nullptr;
    throw std::runtime_error("RpcServer: lws_create_vhost failed");
  }

  int actual = lws_get_vhost_listen_port(vhost);
  if (actual <= 0) {
    lws_context_destroy(context);
    context = nullptr;
    throw std::runtime_error(
        "RpcServer: lws_get_vhost_listen_port returned no port");
  }
  boundPort = actual;
  writePortFile(static_cast<uint16_t>(boundPort));
  ctxt.getLogger().info(
      "cosim", std::format("Server listening on 127.0.0.1:{}",
                           static_cast<unsigned>(boundPort)));

  serviceThread = std::thread([this] { serviceLoop(); });
}

Impl::~Impl() {
  if (context)
    stop(0);
}

void Impl::serviceLoop() {
  while (!shouldStop.load(std::memory_order_acquire)) {
    // Timeout argument is ignored in modern LWS (the event loop blocks until
    // I/O or `lws_cancel_service` arrives), but pass 0 to be safe.
    int n = lws_service(context, 0);
    if (n < 0)
      break;
  }
}

void Impl::setManifest(int esiVersion,
                       const std::vector<uint8_t> &compressedManifest) {
  {
    std::lock_guard<std::mutex> lock(manifestMutex);
    this->esiVersion = esiVersion;
    this->compressedManifest = compressedManifest;
  }
  manifestReady.store(true, std::memory_order_release);
  // Wake the service thread so any deferred `hello` reply fires.
  if (context)
    lws_cancel_service(context);
}

ReadChannelPort &Impl::registerReadPort(const std::string &name,
                                        const std::string &type) {
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

  // Retire the doorbell to wake any consumer/producer that might still be
  // blocked waiting on it.
  readyIds.requestShutdown();

  // Stop the LWS service thread. Setting `shouldStop` first then cancelling
  // service forces the next `lws_service` call to return so the loop exits.
  if (context) {
    shouldStop.store(true, std::memory_order_release);
    lws_cancel_service(context);
    if (serviceThread.joinable())
      serviceThread.join();
    lws_context_destroy(context);
    context = nullptr;
    vhost = nullptr;
  }

  {
    std::lock_guard<std::mutex> lock(sessionMutex);
    clientWsi = nullptr;
  }
}

//===----------------------------------------------------------------------===//
// Impl - LWS callback dispatch (service thread only)
//===----------------------------------------------------------------------===//

int Impl::callback(struct lws *wsi, enum lws_callback_reasons reason,
                   void *user, void *in, size_t len) {
  CosimSessionPss *pss = static_cast<CosimSessionPss *>(user);

  switch (reason) {
  case LWS_CALLBACK_ESTABLISHED: {
    // `pss` points at raw memory LWS allocated for us; placement-new the C++
    // session state into it.
    new (pss) CosimSessionPss();
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (clientWsi) {
      // Singleton model: reject the new client with an application-level
      // error frame, then close with 1013 ("Try Again Later") rather than
      // 1011 ("Internal Error"), which would falsely imply a server bug.
      ctxt.getLogger().warning(
          "cosim", "Rejecting additional client; one already connected");
      json err;
      err["type"] = "error";
      err["error"] = {{"code", "server_busy"},
                      {"message", "cosim server allows only one client at a "
                                  "time; another client is already connected"}};
      pss->outbound.push_back(buildLwsTextFrame(err.dump()));
      pss->closeAfterWrite = true;
      lws_callback_on_writable(wsi);
      return 0;
    }
    clientWsi = wsi;
    ctxt.getLogger().debug("cosim", "Client connected");
    return 0;
  }

  case LWS_CALLBACK_CLOSED: {
    {
      std::lock_guard<std::mutex> lock(sessionMutex);
      if (clientWsi == wsi)
        clientWsi = nullptr;
    }
    ctxt.getLogger().debug("cosim", "Client disconnected");
    if (pss)
      pss->~CosimSessionPss();
    return 0;
  }

  case LWS_CALLBACK_RECEIVE: {
    if (in && len)
      pss->rxBuffer.append(static_cast<const char *>(in), len);
    // Reassemble fragmented messages: collect until the final fragment and
    // no more bytes remain in the current packet.
    if (lws_is_final_fragment(wsi) && lws_remaining_packet_payload(wsi) == 0) {
      bool binary = lws_frame_is_binary(wsi);
      std::string data;
      data.swap(pss->rxBuffer);
      if (binary)
        handleBinaryFrame(data);
      else
        handleControlFrame(wsi, pss, data);
    }
    return 0;
  }

  case LWS_CALLBACK_SERVER_WRITEABLE: {
    if (pss->outbound.empty()) {
      if (pss->closeAfterWrite) {
        const char *reason = "cosim server busy";
        // 1013 = "Try Again Later" (RFC 6455 §7.4); libwebsockets doesn't
        // expose a constant for it in 4.3.x, so cast directly.
        lws_close_reason(wsi, static_cast<enum lws_close_status>(1013),
                         reinterpret_cast<unsigned char *>(
                             const_cast<char *>(reason)),
                         std::strlen(reason));
        return -1;
      }
      return 0;
    }

    WireFrame f = std::move(pss->outbound.front());
    pss->outbound.pop_front();
    lws_write_protocol proto = f.isBinary ? LWS_WRITE_BINARY : LWS_WRITE_TEXT;
    int m = lws_write(wsi, f.writePtr(), f.payloadSize, proto);
    if (m < static_cast<int>(f.payloadSize)) {
      ctxt.getLogger().error("cosim", "lws_write returned short");
      return -1;
    }

    if (!pss->outbound.empty() || pss->closeAfterWrite)
      lws_callback_on_writable(wsi);
    return 0;
  }

  case LWS_CALLBACK_EVENT_WAIT_CANCELLED: {
    if (shouldStop.load(std::memory_order_acquire))
      return 0;
    onEventCanceled();
    return 0;
  }

  default:
    return 0;
  }
}

void Impl::onEventCanceled() {
  // If a `hello` was waiting on the manifest and the manifest is now ready,
  // dispatch the deferred reply. We only need to act on the single live
  // session.
  struct lws *wsi = nullptr;
  CosimSessionPss *pss = nullptr;
  {
    std::lock_guard<std::mutex> lock(sessionMutex);
    wsi = clientWsi;
  }
  if (wsi) {
    pss = static_cast<CosimSessionPss *>(lws_wsi_user(wsi));
    if (pss && pss->helloPending && manifestReady.load(std::memory_order_acquire)) {
      uint64_t reqId = pss->helloRequestId;
      pss->helloPending = false;
      pss->helloDone = true;
      enqueueResult(wsi, pss, reqId, buildHelloResult());
    }
  }

  // Drain dirty channel ids using a zero-timeout `waitDrain`. With a 0ms
  // backoff this returns immediately even when the set is empty, which is
  // what we want from a single-shot service-thread callback.
  std::unordered_set<uint16_t> ids;
  readyIds.waitDrain(ids, std::chrono::milliseconds(0));
  for (uint16_t id : ids)
    drainWritePort(id);
}

void Impl::drainWritePort(uint16_t channelId) {
  // Find the port. Lock channelsMutex only while looking it up; the port
  // pointer itself is stable for the lifetime of the server.
  RpcServerWritePort *writePort = nullptr;
  {
    std::lock_guard<std::mutex> lock(channelsMutex);
    auto it = channelById.find(channelId);
    if (it == channelById.end() ||
        it->second.direction != ChannelDirection::ToClient)
      return;
    writePort = it->second.writePort;
  }
  if (!writePort)
    return;

  // We're on the service thread; the singleton session and its pss are
  // therefore stable for the duration of this call (no concurrent
  // ESTABLISHED/CLOSED can interleave).
  struct lws *wsi = nullptr;
  {
    std::lock_guard<std::mutex> lock(sessionMutex);
    wsi = clientWsi;
  }
  if (!wsi)
    return;
  CosimSessionPss *pss = static_cast<CosimSessionPss *>(lws_wsi_user(wsi));
  if (!pss)
    return;
  // Only push data for subscribed channels; everything else stays queued in
  // the port until the client subscribes (and `handleSubscribe` will then
  // mark the id ready again).
  if (!pss->subscribed.count(channelId))
    return;

  // Drain everything pending for this channel into outbound. We're on the
  // service thread so no other thread will write to `pss->outbound`.
  bool any = false;
  while (auto msg = writePort->writeQueue.pop()) {
    WireFrame f =
        buildLwsBinaryFrame(channelId, msg->getBytes(), msg->getSize());
    pss->outbound.push_back(std::move(f));
    any = true;
  }
  if (any)
    lws_callback_on_writable(wsi);
}

//===----------------------------------------------------------------------===//
// Impl - WebSocket message dispatch
//===----------------------------------------------------------------------===//

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
  if (!port->deliver(payload))
    ctxt.getLogger().debug(
        "cosim",
        std::format("Dropped {} bytes for channel id {}: port not connected",
                    payload.getSize(), static_cast<unsigned>(channelId)));
}

void Impl::handleControlFrame(struct lws *wsi, CosimSessionPss *pss,
                              const std::string &text) {
  json req;
  try {
    req = json::parse(text);
  } catch (const std::exception &e) {
    ctxt.getLogger().error(
        "cosim", std::format("Failed to parse control frame: {}", e.what()));
    enqueueError(wsi, pss, 0, "protocol_error",
                 std::string("Failed to parse JSON: ") + e.what());
    return;
  }

  auto typeIt = req.find("type");
  if (typeIt == req.end() || !typeIt->is_string() ||
      typeIt->get<std::string>() != "request") {
    enqueueError(wsi, pss, 0, "protocol_error",
                 "Control frame missing \"type\":\"request\"");
    return;
  }
  uint64_t requestId = 0;
  if (auto idIt = req.find("request_id");
      idIt != req.end() && idIt->is_number())
    requestId = idIt->get<uint64_t>();
  auto methodIt = req.find("method");
  if (methodIt == req.end() || !methodIt->is_string()) {
    enqueueError(wsi, pss, requestId, "protocol_error", "Missing \"method\"");
    return;
  }
  std::string method = methodIt->get<std::string>();
  json params = req.value("params", json::object());

  if (method == "hello")
    handleHello(wsi, pss, requestId, params);
  else if (method == "subscribe")
    handleSubscribe(wsi, pss, requestId, params);
  else if (method == "unsubscribe")
    handleUnsubscribe(wsi, pss, requestId, params);
  else
    enqueueError(wsi, pss, requestId, "protocol_error",
                 "Unknown method: " + method);
}

//===----------------------------------------------------------------------===//
// Impl - control methods
//===----------------------------------------------------------------------===//

json Impl::buildHelloResult() {
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
  return result;
}

void Impl::handleHello(struct lws *wsi, CosimSessionPss *pss,
                       uint64_t requestId, const json & /*params*/) {
  // If the manifest is not yet ready, stash the request id and defer the
  // reply until `setManifest()` wakes us via `lws_cancel_service`. The IX
  // implementation blocked the network thread on a CV here; we can't do that
  // on the LWS service thread without stalling every other connection.
  if (!manifestReady.load(std::memory_order_acquire)) {
    pss->helloPending = true;
    pss->helloRequestId = requestId;
    return;
  }
  pss->helloDone = true;
  enqueueResult(wsi, pss, requestId, buildHelloResult());
}

void Impl::handleSubscribe(struct lws *wsi, CosimSessionPss *pss,
                           uint64_t requestId, const json &params) {
  auto chIdIt = params.find("channel_id");
  if (chIdIt == params.end() || !chIdIt->is_number_unsigned()) {
    enqueueError(wsi, pss, requestId, "protocol_error",
                 "subscribe requires unsigned \"channel_id\"");
    return;
  }
  uint16_t channelId = chIdIt->get<uint16_t>();

  {
    std::lock_guard<std::mutex> chLock(channelsMutex);
    auto it = channelById.find(channelId);
    if (it == channelById.end()) {
      enqueueError(wsi, pss, requestId, "unknown_channel",
                   std::format("No channel with id {}",
                               static_cast<unsigned>(channelId)));
      return;
    }
    if (it->second.direction != ChannelDirection::ToClient) {
      enqueueError(wsi, pss, requestId, "wrong_direction",
                   std::format("Channel id {} is not a to-client channel",
                               static_cast<unsigned>(channelId)));
      return;
    }
  }

  pss->subscribed.insert(channelId);

  // Enqueue the subscribe-ack BEFORE any data frames. Because the LWS
  // writable callback drains `pss->outbound` strictly in FIFO order, any
  // subsequent data we push will arrive after the ack. This spares clients
  // from having to tolerate data on a channel before they've seen the ack.
  enqueueResult(wsi, pss, requestId, json::object());

  // Now flush anything the port had queued before the subscription arrived.
  // Direct drain on the service thread; no doorbell needed since we're
  // already here.
  drainWritePort(channelId);
}

void Impl::handleUnsubscribe(struct lws *wsi, CosimSessionPss *pss,
                             uint64_t requestId, const json &params) {
  auto chIdIt = params.find("channel_id");
  if (chIdIt == params.end() || !chIdIt->is_number_unsigned()) {
    enqueueError(wsi, pss, requestId, "protocol_error",
                 "unsubscribe requires unsigned \"channel_id\"");
    return;
  }
  uint16_t channelId = chIdIt->get<uint16_t>();

  auto removed = pss->subscribed.erase(channelId);
  if (!removed) {
    enqueueError(wsi, pss, requestId, "not_subscribed",
                 std::format("Channel id {} is not subscribed",
                             static_cast<unsigned>(channelId)));
    return;
  }
  // The ack is the next outbound frame on this connection; since we run on
  // the service thread and `drainWritePort` only enqueues for subscribed
  // channels, no further data frames for this channel can land after this
  // ack — exactly the ordering the protocol spec requires.
  enqueueResult(wsi, pss, requestId, json::object());
}

//===----------------------------------------------------------------------===//
// Impl - outbound enqueue helpers (service thread only)
//===----------------------------------------------------------------------===//

void Impl::enqueueFrame(struct lws *wsi, CosimSessionPss *pss, WireFrame frame) {
  pss->outbound.push_back(std::move(frame));
  lws_callback_on_writable(wsi);
}

void Impl::enqueueResult(struct lws *wsi, CosimSessionPss *pss,
                         uint64_t requestId, const json &result) {
  json resp;
  resp["type"] = "response";
  resp["request_id"] = requestId;
  resp["result"] = result;
  enqueueFrame(wsi, pss, buildLwsTextFrame(resp.dump()));
}

void Impl::enqueueError(struct lws *wsi, CosimSessionPss *pss,
                        uint64_t requestId, const std::string &code,
                        const std::string &message) {
  json resp;
  resp["type"] = "response";
  resp["request_id"] = requestId;
  resp["error"] = {{"code", code}, {"message", message}};
  enqueueFrame(wsi, pss, buildLwsTextFrame(resp.dump()));
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
