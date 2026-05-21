//===- RpcClient.cpp - ESI Cosim RPC client implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Client implementation of the cosim WebSocket + JSON protocol, backed by
// libwebsockets. The wire protocol is documented in cosim-protocol.md.
//
// Two threads cooperate here:
//
//   * The LWS *service thread* (`serviceLoop`) runs `lws_service()` in a
//     loop. It is the only thread allowed to touch the WebSocket. All
//     outbound traffic from other threads is enqueued in `outbound` and
//     dispatched through `lws_cancel_service()` → `LWS_CALLBACK_EVENT_WAIT_
//     CANCELLED` → `lws_callback_on_writable()` → `LWS_CALLBACK_CLIENT_
//     WRITEABLE`.
//
//   * The *transport thread* (`transportLoop`) sleeps on `readyIds` and
//     drains per-channel TSQueues by invoking the user-supplied callbacks.
//     It is kept separate from the service thread so a slow or stuck user
//     callback (which may return `false` to defer a message) does not stall
//     the LWS event loop or other channels.
//
//===----------------------------------------------------------------------===//

#include "esi/backends/RpcClient.h"
#include "esi/Logging.h"
#include "esi/Utils.h"

#include "Base64.h"
#include "RpcWire.h"

#include <libwebsockets.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace esi;
using namespace esi::backends::cosim;
using esi::cosim::buildLwsBinaryFrame;
using esi::cosim::buildLwsTextFrame;
using esi::cosim::WireFrame;
using json = nlohmann::json;

namespace {
class ReadChannelConnectionImpl;
} // namespace

//===----------------------------------------------------------------------===//
// RpcClient::Impl - hides libwebsockets + JSON behind the public header
//===----------------------------------------------------------------------===//

class RpcClient::Impl {
public:
  Impl(Logger &logger, const std::string &hostname, uint16_t port);
  ~Impl();

  Logger &getLogger() { return logger; }

  uint32_t getEsiVersion() const { return esiVersion; }
  std::vector<uint8_t> getCompressedManifest() const { return manifest; }

  bool getChannelDesc(const std::string &name,
                      RpcClient::ChannelDesc &desc) const;
  std::vector<RpcClient::ChannelDesc> listChannels() const;

  void writeToServer(const std::string &channelName, const MessageData &data);

  std::unique_ptr<RpcClient::ReadChannelConnection>
  connectClientReceiver(const std::string &channelName,
                        RpcClient::ReadCallback callback);

  // Helpers used by ReadChannelConnectionImpl.
  void unsubscribe(uint16_t channelId);
  void unregisterCallback(uint16_t channelId);

  // LWS callback dispatch (service thread only).
  int callback(struct lws *wsi, enum lws_callback_reasons reason, void *user,
               void *in, size_t len);

private:
  struct ChannelMeta {
    uint16_t id;
    std::string name;
    std::string type;
    RpcClient::ChannelDirection direction;
  };

  // ---- LWS / receive plumbing ----
  void serviceLoop();
  void handleControlFrame(const std::string &text);
  void handleBinaryFrame(const std::string &data);
  void failAllPending(const std::string &reason);
  void enqueueText(const std::string &text);
  void enqueueBinary(WireFrame frame);

  // ---- request/response plumbing ----
  struct PendingRequest {
    std::promise<json> promise;
  };
  /// Issue a JSON-RPC style request and synchronously await the response.
  /// Throws on transport or server-reported error.
  json call(const std::string &method, json params);

  std::string host;
  uint16_t port;
  Logger &logger;

  // Manifest + channel table cached from the `hello` response.
  uint32_t esiVersion = 0;
  std::vector<uint8_t> manifest;
  std::unordered_map<std::string, ChannelMeta> channelsByName;
  std::unordered_map<uint16_t, ChannelMeta> channelsById;

  // Pending control-plane requests, keyed by id.
  std::mutex pendingMutex;
  std::atomic<uint64_t> nextRequestId{1};
  std::unordered_map<uint64_t, PendingRequest> pending;

  // Per-channel inbound state. Connect-time `subscribe` registers; the
  // ReadChannelConnection destructor unregisters and sends `unsubscribe`.
  //
  // Each entry owns its own non-blocking TSQueue: the LWS service thread
  // pushes into the queue on inbound binary frames, and the transport
  // thread is what actually invokes the user-supplied callback. That
  // decoupling makes the client cross-channel-deadlock-safe regardless of
  // what the user's callback does: a callback that returns `false`
  // (rejects the message) only delays its own channel.
  //
  // Held by `shared_ptr` so the transport thread can copy out an entry
  // under the lock and outlive a concurrent `unregisterCallback`: the
  // cancel flag flips, the transport thread sees it on its next iteration
  // and drops whatever's still queued.
  struct ReadCallbackEntry {
    RpcClient::ReadCallback callback;
    std::atomic<bool> canceled{false};
    uint16_t channelId;
    utils::TSQueue<MessageData> queue;

    ReadCallbackEntry(uint16_t channelId, std::function<void()> notifier)
        : channelId(channelId), queue(std::move(notifier)) {}
  };
  std::mutex readCallbacksMutex;
  std::unordered_map<uint16_t, std::shared_ptr<ReadCallbackEntry>>
      readCallbacks;

  // Doorbell for the transport thread.
  utils::ReadyIdSet<uint16_t> readyIds;
  std::thread transportThread;
  void transportLoop();

  // LWS plumbing. `wsi` is touched from the service thread only (and read
  // by `enqueueText`/`enqueueBinary` to know whether they can wake the
  // service thread to drive a write — strictly via `lws_cancel_service`,
  // which is documented thread-safe).
  struct lws_context *context = nullptr;
  struct lws *wsi = nullptr;
  std::thread serviceThread;
  std::atomic<bool> shouldStop{false};

  // Cross-thread outbound queue. Any thread can push; only the service
  // thread pops in `LWS_CALLBACK_CLIENT_WRITEABLE`.
  std::mutex outboundMutex;
  std::deque<WireFrame> outbound;

  // Service-thread-only receive accumulator for fragmented frames.
  std::string rxBuffer;

  // Connection-open synchronization. The LWS callback fires on the service
  // thread; the constructor blocks here until either Open or Failed fires.
  std::mutex openMutex;
  std::condition_variable openCV;
  enum class OpenState { Pending, Open, Failed } openState{OpenState::Pending};
  std::string openError;

  // Most recent unsolicited server-initiated error (`type="error"`), if any.
  // The server sends one of these before closing the WS to explain why (e.g.
  // `server_busy`). Recorded so the imminent Close path can surface it.
  std::mutex lastServerErrorMutex;
  std::string lastServerError;

  std::atomic<bool> disconnecting{false};
};

//===----------------------------------------------------------------------===//
// ReadChannelConnectionImpl
//===----------------------------------------------------------------------===//

namespace {
class ReadChannelConnectionImpl : public RpcClient::ReadChannelConnection {
public:
  ReadChannelConnectionImpl(RpcClient::Impl *impl, uint16_t channelId)
      : impl(impl), channelId(channelId) {}
  ~ReadChannelConnectionImpl() override { disconnect(); }

  void disconnect() override {
    if (disconnected.exchange(true))
      return;
    impl->unregisterCallback(channelId);
    try {
      impl->unsubscribe(channelId);
    } catch (const std::exception &e) {
      // We're tearing down a per-channel reader from a destructor; can't
      // propagate exceptions. Log so the failure isn't silently lost.
      impl->getLogger().warning(
          "cosim",
          std::string("unsubscribe during disconnect failed: ") + e.what());
    }
  }

private:
  RpcClient::Impl *impl;
  uint16_t channelId;
  std::atomic<bool> disconnected{false};
};
} // namespace

//===----------------------------------------------------------------------===//
// LWS protocol table
//===----------------------------------------------------------------------===//

namespace {
// LWS allocates `per_session_data_size` bytes per connection. The client
// only ever has one connection so we don't need any C++ state in that
// region; route everything through `lws_context_user()`.
struct ClientPss {};

static int rpcClientCallback(struct lws *wsi, enum lws_callback_reasons reason,
                             void *user, void *in, size_t len) {
  struct lws_context *ctx = lws_get_context(wsi);
  if (!ctx)
    return 0;
  void *impl = lws_context_user(ctx);
  if (!impl)
    return 0;
  return static_cast<RpcClient::Impl *>(impl)->callback(wsi, reason, user, in,
                                                        len);
}

static const struct lws_protocols kClientProtocols[] = {
    {"esi-cosim-v3", rpcClientCallback, sizeof(ClientPss), /*rx_buffer_size=*/0,
     /*id=*/0, /*user=*/nullptr, /*tx_packet_size=*/0},
    LWS_PROTOCOL_LIST_TERM};
} // namespace

//===----------------------------------------------------------------------===//
// RpcClient::Impl - construction / `hello` handshake
//===----------------------------------------------------------------------===//

RpcClient::Impl::Impl(Logger &logger, const std::string &hostname,
                      uint16_t port)
    : host(hostname), port(port), logger(logger) {
  // Quiet libwebsockets' own log spam; keep warnings and errors.
  lws_set_log_level(LLL_ERR | LLL_WARN, nullptr);

  struct lws_context_creation_info info = {};
  info.options =
      LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT * 0; // explicit no-SSL globals
  info.port = CONTEXT_PORT_NO_LISTEN;
  info.protocols = kClientProtocols;
  info.user = this;
  info.gid = -1;
  info.uid = -1;

  context = lws_create_context(&info);
  if (!context)
    throw std::runtime_error("RpcClient: lws_create_context failed");

  // Kick off the client connect. The actual TCP/WS handshake happens once
  // `lws_service` runs on the service thread we are about to start.
  struct lws_client_connect_info ci = {};
  ci.context = context;
  ci.address = host.c_str();
  ci.port = static_cast<int>(port);
  ci.path = "/esi/cosim/v3";
  ci.host = ci.address;
  ci.origin = ci.address;
  ci.protocol = "esi-cosim-v3";
  ci.local_protocol_name = "esi-cosim-v3";
  ci.pwsi = &wsi;

  if (!lws_client_connect_via_info(&ci)) {
    lws_context_destroy(context);
    context = nullptr;
    throw std::runtime_error("RpcClient: lws_client_connect_via_info failed");
  }

  serviceThread = std::thread([this] { serviceLoop(); });

  // Wait for the WS to reach Open (or fail).
  {
    std::unique_lock<std::mutex> lock(openMutex);
    openCV.wait(lock, [&] { return openState != OpenState::Pending; });
    if (openState == OpenState::Failed) {
      // Tear down before throwing so the destructor isn't expected to clean
      // up a half-constructed object.
      std::string err = openError;
      shouldStop.store(true, std::memory_order_release);
      lws_cancel_service(context);
      if (serviceThread.joinable())
        serviceThread.join();
      lws_context_destroy(context);
      context = nullptr;
      throw std::runtime_error("RpcClient: failed to connect to ws://" + host +
                               ":" + std::to_string(port) + ": " + err);
    }
  }

  // Send `hello`; the server defers the response until its manifest is ready.
  json result = call("hello", {{"client_protocol_version", 3}});

  if (auto it = result.find("esi_version"); it != result.end())
    esiVersion = it->get<uint32_t>();
  if (auto it = result.find("compressed_manifest_b64"); it != result.end()) {
    std::string raw;
    std::string err = macaron::Base64::Decode(it->get<std::string>(), raw);
    if (!err.empty())
      throw std::runtime_error("RpcClient: invalid base64 in manifest blob: " +
                               err);
    manifest.assign(raw.begin(), raw.end());
  }

  auto channelsIt = result.find("channels");
  if (channelsIt == result.end() || !channelsIt->is_array())
    throw std::runtime_error("RpcClient: hello response missing channels");
  for (const json &c : *channelsIt) {
    ChannelMeta meta;
    meta.id = c.at("channel_id").get<uint16_t>();
    meta.name = c.at("name").get<std::string>();
    meta.type = c.at("type").get<std::string>();
    std::string dir = c.at("direction").get<std::string>();
    meta.direction = dir == "to_server" ? RpcClient::ChannelDirection::ToServer
                                        : RpcClient::ChannelDirection::ToClient;
    channelsByName.emplace(meta.name, meta);
    channelsById.emplace(meta.id, std::move(meta));
  }

  // Start the transport thread now that initialisation is complete.
  transportThread = std::thread([this] { transportLoop(); });
}

RpcClient::Impl::~Impl() {
  disconnecting.store(true);
  failAllPending("client shutdown");
  // Mark every per-channel entry canceled so the transport loop drops any
  // remaining queued frames promptly.
  {
    std::lock_guard<std::mutex> lock(readCallbacksMutex);
    for (auto &[id, entry] : readCallbacks)
      entry->canceled.store(true);
  }

  // Stop the LWS service thread first. Once joined, no more inbound frames
  // can be pushed into per-channel queues.
  if (context) {
    shouldStop.store(true, std::memory_order_release);
    lws_cancel_service(context);
    if (serviceThread.joinable())
      serviceThread.join();
    lws_context_destroy(context);
    context = nullptr;
    wsi = nullptr;
  }

  // Retire the transport thread.
  readyIds.requestShutdown();
  if (transportThread.joinable())
    transportThread.join();
}

//===----------------------------------------------------------------------===//
// RpcClient::Impl - service loop and LWS callback
//===----------------------------------------------------------------------===//

void RpcClient::Impl::serviceLoop() {
  while (!shouldStop.load(std::memory_order_acquire)) {
    int n = lws_service(context, 0);
    if (n < 0)
      break;
  }
}

int RpcClient::Impl::callback(struct lws *wsi, enum lws_callback_reasons reason,
                              void * /*user*/, void *in, size_t len) {
  switch (reason) {
  case LWS_CALLBACK_CLIENT_ESTABLISHED: {
    std::lock_guard<std::mutex> lock(openMutex);
    openState = OpenState::Open;
    openCV.notify_all();
    return 0;
  }

  case LWS_CALLBACK_CLIENT_CONNECTION_ERROR: {
    std::string reason = in && len
                             ? std::string(static_cast<const char *>(in), len)
                             : std::string("connection error");
    {
      std::lock_guard<std::mutex> lock(openMutex);
      if (openState == OpenState::Pending) {
        openState = OpenState::Failed;
        openError = reason;
        openCV.notify_all();
      }
    }
    failAllPending("websocket error: " + reason);
    this->wsi = nullptr;
    return 0;
  }

  case LWS_CALLBACK_CLIENT_CLOSED: {
    std::string reason;
    {
      std::lock_guard<std::mutex> lock(lastServerErrorMutex);
      if (!lastServerError.empty())
        reason = lastServerError;
    }
    if (reason.empty())
      reason = "connection closed";
    {
      std::lock_guard<std::mutex> lock(openMutex);
      if (openState == OpenState::Pending) {
        openState = OpenState::Failed;
        openError = reason;
        openCV.notify_all();
      }
    }
    failAllPending("websocket closed: " + reason);
    this->wsi = nullptr;
    return 0;
  }

  case LWS_CALLBACK_CLIENT_RECEIVE: {
    if (in && len)
      rxBuffer.append(static_cast<const char *>(in), len);
    if (lws_is_final_fragment(wsi) && lws_remaining_packet_payload(wsi) == 0) {
      bool binary = lws_frame_is_binary(wsi);
      std::string data;
      data.swap(rxBuffer);
      if (binary)
        handleBinaryFrame(data);
      else
        handleControlFrame(data);
    }
    return 0;
  }

  case LWS_CALLBACK_CLIENT_WRITEABLE: {
    WireFrame f;
    bool more = false;
    {
      std::lock_guard<std::mutex> lock(outboundMutex);
      if (outbound.empty())
        return 0;
      f = std::move(outbound.front());
      outbound.pop_front();
      more = !outbound.empty();
    }
    lws_write_protocol proto = f.isBinary ? LWS_WRITE_BINARY : LWS_WRITE_TEXT;
    int m = lws_write(wsi, f.writePtr(), f.payloadSize, proto);
    if (m < static_cast<int>(f.payloadSize)) {
      logger.error("cosim", "lws_write returned short");
      return -1;
    }
    if (more)
      lws_callback_on_writable(wsi);
    return 0;
  }

  case LWS_CALLBACK_EVENT_WAIT_CANCELLED: {
    if (shouldStop.load(std::memory_order_acquire))
      return 0;
    // If we have queued outbound frames and a live wsi, ask LWS for a
    // writable callback so we can drain them.
    bool needWrite = false;
    {
      std::lock_guard<std::mutex> lock(outboundMutex);
      needWrite = !outbound.empty();
    }
    if (needWrite && this->wsi)
      lws_callback_on_writable(this->wsi);
    return 0;
  }

  default:
    return 0;
  }
}

//===----------------------------------------------------------------------===//
// RpcClient::Impl - frame handling
//===----------------------------------------------------------------------===//

void RpcClient::Impl::handleControlFrame(const std::string &text) {
  json resp;
  try {
    resp = json::parse(text);
  } catch (const std::exception &) {
    return; // Unparseable text frame: nothing useful we can do.
  }
  auto typeIt = resp.find("type");
  if (typeIt == resp.end() || !typeIt->is_string())
    return;
  std::string type = typeIt->get<std::string>();

  // Unsolicited server-initiated error. The server sends one of these before
  // closing the WS. Stash it so the imminent Close event surfaces this
  // reason instead of the raw WebSocket close reason.
  if (type == "error") {
    if (auto errIt = resp.find("error"); errIt != resp.end()) {
      std::string code = errIt->value("code", std::string("internal"));
      std::string message = errIt->value("message", std::string());
      std::lock_guard<std::mutex> lock(lastServerErrorMutex);
      lastServerError = "server error [" + code + "]: " + message;
    }
    return;
  }

  if (type != "response") {
    // Receivers MUST ignore unrecognised types for forward compatibility.
    logger.debug("cosim",
                 "Ignoring control frame with unknown type \"" + type + "\"");
    return;
  }
  auto idIt = resp.find("request_id");
  if (idIt == resp.end())
    return;
  uint64_t requestId = idIt->get<uint64_t>();

  std::promise<json> *promise = nullptr;
  {
    std::lock_guard<std::mutex> lock(pendingMutex);
    auto it = pending.find(requestId);
    if (it == pending.end())
      return;
    promise = &it->second.promise;
  }

  try {
    if (auto errIt = resp.find("error"); errIt != resp.end()) {
      std::string code = errIt->value("code", std::string("internal"));
      std::string message = errIt->value("message", std::string());
      promise->set_exception(std::make_exception_ptr(
          std::runtime_error("Server error [" + code + "]: " + message)));
    } else if (auto resultIt = resp.find("result"); resultIt != resp.end()) {
      promise->set_value(*resultIt);
    } else {
      // Per the spec, a `type=response` frame MUST carry either `result` or
      // `error`. Surface the protocol violation instead of pretending it
      // succeeded.
      promise->set_exception(std::make_exception_ptr(std::runtime_error(
          "Server response missing both \"result\" and \"error\" fields "
          "(protocol violation) for request_id=" +
          std::to_string(requestId))));
    }
  } catch (const std::exception &e) {
    logger.debug("cosim",
                 std::string("failed to fulfill promise for request_id=") +
                     std::to_string(requestId) + ": " + e.what());
  }

  std::lock_guard<std::mutex> lock(pendingMutex);
  pending.erase(requestId);
}

void RpcClient::Impl::handleBinaryFrame(const std::string &data) {
  if (data.size() < 2)
    return;
  uint16_t channelId =
      static_cast<uint8_t>(data[0]) |
      (static_cast<uint16_t>(static_cast<uint8_t>(data[1])) << 8);

  std::shared_ptr<ReadCallbackEntry> entry;
  {
    std::lock_guard<std::mutex> lock(readCallbacksMutex);
    auto it = readCallbacks.find(channelId);
    if (it == readCallbacks.end()) {
      logger.warning("cosim", "Dropping inbound frame for channel id " +
                                  std::to_string(channelId) +
                                  ": no active subscriber");
      return;
    }
    entry = it->second;
  }

  // Push into the per-channel queue. The queue's notifier rings the
  // transport doorbell; the transport thread is what actually invokes the
  // user callback. We must never block the LWS service thread on user code.
  entry->queue.push(MessageData(
      reinterpret_cast<const uint8_t *>(data.data() + 2), data.size() - 2));
}

void RpcClient::Impl::failAllPending(const std::string &reason) {
  std::lock_guard<std::mutex> lock(pendingMutex);
  for (auto &[id, pr] : pending) {
    try {
      pr.promise.set_exception(
          std::make_exception_ptr(std::runtime_error(reason)));
    } catch (const std::exception &e) {
      logger.debug("cosim", std::string("failAllPending: request_id=") +
                                std::to_string(id) +
                                " already satisfied: " + e.what());
    }
  }
  pending.clear();
}

//===----------------------------------------------------------------------===//
// RpcClient::Impl - cross-thread send helpers
//===----------------------------------------------------------------------===//

void RpcClient::Impl::enqueueText(const std::string &text) {
  WireFrame f = buildLwsTextFrame(text);
  {
    std::lock_guard<std::mutex> lock(outboundMutex);
    outbound.push_back(std::move(f));
  }
  if (context)
    lws_cancel_service(context);
}

void RpcClient::Impl::enqueueBinary(WireFrame frame) {
  {
    std::lock_guard<std::mutex> lock(outboundMutex);
    outbound.push_back(std::move(frame));
  }
  if (context)
    lws_cancel_service(context);
}

//===----------------------------------------------------------------------===//
// RpcClient::Impl - request/response, sends
//===----------------------------------------------------------------------===//

json RpcClient::Impl::call(const std::string &method, json params) {
  uint64_t requestId = nextRequestId.fetch_add(1);
  std::future<json> future;
  {
    std::lock_guard<std::mutex> lock(pendingMutex);
    future = pending[requestId].promise.get_future();
  }

  json req;
  req["type"] = "request";
  req["request_id"] = requestId;
  req["method"] = method;
  req["params"] = std::move(params);
  enqueueText(req.dump());

  return future.get();
}

void RpcClient::Impl::writeToServer(const std::string &channelName,
                                    const MessageData &data) {
  auto it = channelsByName.find(channelName);
  if (it == channelsByName.end())
    throw std::runtime_error("Unknown channel '" + channelName + "'");
  if (it->second.direction != RpcClient::ChannelDirection::ToServer)
    throw std::runtime_error("Channel '" + channelName +
                             "' is not a to-server channel");

  enqueueBinary(buildLwsBinaryFrame(it->second.id, data.getBytes(),
                                    data.getSize()));
}

//===----------------------------------------------------------------------===//
// RpcClient::Impl - channel descriptor lookup
//===----------------------------------------------------------------------===//

bool RpcClient::Impl::getChannelDesc(const std::string &name,
                                     RpcClient::ChannelDesc &desc) const {
  auto it = channelsByName.find(name);
  if (it == channelsByName.end())
    return false;
  desc.name = it->second.name;
  desc.type = it->second.type;
  desc.dir = it->second.direction;
  return true;
}

std::vector<RpcClient::ChannelDesc> RpcClient::Impl::listChannels() const {
  std::vector<RpcClient::ChannelDesc> out;
  out.reserve(channelsByName.size());
  for (const auto &[name, meta] : channelsByName) {
    RpcClient::ChannelDesc d;
    d.name = meta.name;
    d.type = meta.type;
    d.dir = meta.direction;
    out.push_back(std::move(d));
  }
  return out;
}

//===----------------------------------------------------------------------===//
// RpcClient::Impl - subscribe / unsubscribe
//===----------------------------------------------------------------------===//

std::unique_ptr<RpcClient::ReadChannelConnection>
RpcClient::Impl::connectClientReceiver(const std::string &channelName,
                                       RpcClient::ReadCallback callback) {
  auto it = channelsByName.find(channelName);
  if (it == channelsByName.end())
    throw std::runtime_error("Unknown channel '" + channelName + "'");
  if (it->second.direction != RpcClient::ChannelDirection::ToClient)
    throw std::runtime_error("Channel '" + channelName +
                             "' is not a to-client channel");
  uint16_t channelId = it->second.id;

  // Register the callback first so any racing inbound binary frame after the
  // server's subscribe-ack has somewhere to land.
  {
    std::lock_guard<std::mutex> lock(readCallbacksMutex);
    auto entry = std::make_shared<ReadCallbackEntry>(
        channelId, [this, channelId] { readyIds.markReady(channelId); });
    entry->callback = std::move(callback);
    readCallbacks[channelId] = std::move(entry);
  }

  try {
    call("subscribe", {{"channel_id", channelId}});
  } catch (...) {
    std::lock_guard<std::mutex> lock(readCallbacksMutex);
    readCallbacks.erase(channelId);
    throw;
  }

  return std::make_unique<ReadChannelConnectionImpl>(this, channelId);
}

void RpcClient::Impl::unsubscribe(uint16_t channelId) {
  if (disconnecting.load())
    return;
  // Skip the RPC if the WS is no longer up (server tore down first, network
  // drop, etc.). The unsubscribe is moot anyway and otherwise every
  // per-channel cleanup during teardown would generate a noisy "failed to
  // send unsubscribe" warning.
  {
    std::lock_guard<std::mutex> lock(openMutex);
    if (openState != OpenState::Open)
      return;
  }
  if (!wsi)
    return;
  call("unsubscribe", {{"channel_id", channelId}});
}

void RpcClient::Impl::unregisterCallback(uint16_t channelId) {
  std::shared_ptr<ReadCallbackEntry> entry;
  {
    std::lock_guard<std::mutex> lock(readCallbacksMutex);
    auto it = readCallbacks.find(channelId);
    if (it == readCallbacks.end())
      return;
    entry = it->second;
    readCallbacks.erase(it);
  }
  // Flip the cancel flag on the entry that any in-flight transport-loop
  // iteration already copied out, so it bails out promptly.
  entry->canceled.store(true);
  // Nudge the transport thread so it observes the cancelation promptly.
  readyIds.markReady(channelId);
}

//===----------------------------------------------------------------------===//
// RpcClient::Impl - transport thread (per-channel deadlock isolation)
//===----------------------------------------------------------------------===//

void RpcClient::Impl::transportLoop() {
  // Channels whose last delivery attempt left a message at the head of the
  // queue (user callback returned `false`). We retry them after a short
  // backoff so other channels still get a chance every wake, and so a
  // permanently-busy channel doesn't pin the CPU.
  std::unordered_set<uint16_t> retry;

  while (true) {
    std::unordered_set<uint16_t> ids;
    if (!readyIds.waitDrain(
            ids, retry.empty() ? std::optional<std::chrono::milliseconds>{}
                               : std::chrono::milliseconds(1)))
      return;
    ids.insert(retry.begin(), retry.end());
    retry.clear();

    for (uint16_t id : ids) {
      std::shared_ptr<ReadCallbackEntry> entry;
      {
        std::lock_guard<std::mutex> lock(readCallbacksMutex);
        auto it = readCallbacks.find(id);
        if (it == readCallbacks.end())
          continue;
        entry = it->second;
      }

      if (entry->canceled.load() || disconnecting.load())
        continue;

      // Best-effort drain. `TSQueue::pop(callback)` peeks at the front and
      // only pops if our callback returns true; on `false` we leave the
      // message at the head and mark the channel for retry.
      bool stuck = false;
      while (!stuck && !entry->canceled.load() && !disconnecting.load()) {
        bool gotOne = false;
        entry->queue.pop([&](const MessageData &data) {
          gotOne = true;
          std::unique_ptr<SegmentedMessageData> msg =
              std::make_unique<MessageData>(data);
          if (entry->callback(msg))
            return true;
          stuck = true;
          return false;
        });
        if (!gotOne)
          break; // queue empty
        if (stuck)
          retry.insert(id);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// RpcClient - public pass-throughs (unchanged API)
//===----------------------------------------------------------------------===//

RpcClient::RpcClient(Logger &logger, const std::string &hostname, uint16_t port)
    : impl(std::make_unique<Impl>(logger, hostname, port)) {}

RpcClient::~RpcClient() = default;

uint32_t RpcClient::getEsiVersion() const { return impl->getEsiVersion(); }

std::vector<uint8_t> RpcClient::getCompressedManifest() const {
  return impl->getCompressedManifest();
}

bool RpcClient::getChannelDesc(const std::string &channelName,
                               ChannelDesc &desc) const {
  return impl->getChannelDesc(channelName, desc);
}

std::vector<RpcClient::ChannelDesc> RpcClient::listChannels() const {
  return impl->listChannels();
}

void RpcClient::writeToServer(const std::string &channelName,
                              const MessageData &data) {
  impl->writeToServer(channelName, data);
}

std::unique_ptr<RpcClient::ReadChannelConnection>
RpcClient::connectClientReceiver(const std::string &channelName,
                                 ReadCallback callback) {
  return impl->connectClientReceiver(channelName, std::move(callback));
}
