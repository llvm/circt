//===- RpcClient.cpp - ESI Cosim RPC client implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Client implementation of the cosim WebSocket + JSON protocol. The wire
// protocol is documented in cosim-protocol.md.
//
//===----------------------------------------------------------------------===//

#include "esi/backends/RpcClient.h"
#include "esi/Logging.h"
#include "esi/Utils.h"

#include "RpcWire.h"

#include <ixwebsocket/IXBase64.h>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXWebSocketCloseConstants.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace esi;
using namespace esi::backends::cosim;
using json = nlohmann::json;

//===----------------------------------------------------------------------===//
// RpcClient::Impl - hides IXWebSocket + JSON behind the public header
//===----------------------------------------------------------------------===//

namespace {
class ReadChannelConnectionImpl;
} // namespace

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

  void connectToServer(const std::string &channelName,
                       RpcClient::CompletionCallback onConnected);
  void writeToServer(const std::string &channelName, const MessageData &data);

  std::unique_ptr<RpcClient::ReadChannelConnection>
  connectClientReceiver(const std::string &channelName,
                        RpcClient::ReadCallback callback,
                        RpcClient::CompletionCallback onSubscribed);

  // Helpers used by ReadChannelConnectionImpl.
  void unsubscribe(uint64_t channelId);
  void unregisterCallback(uint64_t channelId);

private:
  // Completion of an RPC request. Called exactly once, on IX's network thread
  // or the thread that fails the request, with either the result or an
  // exception. Must not block.
  using Completion =
      std::function<void(const json *result, std::exception_ptr error)>;

  struct ChannelMeta {
    uint64_t id;
    std::string name;
    std::string type;
    RpcClient::ChannelDirection direction;
  };

  // ---- websocket lifecycle ----
  void onMessage(const ix::WebSocketMessagePtr &msg);
  void handleControlFrame(const std::string &text);
  void handleBinaryFrame(const std::string &data);
  void failAllPending(const std::string &reason);

  // ---- request/response plumbing ----

  /// Issue a JSON-RPC style request and synchronously await the response.
  /// Throws on error (transport or server-reported) or once `timeout` passes
  /// without a response. With no timeout, waits until a response arrives or
  /// the connection is lost.
  json
  call(const std::string &method, json params,
       std::optional<std::chrono::seconds> timeout = std::chrono::seconds(30));
  /// Issue a request without waiting. `done` runs once the response arrives
  /// or the request fails (possibly before this returns).
  void callAsync(const std::string &method, json params, Completion done);

  /// Look up a to-server channel by name, throwing if it isn't one.
  const ChannelMeta &getToServerChannel(const std::string &channelName) const;
  /// Send one data frame. Throws on a transport failure.
  void sendFrame(const ChannelMeta &meta, const MessageData &data);

  std::string host;
  Logger &logger;
  ix::WebSocket ws;

  // Manifest + channel table cached from the `hello` response.
  uint32_t esiVersion = 0;
  std::vector<uint8_t> manifest;
  std::unordered_map<std::string, ChannelMeta> channelsByName;
  std::unordered_map<uint64_t, ChannelMeta> channelsById;

  // Set (under `pendingMutex`) once the connection is lost, so no new
  // request can be left pending forever.
  std::atomic<bool> connectionLost{false};

  // Pending control-plane requests, keyed by id.
  std::mutex pendingMutex;
  std::atomic<uint64_t> nextRequestId{1};
  std::unordered_map<uint64_t, Completion> pending;

  // Per-channel read state. Connection-time `subscribe` registers; the
  // ReadChannelConnection destructor unregisters and sends `unsubscribe`.
  //
  // Each entry owns its own non-blocking TSQueue: `handleBinaryFrame` only
  // pushes into the queue (on IX's single per-connection network thread),
  // and a dedicated transport thread is what actually invokes the
  // user-supplied callback. That decoupling is what makes the client
  // cross-channel-deadlock-safe regardless of what the user's callback
  // does: a callback that returns `false` (rejects the message) only
  // delays its own channel -- the IX thread is free to keep delivering on
  // other channels.
  //
  // The queue's `notifier` rings the server-wide doorbell
  // (`Impl::markReady`) so the transport thread wakes up only when there's
  // work for a specific channel.
  //
  // Held by `shared_ptr` so the transport thread can copy the entry under
  // the lock and outlive a concurrent `unregisterCallback` that erases the
  // map entry: the cancel flag flips, the transport thread sees it on its
  // next iteration, and drops anything still in the queue.
  struct ReadCallbackEntry {
    RpcClient::ReadCallback callback;
    std::atomic<bool> canceled{false};
    uint64_t channelId;
    utils::TSQueue<MessageData> queue;

    ReadCallbackEntry(uint64_t channelId, std::function<void()> notifier)
        : channelId(channelId), queue(std::move(notifier)) {}
  };
  std::mutex readCallbacksMutex;
  std::unordered_map<uint64_t, std::shared_ptr<ReadCallbackEntry>>
      readCallbacks;

  // Dirty-set doorbell for the transport thread. Each entry's TSQueue
  // notifier inserts the channel id into `readyIds` and wakes the consumer.
  utils::ReadyIdSet<uint64_t> readyIds;
  std::thread transportThread;

  void transportLoop();

  // Connection-open synchronization. The WS callback fires on IX's background
  // thread, so the constructor blocks here until either Open or Error fires.
  std::mutex openMutex;
  std::condition_variable openCV;
  enum class OpenState { Pending, Open, Failed } openState{OpenState::Pending};
  std::string openError;

  // Most recent unsolicited server-initiated error (`type="error"`), if any.
  // The server sends one of these before closing the WS to explain why (e.g.
  // `server_busy`). Recorded so a subsequent Close/Error path can surface it
  // instead of the much-less-actionable raw WebSocket close reason.
  std::mutex lastServerErrorMutex;
  std::string lastServerError;

  std::atomic<bool> disconnecting{false};

  // Cross-thread fault propagation out of the IX network thread. See
  // FaultStash docs in RpcWire.h.
  ::esi::cosim::FaultStash faultStash;
};

//===----------------------------------------------------------------------===//
// ReadChannelConnectionImpl
//
// Held by the consumer of `connectClientReceiver`. Destructor unregisters the
// callback and sends `unsubscribe`. The actual incoming bytes are delivered
// by the shared websocket callback in RpcClient::Impl.
//===----------------------------------------------------------------------===//

namespace {
class ReadChannelConnectionImpl : public RpcClient::ReadChannelConnection {
public:
  ReadChannelConnectionImpl(RpcClient::Impl *impl, uint64_t channelId)
      : impl(impl), channelId(channelId) {}
  ~ReadChannelConnectionImpl() override { disconnect(); }

  void disconnect() override {
    if (disconnected.exchange(true))
      return;
    impl->unregisterCallback(channelId);
    try {
      impl->unsubscribe(channelId);
    } catch (const std::exception &e) {
      // We're tearing down a per-channel reader and can't propagate the
      // exception (this runs in a destructor). Log it so the failure isn't
      // silently lost.
      impl->getLogger().warning(
          "cosim",
          std::string("unsubscribe during disconnect failed: ") + e.what());
    }
  }

private:
  RpcClient::Impl *impl;
  uint64_t channelId;
  std::atomic<bool> disconnected{false};
};
} // namespace

//===----------------------------------------------------------------------===//
// RpcClient::Impl - construction / `hello` handshake
//===----------------------------------------------------------------------===//

RpcClient::Impl::Impl(Logger &logger, const std::string &hostname,
                      uint16_t port)
    : host(hostname), logger(logger) {
  // On Windows, `ix::initNetSystem()` calls `WSAStartup`; bail out cleanly if
  // it fails rather than letting the WS start path fail with a less
  // actionable `WSANOTINITIALISED`. No-op on other platforms.
  if (!ix::initNetSystem())
    throw std::runtime_error(
        "RpcClient: ix::initNetSystem() failed (WSAStartup)");

  ws.setUrl("ws://" + host + ":" + std::to_string(port) + "/esi/cosim/v" +
            std::to_string(::esi::cosim::kProtocolVersion));
  ws.disablePerMessageDeflate();
  ws.disableAutomaticReconnection();

  ws.setOnMessageCallback(
      [this](const ix::WebSocketMessagePtr &msg) { onMessage(msg); });

  ws.start();

  // Wait for the WS to reach Open (or fail).
  {
    std::unique_lock<std::mutex> lock(openMutex);
    openCV.wait(lock, [&] { return openState != OpenState::Pending; });
    if (openState == OpenState::Failed)
      throw std::runtime_error("RpcClient: failed to connect to " +
                               ws.getUrl() + ": " + openError);
  }

  // Send `hello`; the server blocks the response until its manifest is ready.
  json result = call(
      "hello", {{"client_protocol_version", ::esi::cosim::kProtocolVersion}});
  auto verIt = result.find("protocol_version");
  if (verIt == result.end() || !verIt->is_number_unsigned() ||
      verIt->get<uint64_t>() != ::esi::cosim::kProtocolVersion)
    throw std::runtime_error(
        "RpcClient: server speaks an unsupported cosim protocol version (" +
        (verIt == result.end() ? std::string("none") : verIt->dump()) +
        "); expected " + std::to_string(::esi::cosim::kProtocolVersion));

  // Cache manifest blob.
  if (auto it = result.find("esi_version"); it != result.end())
    esiVersion = it->get<uint32_t>();
  if (auto it = result.find("compressed_manifest_b64"); it != result.end()) {
    // macaron::Base64::Decode returns an empty error string on success and
    // writes raw bytes into the out string; reinterpret as a byte vector.
    std::string raw;
    std::string err = macaron::Base64::Decode(it->get<std::string>(), raw);
    if (!err.empty())
      throw std::runtime_error("RpcClient: invalid base64 in manifest blob: " +
                               err);
    manifest.assign(raw.begin(), raw.end());
  }

  // Cache channel table.
  auto channelsIt = result.find("channels");
  if (channelsIt == result.end() || !channelsIt->is_array())
    throw std::runtime_error("RpcClient: hello response missing channels");
  for (const json &c : *channelsIt) {
    ChannelMeta meta;
    meta.id = c.at("channel_id").get<uint64_t>();
    meta.name = c.at("name").get<std::string>();
    meta.type = c.at("type").get<std::string>();
    std::string dir = c.at("direction").get<std::string>();
    meta.direction = dir == "to_server" ? RpcClient::ChannelDirection::ToServer
                                        : RpcClient::ChannelDirection::ToClient;
    channelsByName.emplace(meta.name, meta);
    channelsById.emplace(meta.id, std::move(meta));
  }

  // Start the transport thread now that initialisation is complete. It
  // sleeps on `readyIds`'s internal CV until a per-channel TSQueue push
  // wakes it.
  transportThread = std::thread([this] { transportLoop(); });
}

RpcClient::Impl::~Impl() {
  disconnecting = true;
  failAllPending("client shutdown");
  // Mark every per-channel entry canceled so the transport loop drops any
  // remaining queued frames promptly instead of spinning on the user
  // callback.
  {
    std::lock_guard<std::mutex> lock(readCallbacksMutex);
    for (auto &[id, entry] : readCallbacks)
      entry->canceled.store(true);
  }
  // Stop IX *first* so no more `handleBinaryFrame` calls can fire and push
  // into the per-channel queues. ws.stop() joins IX's internal threads
  // synchronously.
  ws.stop();
  // Now retire the transport thread; nothing else is producing for it.
  if (transportThread.joinable()) {
    readyIds.requestShutdown();
    transportThread.join();
  }
}

//===----------------------------------------------------------------------===//
// RpcClient::Impl - WebSocket message dispatch
//===----------------------------------------------------------------------===//

void RpcClient::Impl::onMessage(const ix::WebSocketMessagePtr &msg) {
  switch (msg->type) {
  case ix::WebSocketMessageType::Open: {
    std::lock_guard<std::mutex> lock(openMutex);
    openState = OpenState::Open;
    openCV.notify_all();
    return;
  }
  case ix::WebSocketMessageType::Error: {
    std::string reason = msg->errorInfo.reason;
    {
      std::lock_guard<std::mutex> lock(lastServerErrorMutex);
      if (!lastServerError.empty())
        reason = lastServerError;
    }
    {
      std::lock_guard<std::mutex> lock(openMutex);
      if (openState == OpenState::Pending) {
        openState = OpenState::Failed;
        openError = reason;
        openCV.notify_all();
      }
    }
    failAllPending("websocket error: " + reason);
    return;
  }
  case ix::WebSocketMessageType::Close: {
    std::string reason = msg->closeInfo.reason;
    {
      std::lock_guard<std::mutex> lock(lastServerErrorMutex);
      if (!lastServerError.empty())
        reason = lastServerError;
    }
    // If the server closed us before `Open` settled (e.g. immediate
    // `server_busy` reject), surface that as an Open failure so the
    // constructor's wait unblocks with a useful error.
    {
      std::lock_guard<std::mutex> lock(openMutex);
      if (openState == OpenState::Pending) {
        openState = OpenState::Failed;
        openError = reason;
        openCV.notify_all();
      }
    }
    failAllPending("websocket closed: " + reason);
    return;
  }
  case ix::WebSocketMessageType::Message:
    // An exception escaping this callback would kill IX's network thread.
    // Stash the first one so the next public RpcClient method rethrows it
    // on the user's thread.
    try {
      if (msg->binary)
        handleBinaryFrame(msg->str);
      else
        handleControlFrame(msg->str);
    } catch (...) {
      faultStash.record(std::current_exception());
    }
    return;
  default:
    return;
  }
}

void RpcClient::Impl::handleControlFrame(const std::string &text) {
  json resp;
  try {
    resp = json::parse(text);
  } catch (const std::exception &e) {
    // Unparseable text frame: nothing useful we can do.
    return;
  }
  auto typeIt = resp.find("type");
  if (typeIt == resp.end() || !typeIt->is_string())
    return;
  std::string type = typeIt->get<std::string>();

  // Unsolicited server-initiated error. The server sends one of these before
  // closing the WS (e.g. `server_busy`). Stash the message so the imminent
  // Close event surfaces this human-readable reason instead of just the raw
  // WebSocket close reason.
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
    // Unknown control-frame type. Per the protocol, receivers MUST ignore
    // unrecognised fields/types so the wire format can evolve additively
    // under the same `protocol_version`, but warn so genuine misbehavior
    // doesn't go silently dropped.
    logger.debug("cosim",
                 "Ignoring control frame with unknown type \"" + type + "\"");
    return;
  }
  auto idIt = resp.find("request_id");
  if (idIt == resp.end())
    return;
  uint64_t requestId = idIt->get<uint64_t>();

  Completion done;
  {
    std::lock_guard<std::mutex> lock(pendingMutex);
    auto it = pending.find(requestId);
    if (it == pending.end())
      return;
    done = std::move(it->second);
    // Erase under the lock so a concurrent `failAllPending` won't also
    // complete the same request.
    pending.erase(it);
  }

  if (auto errIt = resp.find("error"); errIt != resp.end()) {
    std::string code = errIt->value("code", std::string("internal"));
    std::string message = errIt->value("message", std::string());
    done(nullptr, std::make_exception_ptr(std::runtime_error(
                      "Server error [" + code + "]: " + message)));
  } else if (auto resultIt = resp.find("result"); resultIt != resp.end()) {
    done(&*resultIt, nullptr);
  } else {
    // Per the wire spec, a `type=response` frame MUST carry either
    // `result` (success) or `error` (failure). Surface the protocol
    // violation to the caller instead of pretending it succeeded.
    done(nullptr,
         std::make_exception_ptr(std::runtime_error(
             "Server response missing both \"result\" and \"error\" fields "
             "(protocol violation) for request_id=" +
             std::to_string(requestId))));
  }
}

void RpcClient::Impl::handleBinaryFrame(const std::string &data) {
  uint64_t channelId;
  const uint8_t *payloadBytes;
  size_t payloadSize;
  if (!::esi::cosim::parseDataFrame(data, channelId, payloadBytes, payloadSize))
    return;

  std::shared_ptr<ReadCallbackEntry> entry;
  {
    std::lock_guard<std::mutex> lock(readCallbacksMutex);
    auto it = readCallbacks.find(channelId);
    if (it == readCallbacks.end()) {
      // Per the protocol, the server MUST stop sending frames for a channel
      // after the unsubscribe-ack. A frame for an unknown channel is
      // therefore either a server bug or a misrouted message: warn rather
      // than silently dropping so the issue is visible.
      logger.warning("cosim", "Dropping inbound frame for channel id " +
                                  std::to_string(channelId) +
                                  ": no active subscriber");
      return;
    }
    entry = it->second;
  }

  // Push the payload into the per-channel queue and return immediately;
  // never block IX's network thread on the user-supplied callback. The
  // queue's notifier rings the transport doorbell, and the dedicated
  // transport thread is what actually invokes the callback.
  entry->queue.push(MessageData(payloadBytes, payloadSize));
}

void RpcClient::Impl::failAllPending(const std::string &reason) {
  std::unordered_map<uint64_t, Completion> failed;
  {
    std::lock_guard<std::mutex> lock(pendingMutex);
    connectionLost.store(true);
    failed.swap(pending);
  }
  auto error = std::make_exception_ptr(std::runtime_error(reason));
  for (auto &[id, done] : failed)
    done(nullptr, error);
}

//===----------------------------------------------------------------------===//
// RpcClient::Impl - request/response, sends
//===----------------------------------------------------------------------===//

void RpcClient::Impl::callAsync(const std::string &method, json params,
                                Completion done) {
  uint64_t requestId = nextRequestId.fetch_add(1);
  {
    std::lock_guard<std::mutex> lock(pendingMutex);
    // Checked under the lock `failAllPending` sets it under, so a request
    // can't slip in after the connection's pending requests were failed.
    if (connectionLost.load()) {
      done(nullptr, std::make_exception_ptr(std::runtime_error(
                        "RpcClient: connection to cosim server lost")));
      return;
    }
    pending.emplace(requestId, std::move(done));
  }

  json req;
  req["type"] = "request";
  req["request_id"] = requestId;
  req["method"] = method;
  req["params"] = std::move(params);
  std::string text = req.dump();

  // No send-side mutex needed: IXWebSocket's `sendUtf8Text` is internally
  // serialized with `sendBinary`, and every cosim frame is a single send
  // call (no header+body splits).
  auto info = ws.sendUtf8Text(text);
  if (!info.success) {
    Completion failed;
    {
      std::lock_guard<std::mutex> plock(pendingMutex);
      auto it = pending.find(requestId);
      if (it == pending.end())
        return; // Already failed by `failAllPending`.
      failed = std::move(it->second);
      pending.erase(it);
    }
    failed(nullptr, std::make_exception_ptr(std::runtime_error(
                        "RpcClient: failed to send " + method)));
  }
}

json RpcClient::Impl::call(const std::string &method, json params,
                           std::optional<std::chrono::seconds> timeout) {
  faultStash.check();
  auto promise = std::make_shared<std::promise<json>>();
  std::future<json> future = promise->get_future();
  callAsync(method, std::move(params),
            [promise](const json *result, std::exception_ptr error) {
              if (error)
                promise->set_exception(error);
              else
                promise->set_value(*result);
            });

  // Bounded wait so callers cannot hang forever on a missing/lost response.
  // On WS close or error, `failAllPending()` fires and the future becomes
  // ready immediately. The timeout only matters if the server stays
  // connected but never replies (server bug, lost control frame). A late
  // response then completes the abandoned promise harmlessly.
  if (!timeout)
    future.wait();
  else if (future.wait_for(*timeout) != std::future_status::ready)
    throw std::runtime_error("RpcClient: timed out waiting for response to " +
                             method);
  return future.get();
}

const RpcClient::Impl::ChannelMeta &
RpcClient::Impl::getToServerChannel(const std::string &channelName) const {
  auto it = channelsByName.find(channelName);
  if (it == channelsByName.end())
    throw std::runtime_error("Unknown channel '" + channelName + "'");
  if (it->second.direction != RpcClient::ChannelDirection::ToServer)
    throw std::runtime_error("Channel '" + channelName +
                             "' is not a to-server channel");
  return it->second;
}

void RpcClient::Impl::sendFrame(const ChannelMeta &meta,
                                const MessageData &data) {
  std::string frame =
      esi::cosim::buildDataFrame(meta.id, data.getBytes(), data.getSize());
  // IXWebSocket serializes sends internally; no extra mutex needed.
  auto info = ws.sendBinary(frame);
  if (!info.success)
    throw std::runtime_error("Failed to send data on channel '" + meta.name +
                             "'");
}

void RpcClient::Impl::connectToServer(
    const std::string &channelName, RpcClient::CompletionCallback onConnected) {
  faultStash.check();
  const ChannelMeta &meta = getToServerChannel(channelName);
  // The server answers only once the simulation side connects its port,
  // which may legitimately take arbitrarily long. A lost connection fails
  // the request.
  callAsync("connect", {{"channel_id", meta.id}},
            [onConnected = std::move(onConnected)](const json *,
                                                   std::exception_ptr error) {
              if (onConnected)
                onConnected(error);
            });
}

void RpcClient::Impl::writeToServer(const std::string &channelName,
                                    const MessageData &data) {
  faultStash.check();
  sendFrame(getToServerChannel(channelName), data);
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
RpcClient::Impl::connectClientReceiver(
    const std::string &channelName, RpcClient::ReadCallback callback,
    RpcClient::CompletionCallback onSubscribed) {
  faultStash.check();
  auto it = channelsByName.find(channelName);
  if (it == channelsByName.end())
    throw std::runtime_error("Unknown channel '" + channelName + "'");
  if (it->second.direction != RpcClient::ChannelDirection::ToClient)
    throw std::runtime_error("Channel '" + channelName +
                             "' is not a to-client channel");
  uint64_t channelId = it->second.id;

  // Register the callback first so any racing inbound binary frame after the
  // server's subscribe-ack has somewhere to land.
  auto entry = std::make_shared<ReadCallbackEntry>(
      channelId, [this, channelId] { readyIds.markReady(channelId); });
  entry->callback = std::move(callback);
  {
    std::lock_guard<std::mutex> lock(readCallbacksMutex);
    readCallbacks[channelId] = entry;
  }

  // Don't wait for the reply: the server answers only once the simulation
  // side of the channel is connected, which may take arbitrarily long.
  std::weak_ptr<ReadCallbackEntry> weakEntry = entry;
  callAsync("subscribe", {{"channel_id", channelId}},
            [this, channelId, weakEntry,
             onSubscribed = std::move(onSubscribed)](const json *,
                                                     std::exception_ptr error) {
              if (error) {
                // Unregister our entry, unless it was already replaced or
                // removed (e.g. the connection was disconnected).
                std::lock_guard<std::mutex> lock(readCallbacksMutex);
                auto it = readCallbacks.find(channelId);
                if (it != readCallbacks.end() && it->second == weakEntry.lock())
                  readCallbacks.erase(it);
              }
              if (onSubscribed)
                onSubscribed(error);
            });

  return std::make_unique<ReadChannelConnectionImpl>(this, channelId);
}

void RpcClient::Impl::unsubscribe(uint64_t channelId) {
  if (disconnecting)
    return;
  // Skip the RPC if the WS is already closed (server tore down first, network
  // drop, etc.). The unsubscribe is moot anyway, and otherwise every
  // per-channel cleanup during teardown would generate a noisy "failed to
  // send unsubscribe" warning from ReadChannelConnectionImpl::disconnect.
  if (ws.getReadyState() != ix::ReadyState::Open)
    return;
  call("unsubscribe", {{"channel_id", channelId}});
}

void RpcClient::Impl::unregisterCallback(uint64_t channelId) {
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
  // iteration already copied out, so it bails out promptly and drops any
  // remaining queued frames.
  entry->canceled.store(true);
  // Nudge the transport thread so it observes the cancelation promptly and
  // drains+drops whatever's still queued.
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
  std::unordered_set<uint64_t> retry;

  while (true) {
    std::unordered_set<uint64_t> ids;
    if (!readyIds.waitDrain(
            ids, retry.empty() ? std::optional<std::chrono::milliseconds>{}
                               : std::chrono::milliseconds(1)))
      return;
    ids.insert(retry.begin(), retry.end());
    retry.clear();

    for (uint64_t id : ids) {
      std::shared_ptr<ReadCallbackEntry> entry;
      {
        std::lock_guard<std::mutex> lock(readCallbacksMutex);
        auto it = readCallbacks.find(id);
        if (it == readCallbacks.end())
          continue;
        entry = it->second;
      }

      if (entry->canceled.load() || disconnecting)
        continue;

      // Best-effort drain. `TSQueue::pop(callback)` peeks at the front and
      // only pops if our callback returns true; on `false` we leave the
      // message at the head and mark the channel for retry.
      bool stuck = false;
      while (!stuck && !entry->canceled.load() && !disconnecting) {
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

void RpcClient::connectToServer(const std::string &channelName,
                                CompletionCallback onConnected) {
  impl->connectToServer(channelName, std::move(onConnected));
}

void RpcClient::writeToServer(const std::string &channelName,
                              const MessageData &data) {
  impl->writeToServer(channelName, data);
}

std::unique_ptr<RpcClient::ReadChannelConnection>
RpcClient::connectClientReceiver(const std::string &channelName,
                                 ReadCallback callback,
                                 CompletionCallback onSubscribed) {
  return impl->connectClientReceiver(channelName, std::move(callback),
                                     std::move(onSubscribed));
}
