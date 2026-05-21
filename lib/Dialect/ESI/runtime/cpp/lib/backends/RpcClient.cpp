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

  void writeToServer(const std::string &channelName, const MessageData &data);

  std::unique_ptr<RpcClient::ReadChannelConnection>
  connectClientReceiver(const std::string &channelName,
                        RpcClient::ReadCallback callback);

  // Helpers used by ReadChannelConnectionImpl.
  void unsubscribe(uint16_t channelId);
  void unregisterCallback(uint16_t channelId);

private:
  struct ChannelMeta {
    uint16_t id;
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
  //
  // The promise is held by `shared_ptr` so that `handleControlFrame` can
  // copy out a strong reference under `pendingMutex`, release the mutex,
  // and then fulfill the promise without racing with `failAllPending()`
  // (which can erase the map entry from a parallel shutdown/close path).
  struct PendingRequest {
    std::shared_ptr<std::promise<json>> promise;
  };
  /// Issue a JSON-RPC style request and synchronously await the response.
  /// Throws on error (transport or server-reported).
  json call(const std::string &method, json params);

  std::string host;
  Logger &logger;
  ix::WebSocket ws;

  // Manifest + channel table cached from the `hello` response.
  uint32_t esiVersion = 0;
  std::vector<uint8_t> manifest;
  std::unordered_map<std::string, ChannelMeta> channelsByName;
  std::unordered_map<uint16_t, ChannelMeta> channelsById;

  // Pending control-plane requests, keyed by id.
  std::mutex pendingMutex;
  std::atomic<uint64_t> nextRequestId{1};
  std::unordered_map<uint64_t, PendingRequest> pending;

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
    uint16_t channelId;
    utils::TSQueue<MessageData> queue;

    ReadCallbackEntry(uint16_t channelId, std::function<void()> notifier)
        : channelId(channelId), queue(std::move(notifier)) {}
  };
  std::mutex readCallbacksMutex;
  std::unordered_map<uint16_t, std::shared_ptr<ReadCallbackEntry>>
      readCallbacks;

  // Dirty-set doorbell for the transport thread. Each entry's TSQueue
  // notifier inserts the channel id into `readyIds` and wakes the consumer.
  utils::ReadyIdSet<uint16_t> readyIds;
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
  uint16_t channelId;
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

  ws.setUrl("ws://" + host + ":" + std::to_string(port) + "/esi/cosim/v3");
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
  json result = call("hello", {{"client_protocol_version", 3}});

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
    meta.id = c.at("channel_id").get<uint16_t>();
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

  std::shared_ptr<std::promise<json>> promise;
  {
    std::lock_guard<std::mutex> lock(pendingMutex);
    auto it = pending.find(requestId);
    if (it == pending.end())
      return;
    promise = it->second.promise;
    // Erase under the lock so a concurrent `failAllPending` won't also
    // try to fulfill the same promise.
    pending.erase(it);
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
      // Per the wire spec, a `type=response` frame MUST carry either
      // `result` (success) or `error` (failure). Surface the protocol
      // violation to the caller instead of pretending it succeeded.
      promise->set_exception(std::make_exception_ptr(std::runtime_error(
          "Server response missing both \"result\" and \"error\" fields "
          "(protocol violation) for request_id=" +
          std::to_string(requestId))));
    }
  } catch (const std::exception &e) {
    // Promise might be already satisfied if we raced with a shutdown that
    // already fulfilled it via failAllPending. Benign, but log so unexpected
    // duplicate responses from the server don't go entirely unnoticed.
    logger.debug("cosim",
                 std::string("failed to fulfill promise for request_id=") +
                     std::to_string(requestId) + ": " + e.what());
  }
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
  entry->queue.push(MessageData(
      reinterpret_cast<const uint8_t *>(data.data() + 2), data.size() - 2));
}

void RpcClient::Impl::failAllPending(const std::string &reason) {
  std::lock_guard<std::mutex> lock(pendingMutex);
  for (auto &[id, pr] : pending) {
    try {
      pr.promise->set_exception(
          std::make_exception_ptr(std::runtime_error(reason)));
    } catch (const std::exception &e) {
      // Already-satisfied promise; benign but log it in case a real bug is
      // racing with shutdown.
      logger.debug("cosim", std::string("failAllPending: request_id=") +
                                std::to_string(id) +
                                " already satisfied: " + e.what());
    }
  }
  pending.clear();
}

//===----------------------------------------------------------------------===//
// RpcClient::Impl - request/response, sends
//===----------------------------------------------------------------------===//

json RpcClient::Impl::call(const std::string &method, json params) {
  faultStash.check();
  uint64_t requestId = nextRequestId.fetch_add(1);
  std::future<json> future;
  {
    std::lock_guard<std::mutex> lock(pendingMutex);
    auto &entry = pending[requestId];
    entry.promise = std::make_shared<std::promise<json>>();
    future = entry.promise->get_future();
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
    std::lock_guard<std::mutex> plock(pendingMutex);
    pending.erase(requestId);
    throw std::runtime_error("RpcClient: failed to send " + method);
  }

  // Bounded wait so callers cannot hang forever on a missing/lost response.
  // On WS close or error, `failAllPending()` fires and the future becomes
  // ready immediately. The timeout only matters if the server stays
  // connected but never replies (server bug, lost control frame).
  constexpr auto kCallTimeout = std::chrono::seconds(30);
  if (future.wait_for(kCallTimeout) != std::future_status::ready) {
    // Drop the pending entry so a late response doesn't fulfill a
    // destroyed promise (with shared_ptr it's still safe, but it would
    // otherwise leak the slot forever).
    std::lock_guard<std::mutex> plock(pendingMutex);
    pending.erase(requestId);
    throw std::runtime_error("RpcClient: timed out waiting for response to " +
                             method);
  }
  return future.get();
}

void RpcClient::Impl::writeToServer(const std::string &channelName,
                                    const MessageData &data) {
  faultStash.check();
  auto it = channelsByName.find(channelName);
  if (it == channelsByName.end())
    throw std::runtime_error("Unknown channel '" + channelName + "'");
  if (it->second.direction != RpcClient::ChannelDirection::ToServer)
    throw std::runtime_error("Channel '" + channelName +
                             "' is not a to-server channel");

  std::string frame = esi::cosim::buildDataFrame(it->second.id, data.getBytes(),
                                                 data.getSize());
  // IXWebSocket serializes sends internally; no extra mutex needed.
  auto info = ws.sendBinary(frame);
  if (!info.success)
    throw std::runtime_error("Failed to send data on channel '" + channelName +
                             "'");
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
  faultStash.check();
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

void RpcClient::writeToServer(const std::string &channelName,
                              const MessageData &data) {
  impl->writeToServer(channelName, data);
}

std::unique_ptr<RpcClient::ReadChannelConnection>
RpcClient::connectClientReceiver(const std::string &channelName,
                                 ReadCallback callback) {
  return impl->connectClientReceiver(channelName, std::move(callback));
}
