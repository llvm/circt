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

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <format>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
  void messageConsumed(ReadChannelPort &port);

  void stop(uint32_t timeoutMS);
  int getPort() const { return boundPort; }

  // Dirty-set doorbell for the transport thread. Public so each port's
  // TSQueue can pass `[&impl, id]{ impl.readyIds.markReady(id); }` as its
  // push-notifier; also called directly from `handleSubscribe` to flush
  // anything queued before the subscription arrived.
  utils::ReadyIdSet<uint64_t> readyIds;

  /// Log that a `size`-byte frame for channel `channelId` was dropped since
  /// the simulator-side consumer refused it.
  void logRefusedFrame(uint64_t channelId, size_t size);

private:
  // Reuse the public direction enum from the client side rather than
  // duplicating it; both ends of the cosim transport agree on the same two
  // values.
  using ChannelDirection = backends::cosim::RpcClient::ChannelDirection;
  struct ChannelInfo {
    uint64_t id;
    std::string name;
    std::string typeId;
    ChannelDirection direction;
    RpcServerReadPort *readPort = nullptr;
    RpcServerWritePort *writePort = nullptr;
  };

  // Locking rules. IXWebSocket may invoke the close callback (`onClose`)
  // synchronously from inside a send on a dead socket, on whichever thread is
  // sending. So:
  //  - Never send while holding `channelsMutex`, `sessionMutex` or a port's
  //    `connectMutex`; `onClose` takes the first two.
  //  - Sending while holding a session's `subscribedMutex` is allowed (it
  //    orders data frames against the unsubscribe-ack), so never wait for
  //    `subscribedMutex` while holding any other lock.
  //  - Senders hold a `shared_ptr` to the session, which owns the socket,
  //    so `onClose` never has to wait for them. Since `onClose` can't take
  //    `subscribedMutex` either, per-port client state is tagged with the
  //    session's generation, and `onClose` retires the generation so late
  //    updates from in-progress requests are ignored.
  //  - Within a port, only the write queue's and the doorbell's locks are
  //    taken while holding its `stateMutex`.
  struct ClientSession {
    std::shared_ptr<ix::WebSocket> ws;
    // The set of `to_client` channel ids the client subscribed to. Guarded by
    // `subscribedMutex`.
    std::mutex subscribedMutex;
    std::unordered_set<uint64_t> subscribed;
    // True once `hello` has been answered.
    std::atomic<bool> helloDone = false;
    // Set once the client has disconnected.
    std::atomic<bool> closed = false;
    // Distinguishes this session from earlier ones, so state belonging to an
    // earlier client (e.g. credits for frames it sent) isn't given to this
    // one.
    uint64_t generation = 0;
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
  std::unordered_map<uint64_t, ChannelInfo> channelById;
  std::unordered_map<std::string, uint64_t> idByName;
  uint64_t nextChannelId = 0;

  // Session state. v4 of the protocol allows a single concurrent client.
  std::mutex sessionMutex;
  std::shared_ptr<ClientSession> session;
  uint64_t nextSessionGeneration = 1;

  /// The active session, if any. The returned pointer stays valid after the
  /// client disconnects; sends on it then just fail.
  std::shared_ptr<ClientSession> getSession() {
    std::lock_guard<std::mutex> lock(sessionMutex);
    return session;
  }
  /// The active session if `ws` is its socket, else null. Requests are
  /// handled through this, so one still being handled when its client
  /// disconnects can't act on a replacement client's session.
  std::shared_ptr<ClientSession> sessionFor(const ix::WebSocket &ws) {
    std::shared_ptr<ClientSession> s = getSession();
    if (s && s->ws.get() != &ws)
      return nullptr;
    return s;
  }

  // Transport thread; drains `readyIds` and dispatches per channel id.
  std::thread transportThread;

  // The IXWebSocket server.
  std::unique_ptr<ix::WebSocketServer> server;
  int boundPort = -1;

  // Connection callbacks.
  void onClientMessage(const std::weak_ptr<ix::WebSocket> &wsWeak,
                       ix::WebSocket &ws, const ix::WebSocketMessagePtr &msg);
  void onOpen(const std::weak_ptr<ix::WebSocket> &wsWeak);
  void onClose(ix::WebSocket &ws);
  void handleBinaryFrame(uint64_t generation, const std::string &data);
  void handleControlFrame(ix::WebSocket &ws, const std::string &text);
  void handleEvent(uint64_t generation, const json &frame);
  void handleHello(ix::WebSocket &ws, uint64_t requestId, const json &params);
  void handleSubscribe(ix::WebSocket &ws, uint64_t requestId,
                       const json &params);
  void handleUnsubscribe(ix::WebSocket &ws, uint64_t requestId,
                         const json &params);
  /// Look up a channel. If it doesn't exist or doesn't have direction `dir`,
  /// reply with an error and return nullopt.
  std::optional<ChannelInfo> findChannel(ix::WebSocket &ws, uint64_t requestId,
                                         uint64_t channelId,
                                         ChannelDirection dir);
  void sendResult(ix::WebSocket &ws, uint64_t requestId, const json &result);
  void sendError(ix::WebSocket &ws, uint64_t requestId, const std::string &code,
                 const std::string &message);

  void transportLoop();
  /// Send a batched `consumed` event for any to-server messages the
  /// simulation has consumed since the last call. Transport thread only.
  void sendConsumedEvent(const std::unordered_set<uint64_t> &ids);
  /// Forget all per-channel state belonging to client session `generation`,
  /// which has disconnected.
  void retireSession(uint64_t generation);

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
/// driver. The client bounds how many messages it sends ahead of the
/// simulation using credits which this port returns via `markConsumed()`.
///
/// Frames which arrive while the port is not connected are parked here, in
/// order, and handed over when it connects. They stay in flight until the
/// simulation consumes them, so the client is backpressured rather than
/// having its frames dropped.
///
/// Each frame's credit belongs to the client session which sent it. Frames
/// outlive their session (in the base class's queue or parked here), so they
/// are tracked by session, and credits for a session which has gone are
/// dropped rather than handed to the next client: a new client starts with
/// nothing in flight.
class RpcServerReadPort : public ReadChannelPort {
public:
  RpcServerReadPort(Type *type, Impl &impl, uint64_t channelId)
      : ReadChannelPort(type), impl(impl), channelId(channelId) {}

  /// Polling-mode connect: force the internal queue to be unbounded
  /// (`bufferSize == 0`) regardless of what the caller asked for.
  void connect(const ConnectOptions &options = {}) override {
    ConnectOptions forced = options;
    forced.bufferSize = 0;
    ReadChannelPort::connect(forced);
    markConnected();
  }
  void connect(ReadCallback callback,
               const ConnectOptions &options = {}) override {
    ReadChannelPort::connect(std::move(callback), options);
    markConnected();
  }
  using ReadChannelPort::connect;

  /// Client session `generation` closed: its credits are dropped.
  void clientDisconnected(uint64_t generation) {
    std::lock_guard<std::recursive_mutex> lock(creditMutex);
    retiredGeneration = (std::max)(retiredGeneration, generation);
    if (unreportedGeneration <= generation)
      unreported = 0;
  }

  void disconnect() override {
    {
      std::lock_guard<std::mutex> lock(connectMutex);
      connected = false;
    }
    // Held across the base class's disconnect so no frame is handed over and
    // counted in `held` after its queue is torn down (see `handOver()`).
    std::lock_guard<std::recursive_mutex> lock(creditMutex);
    ReadChannelPort::disconnect();
    // The base class discarded whatever it had queued for the simulation.
    // Return those frames' credits, like the server stops counting a client's
    // unconsumed frames on `unsubscribe`.
    for (auto [generation, count] : held)
      credit(generation, count);
    held.clear();
  }

  /// Hand one inbound frame from client session `generation` to the
  /// simulation, or park it if the port isn't connected (or earlier frames
  /// are still parked). Returns `false` only if a connected callback-mode
  /// consumer refused the frame, in which case the caller should drop it and
  /// call `dropped()`.
  bool deliver(uint64_t generation, const MessageData &data) {
    {
      std::lock_guard<std::mutex> lock(connectMutex);
      if (!connected || !parked.empty()) {
        parked.emplace_back(generation, data);
        return true;
      }
    }
    if (handOver(generation, data))
      return true;
    // Lost a race with `disconnect()`: park it at the front, since it arrived
    // before anything parked since.
    std::lock_guard<std::mutex> lock(connectMutex);
    if (connected)
      return false;
    parked.emplace_front(generation, data);
    return true;
  }

  /// A frame from client session `generation` was dropped rather than handed
  /// to the simulation. Return its credit.
  void dropped(uint64_t generation) {
    std::lock_guard<std::recursive_mutex> lock(creditMutex);
    credit(generation, 1);
  }

  /// The simulation consumed the oldest frame handed to it. Return its credit
  /// to the session which sent it.
  void markConsumed() {
    std::lock_guard<std::recursive_mutex> lock(creditMutex);
    // Empty if its credit was already returned by `disconnect()`.
    if (held.empty())
      return;
    auto &[generation, count] = held.front();
    credit(generation, 1);
    if (--count == 0)
      held.pop_front();
  }

  /// Take (and zero) the credits not yet returned to client session
  /// `generation`.
  uint64_t takeConsumed(uint64_t generation) {
    std::lock_guard<std::recursive_mutex> lock(creditMutex);
    if (generation != unreportedGeneration)
      return 0;
    return std::exchange(unreported, 0);
  }

private:
  /// Pass one frame from session `generation` to the base class's callback /
  /// polling queue, counting it in `held` if that worked.
  bool handOver(uint64_t generation, const MessageData &data) {
    std::unique_ptr<SegmentedMessageData> msg =
        std::make_unique<MessageData>(data);
    // Held throughout, so `disconnect()` sees the frame either counted in
    // `held` or not handed over at all. Counted first, since a callback-mode
    // consumer may report it consumed from within the callback.
    std::lock_guard<std::recursive_mutex> lock(creditMutex);
    if (!held.empty() && held.back().first == generation)
      ++held.back().second;
    else
      held.emplace_back(generation, 1);
    bool ok = false;
    try {
      ok = invokeCallback(msg);
    } catch (...) {
      unhold();
      throw;
    }
    if (!ok)
      unhold();
    return ok;
  }

  /// Undo the count of the frame `handOver()` failed to hand over.
  void unhold() {
    if (held.empty())
      return;
    if (--held.back().second == 0)
      held.pop_back();
  }

  /// Return `count` credits to session `generation`, unless it has gone.
  /// Caller holds `creditMutex`.
  void credit(uint64_t generation, uint64_t count) {
    if (generation <= retiredGeneration || generation < unreportedGeneration)
      return;
    if (generation != unreportedGeneration) {
      unreportedGeneration = generation;
      unreported = 0;
    }
    unreported += count;
    impl.readyIds.markReady(channelId);
  }

  /// Called after the base class has finished connecting (so inbound frames
  /// will be accepted). Flushes parked frames in order.
  void markConnected() {
    std::unique_lock<std::mutex> lock(connectMutex);
    connected = true;
    // `deliver()` keeps appending to `parked` (rather than delivering
    // directly) until it's empty, which preserves ordering while we flush
    // without holding the lock across the callback.
    while (!parked.empty()) {
      auto [generation, front] = parked.front();
      lock.unlock();
      bool ok = handOver(generation, front);
      lock.lock();
      if (!ok) {
        if (!connected)
          break; // Disconnected again; leave it parked.
        // Refused by a callback-mode consumer: drop it, like a frame which
        // arrives while connected (see `deliver()`), rather than leave it
        // blocking everything after it.
        impl.logRefusedFrame(channelId, front.getSize());
        dropped(generation);
      }
      parked.pop_front();
    }
  }

  Impl &impl;
  uint64_t channelId;

  // Guards the credit state below. Also held while a frame is handed over
  // (see `handOver()`), and recursive since a callback-mode consumer may
  // report a frame consumed from within the callback.
  std::recursive_mutex creditMutex;
  // Frames handed to the base class and not yet consumed, oldest first, as
  // runs of (client session generation, count).
  std::deque<std::pair<uint64_t, uint64_t>> held;
  // Credits not yet returned to session `unreportedGeneration`.
  uint64_t unreported = 0;
  uint64_t unreportedGeneration = 0;
  // Sessions up to this one have closed; their credits are dropped.
  uint64_t retiredGeneration = 0;

  std::mutex connectMutex;
  bool connected = false;
  // Frames received while not connected, oldest first, with the generation
  // of the session which sent them.
  std::deque<std::pair<uint64_t, MessageData>> parked;
};

/// Write queue for "to client" channels. Writes go into the per-port TSQueue;
/// the queue's notifier hook rings the transport doorbell so the transport
/// thread wakes up and drains across the WebSocket. DPI threads must never
/// block on I/O, so this path is strictly non-blocking.
///
/// Flow control: the port's in-flight count is the number of messages queued
/// here plus the number sent to the client but not yet reported consumed.
/// `tryWrite()` refuses new messages while that count is at the in-flight
/// limit. The limit is set by the client on `subscribe` (0 means unbounded)
/// and reverts to the default when there is no subscriber.
class RpcServerWritePort : public WriteChannelPort {
public:
  using RpcClient = backends::cosim::RpcClient;

  RpcServerWritePort(Type *type, Impl &impl, uint64_t channelId)
      : WriteChannelPort(type), channelId(channelId), impl(impl),
        writeQueue([&impl, channelId] { impl.readyIds.markReady(channelId); }) {
  }

  uint64_t getChannelId() const { return channelId; }
  uint64_t channelId;
  utils::TSQueue<MessageData> writeQueue;

  /// Pop the next queued message for sending to client session
  /// `generation`, accounting it as sent but unacknowledged. Returns nullopt
  /// once that session is retired, so nothing is sent (and counted) for a
  /// client which has gone away.
  std::optional<MessageData> popForSend(uint64_t generation) {
    std::lock_guard<std::mutex> lock(stateMutex);
    if (generation <= retiredGeneration || generation != ownerGeneration)
      return std::nullopt;
    std::optional<MessageData> msg = writeQueue.pop();
    if (!msg)
      return std::nullopt;
    // Increment `unacked` before decrementing `queued` so a concurrent
    // `hasSpace()` never under-counts the in-flight total.
    unacked.fetch_add(1);
    queued.fetch_sub(1);
    return msg;
  }

  /// Client session `generation` reported `count` messages consumed. Ignored
  /// unless it owns the flow-control state (see `subscribed()`).
  void ack(uint64_t count, uint64_t generation) {
    std::lock_guard<std::mutex> lock(stateMutex);
    if (generation <= retiredGeneration || generation != ownerGeneration)
      return;
    // Clamp at zero, just in case.
    unacked.store(unacked.load() - (std::min)(unacked.load(), count));
    spaceCv.notify_all();
  }

  /// Client session `generation` subscribed with in-flight limit `limit`. It
  /// now owns the flow-control state; anything an earlier session had in
  /// flight is gone. Ignored if that session is already retired.
  void subscribed(uint64_t limit, uint64_t generation) {
    std::lock_guard<std::mutex> lock(stateMutex);
    if (generation <= retiredGeneration)
      return;
    if (generation != ownerGeneration) {
      ownerGeneration = generation;
      unacked.store(0);
    }
    maxInFlight.store(limit);
    spaceCv.notify_all();
  }

  /// Client session `generation` unsubscribed: messages it had not
  /// acknowledged are gone, so stop counting them.
  void unsubscribed(uint64_t generation) {
    std::lock_guard<std::mutex> lock(stateMutex);
    if (generation == ownerGeneration)
      resetFlowControlLocked();
  }

  /// Client session `generation` closed: forget its flow-control state (if a
  /// later session hasn't already taken it over) and ignore anything else
  /// which arrives on its behalf.
  void retireClient(uint64_t generation) {
    std::lock_guard<std::mutex> lock(stateMutex);
    retiredGeneration = (std::max)(retiredGeneration, generation);
    if (ownerGeneration <= generation)
      resetFlowControlLocked();
  }

  /// Wakes any `write()` blocked at the in-flight limit, which then throws.
  void disconnect() override {
    {
      std::lock_guard<std::mutex> lock(stateMutex);
      disconnected = true;
    }
    spaceCv.notify_all();
    WriteChannelPort::disconnect();
  }

protected:
  void connectImpl(const ConnectOptions &) override {
    std::lock_guard<std::mutex> lock(stateMutex);
    disconnected = false;
  }

  /// Blocks while the in-flight limit is reached. The simulation must not
  /// call this, since the client may need it to make progress before it
  /// consumes anything; it should use `tryWrite()`. Throws if the port is
  /// disconnected meanwhile.
  void writeImpl(const MessageData &data) override {
    std::unique_lock<std::mutex> lock(stateMutex);
    spaceCv.wait(lock, [&] { return disconnected || hasSpace(); });
    if (disconnected)
      throw std::runtime_error("Channel id " + std::to_string(channelId) +
                               " disconnected while writing");
    pushLocked(data);
  }
  bool tryWriteImpl(const MessageData &data) override {
    std::lock_guard<std::mutex> lock(stateMutex);
    if (!hasSpace())
      return false;
    pushLocked(data);
    return true;
  }

private:
  void pushLocked(const MessageData &data) {
    queued.fetch_add(1);
    writeQueue.push(data);
  }

  void resetFlowControlLocked() {
    unacked.store(0);
    maxInFlight.store(RpcClient::DefaultToClientMaxInFlight);
    spaceCv.notify_all();
  }

  bool hasSpace() const {
    uint64_t limit = maxInFlight.load();
    return limit == 0 || queued.load() + unacked.load() < limit;
  }

  Impl &impl;
  std::atomic<uint64_t> queued{0};
  std::atomic<uint64_t> unacked{0};
  std::atomic<uint64_t> maxInFlight{RpcClient::DefaultToClientMaxInFlight};
  // Guards the generations and `disconnected`, and orders flow-control
  // updates against them. Only the write queue's and doorbell's locks are
  // taken while holding it.
  std::mutex stateMutex;
  // Notified whenever room may have appeared, or the port disconnected.
  std::condition_variable spaceCv;
  bool disconnected = false;
  uint64_t retiredGeneration = 0;
  // The session which subscribed last, and so owns the flow-control state.
  uint64_t ownerGeneration = 0;
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

  // Use the connection callback (rather than the simpler client-message one)
  // because it hands us an owning reference to the socket, which the session
  // holds so senders on other threads can't outlive it.
  server->setOnConnectionCallback(
      [this](std::weak_ptr<ix::WebSocket> wsWeak,
             std::shared_ptr<ix::ConnectionState> /*state*/) {
        std::shared_ptr<ix::WebSocket> ws = wsWeak.lock();
        if (!ws)
          return;
        ix::WebSocket *raw = ws.get();
        ws->setOnMessageCallback(
            [this, wsWeak, raw](const ix::WebSocketMessagePtr &msg) {
              onClientMessage(wsWeak, *raw, msg);
            });
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
  uint64_t id = nextChannelId++;
  auto port = std::make_unique<RpcServerReadPort>(new Type(type), *this, id);
  RpcServerReadPort *raw = port.get();
  readPorts.emplace(name, std::move(port));

  ChannelInfo info{id, name, type, ChannelDirection::ToServer, raw, nullptr};
  channelById.emplace(id, std::move(info));
  idByName.emplace(name, id);
  return *raw;
}

void Impl::messageConsumed(ReadChannelPort &port) {
  // Only ports created by `registerReadPort` are handed out, so this cast is
  // safe for any well-behaved caller.
  static_cast<RpcServerReadPort &>(port).markConsumed();
}

WriteChannelPort &Impl::registerWritePort(const std::string &name,
                                          const std::string &type) {
  faultStash.check();
  std::lock_guard<std::mutex> lock(channelsMutex);
  uint64_t id = nextChannelId++;
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

  std::shared_ptr<ClientSession> old;
  {
    std::lock_guard<std::mutex> lock(sessionMutex);
    old.swap(session);
  }

  // Surface any fault the IX network thread caught while we were running.
  // Done last so the rest of the shutdown sequence completes regardless.
  faultStash.check();
}

//===----------------------------------------------------------------------===//
// Impl - WebSocket message dispatch
//===----------------------------------------------------------------------===//

void Impl::onClientMessage(const std::weak_ptr<ix::WebSocket> &wsWeak,
                           ix::WebSocket &ws,
                           const ix::WebSocketMessagePtr &msg) {
  switch (msg->type) {
  case ix::WebSocketMessageType::Open:
    onOpen(wsWeak);
    return;
  case ix::WebSocketMessageType::Close:
  case ix::WebSocketMessageType::Error: {
    ctxt.getLogger().debug(
        "cosim", std::format("Client disconnected: {}",
                             msg->type == ix::WebSocketMessageType::Error
                                 ? msg->errorInfo.reason
                                 : msg->closeInfo.reason));
    onClose(ws);
    return;
  }
  case ix::WebSocketMessageType::Message:
    // An exception escaping this callback would kill IX's network thread.
    // Stash the first one so the next public RpcServer method rethrows it
    // on the user's thread.
    try {
      if (msg->binary) {
        // Frames from a rejected additional client are ignored.
        if (std::shared_ptr<ClientSession> s = sessionFor(ws))
          handleBinaryFrame(s->generation, msg->str);
      } else {
        handleControlFrame(ws, msg->str);
      }
    } catch (...) {
      faultStash.record(std::current_exception());
    }
    return;
  default:
    return;
  }
}

void Impl::onOpen(const std::weak_ptr<ix::WebSocket> &wsWeak) {
  std::shared_ptr<ix::WebSocket> ws = wsWeak.lock();
  if (!ws)
    return;
  {
    std::lock_guard<std::mutex> lock(sessionMutex);
    if (!session) {
      session = std::make_shared<ClientSession>();
      session->ws = ws;
      session->generation = nextSessionGeneration++;
      ctxt.getLogger().debug("cosim", "Client connected");
      return;
    }
  }
  // Single-client model: a session is already active. Send an unsolicited
  // JSON error frame so the new client gets an actionable, application-level
  // reason, then close with 1013 ("Try Again Later") rather than 1011
  // ("Internal Error") -- the latter falsely implies a server-side bug.
  ctxt.getLogger().warning(
      "cosim", "Rejecting additional client; one already connected");
  json err;
  err["type"] = "error";
  err["error"] = {{"code", "server_busy"},
                  {"message", "cosim server allows only one client at a "
                              "time; another client is already connected"}};
  ws->sendUtf8Text(err.dump());
  // 1013 = Try Again Later (RFC 6455 §7.4).
  ws->close(1013, "cosim server busy: another client is already connected");
}

void Impl::onClose(ix::WebSocket &ws) {
  std::shared_ptr<ClientSession> old;
  {
    std::lock_guard<std::mutex> lock(sessionMutex);
    // A rejected second client closing must not tear down the active one.
    if (!session || session->ws.get() != &ws)
      return;
    old.swap(session);
  }
  old->closed.store(true);
  // Must not wait for in-progress sends: this may be running inside one (see
  // the locking rules on ClientSession). Senders hold their own reference.
  // Requests from this session still being handled are fenced off by
  // generation instead.
  retireSession(old->generation);
}

void Impl::retireSession(uint64_t generation) {
  std::lock_guard<std::mutex> lock(channelsMutex);
  for (auto &[name, port] : writePorts)
    port->retireClient(generation);
  for (auto &[name, port] : readPorts)
    port->clientDisconnected(generation);
}

void Impl::handleBinaryFrame(uint64_t generation, const std::string &data) {
  uint64_t channelId;
  const uint8_t *payloadBytes;
  size_t payloadSize;
  if (!parseDataFrame(data, channelId, payloadBytes, payloadSize)) {
    ctxt.getLogger().error("cosim",
                           "Received binary frame shorter than 8-byte header");
    return;
  }

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
                             channelId));
    return;
  }

  MessageData payload(payloadBytes, payloadSize);
  // RpcServerReadPort always has an unbounded internal queue so `deliver` is
  // non-blocking: it enqueues into `ReadChannelPort`'s internal polling buffer
  // (or parks the frame until the port connects) and returns. A `false`
  // return means a callback-mode consumer refused it; drop the frame but
  // still return the credit so the client's in-flight count doesn't leak.
  if (!port->deliver(generation, payload)) {
    logRefusedFrame(channelId, payload.getSize());
    port->dropped(generation);
  }
}

void Impl::logRefusedFrame(uint64_t channelId, size_t size) {
  ctxt.getLogger().warning(
      "cosim", std::format("Dropped {} bytes for channel id {}: refused by "
                           "the simulator-side consumer",
                           size, channelId));
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
  if (typeIt != req.end() && typeIt->is_string() &&
      typeIt->get<std::string>() == "event") {
    // Events from a rejected additional client are ignored.
    if (std::shared_ptr<ClientSession> s = sessionFor(ws))
      handleEvent(s->generation, req);
    return;
  }
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

/// Parse the `channels` list of a `consumed` event into (channel_id, count)
/// pairs. Returns false (with `err` set) on a malformed event.
static bool parseConsumedEvent(const json &frame,
                               std::vector<std::pair<uint64_t, uint64_t>> &out,
                               std::string &err) {
  auto paramsIt = frame.find("params");
  if (paramsIt == frame.end() || !paramsIt->is_object()) {
    err = "missing \"params\" object";
    return false;
  }
  auto chIt = paramsIt->find("channels");
  if (chIt == paramsIt->end() || !chIt->is_array()) {
    err = "missing \"channels\" array";
    return false;
  }
  for (const json &c : *chIt) {
    auto idIt = c.find("channel_id");
    auto countIt = c.find("count");
    if (idIt == c.end() || !idIt->is_number_unsigned() || countIt == c.end() ||
        !countIt->is_number_unsigned()) {
      err = "entries need unsigned \"channel_id\" and \"count\"";
      return false;
    }
    out.emplace_back(idIt->get<uint64_t>(), countIt->get<uint64_t>());
  }
  return true;
}

void Impl::handleEvent(uint64_t generation, const json &frame) {
  std::string event = frame.value("event", std::string());
  if (event != "consumed") {
    // Events carry no request_id, so there's nobody to reply to. Ignore
    // unknown events per the protocol's additive-evolution rule.
    ctxt.getLogger().debug("cosim",
                           "Ignoring unknown client event \"" + event + "\"");
    return;
  }
  std::vector<std::pair<uint64_t, uint64_t>> acks;
  std::string err;
  if (!parseConsumedEvent(frame, acks, err)) {
    ctxt.getLogger().warning("cosim", "Malformed consumed event: " + err);
    return;
  }
  std::lock_guard<std::mutex> lock(channelsMutex);
  for (auto [id, count] : acks) {
    auto it = channelById.find(id);
    if (it == channelById.end() || !it->second.writePort) {
      ctxt.getLogger().warning(
          "cosim",
          std::format("consumed event for unknown to-client channel id {}",
                      id));
      continue;
    }
    it->second.writePort->ack(count, generation);
  }
}

void Impl::handleHello(ix::WebSocket &ws, uint64_t requestId,
                       const json &params) {
  auto verIt = params.find("client_protocol_version");
  if (verIt == params.end() || !verIt->is_number_unsigned() ||
      verIt->get<uint64_t>() != kProtocolVersion) {
    std::string got = verIt == params.end() ? "none" : verIt->dump();
    sendError(ws, requestId, "unsupported_version",
              std::format("Server speaks cosim protocol v{}; client requested "
                          "{}. Update the ESI runtime on both sides.",
                          kProtocolVersion, got));
    return;
  }

  // Block until the manifest has been set. This replaces the gRPC-era poll
  // loop on the client side.
  {
    std::unique_lock<std::mutex> lock(manifestMutex);
    manifestReadyCV.wait(lock, [&] { return manifestReady; });
  }

  json result;
  result["protocol_version"] = kProtocolVersion;
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
    for (uint64_t i = 0; i < nextChannelId; ++i) {
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

  if (std::shared_ptr<ClientSession> s = sessionFor(ws))
    s->helloDone = true;

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
  uint64_t channelId = chIdIt->get<uint64_t>();
  uint64_t maxInFlight = backends::cosim::RpcClient::DefaultToClientMaxInFlight;
  if (auto limitIt = params.find("max_in_flight"); limitIt != params.end()) {
    if (!limitIt->is_number_unsigned()) {
      sendError(ws, requestId, "protocol_error",
                "\"max_in_flight\" must be an unsigned integer");
      return;
    }
    maxInFlight = limitIt->get<uint64_t>();
  }

  std::optional<ChannelInfo> info =
      findChannel(ws, requestId, channelId, ChannelDirection::ToClient);
  if (!info)
    return;
  std::shared_ptr<ClientSession> s = sessionFor(ws);
  if (!s) {
    sendError(ws, requestId, "internal", "No active session");
    return;
  }
  {
    std::lock_guard<std::mutex> subLock(s->subscribedMutex);
    if (s->subscribed.count(channelId)) {
      sendError(ws, requestId, "protocol_error",
                std::format("Channel id {} is already subscribed", channelId));
      return;
    }
    s->subscribed.insert(channelId);
    info->writePort->subscribed(maxInFlight, s->generation);
    // Send the subscribe-ack BEFORE waking the transport thread. IXWebSocket
    // queues sends in FIFO order on a given connection, so as long as the ack
    // is enqueued first, any data frames the transport thread emits next will
    // arrive after it. Holding `subscribedMutex` keeps the transport thread
    // from sending on this channel before the ack, too.
    sendResult(ws, requestId, json::object());
  }
  // Kick the transport thread: if the port already has queued data (typical
  // for accelerator-startup writes that landed before the client
  // subscribed), this is what flushes it.
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
  uint64_t channelId = chIdIt->get<uint64_t>();

  RpcServerWritePort *writePort = nullptr;
  {
    std::lock_guard<std::mutex> chLock(channelsMutex);
    auto it = channelById.find(channelId);
    if (it != channelById.end())
      writePort = it->second.writePort;
  }

  std::shared_ptr<ClientSession> s = sessionFor(ws);
  if (!s) {
    sendError(ws, requestId, "internal", "No active session");
    return;
  }
  // Holding `subscribedMutex` across the ack keeps the transport thread from
  // sending another data frame on this channel before it.
  {
    std::lock_guard<std::mutex> subLock(s->subscribedMutex);
    auto removed = s->subscribed.erase(channelId);
    if (!removed) {
      sendError(ws, requestId, "not_subscribed",
                std::format("Channel id {} is not subscribed", channelId));
      return;
    }
    // The client drops anything it hasn't consumed once it unsubscribes.
    // Reset under the same lock so the transport thread can't send (and
    // count) another frame on this channel in between.
    if (writePort)
      writePort->unsubscribed(s->generation);
    sendResult(ws, requestId, json::object());
  }
}

std::optional<Impl::ChannelInfo> Impl::findChannel(ix::WebSocket &ws,
                                                   uint64_t requestId,
                                                   uint64_t channelId,
                                                   ChannelDirection dir) {
  std::optional<ChannelInfo> info;
  {
    std::lock_guard<std::mutex> chLock(channelsMutex);
    auto it = channelById.find(channelId);
    if (it != channelById.end())
      info = it->second;
  }
  if (!info) {
    sendError(ws, requestId, "unknown_channel",
              std::format("No channel with id {}", channelId));
    return std::nullopt;
  }
  if (info->direction != dir) {
    sendError(ws, requestId, "wrong_direction",
              std::format("Channel id {} is not a {} channel", channelId,
                          dir == ChannelDirection::ToServer ? "to-server"
                                                            : "to-client"));
    return std::nullopt;
  }
  return info;
}

void Impl::sendResult(ix::WebSocket &ws, uint64_t requestId,
                      const json &result) {
  json resp;
  resp["type"] = "response";
  resp["request_id"] = requestId;
  resp["result"] = result;
  // IXWebSocket serializes concurrent send calls internally. Callers must
  // follow the locking rules on ClientSession.
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
// Impl - transport thread
//
// Drains `to_client` write queues onto the WebSocket and returns credits for
// consumed `to_server` messages.
//===----------------------------------------------------------------------===//

void Impl::transportLoop() {
  while (true) {
    std::unordered_set<uint64_t> ids;
    if (!readyIds.waitDrain(ids))
      return;
    sendConsumedEvent(ids);
    for (uint64_t id : ids) {
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
      // re-acquire `subscribedMutex` on every iteration so
      // `handleUnsubscribe` can interpose between frames: once it erases the
      // subscription and sends the unsubscribe-ack, the next iteration here
      // sees the channel as not subscribed and breaks, guaranteeing the ack
      // is the last `to_client` frame on that channel (as the protocol spec
      // requires). Our reference to the session keeps its socket alive
      // against a concurrent `onClose()`.
      std::shared_ptr<ClientSession> s = getSession();
      if (!s)
        continue;
      while (true) {
        if (readyIds.isShutdown())
          return;
        if (s->closed.load())
          break;
        std::lock_guard<std::mutex> subLock(s->subscribedMutex);
        if (!s->subscribed.count(id))
          break;
        std::optional<MessageData> msg = writePort->popForSend(s->generation);
        if (!msg)
          break;
        std::string frame = buildDataFrame(id, msg->getBytes(), msg->getSize());
        // IXWebSocket serializes concurrent send calls internally and queues
        // them in FIFO order on the WS.
        s->ws->sendBinary(frame);
      }
    }
  }
}

void Impl::sendConsumedEvent(const std::unordered_set<uint64_t> &ids) {
  // Credits belong to the session which sent the frames, so are only
  // returned to that one. With no client there's nobody holding any.
  std::shared_ptr<ClientSession> s = getSession();
  if (!s || !s->helloDone)
    return;
  std::vector<std::pair<uint64_t, RpcServerReadPort *>> ports;
  {
    std::lock_guard<std::mutex> chLock(channelsMutex);
    for (uint64_t id : ids) {
      auto it = channelById.find(id);
      if (it != channelById.end() && it->second.readPort)
        ports.emplace_back(id, it->second.readPort);
    }
  }
  json channels = json::array();
  for (auto [id, port] : ports)
    if (uint64_t count = port->takeConsumed(s->generation))
      channels.push_back({{"channel_id", id}, {"count", count}});
  if (channels.empty())
    return;

  json event;
  event["type"] = "event";
  event["event"] = "consumed";
  event["params"] = {{"channels", std::move(channels)}};
  s->ws->sendUtf8Text(event.dump());
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

void RpcServer::messageConsumed(ReadChannelPort &port) {
  if (!impl)
    throw std::runtime_error("Server not running");
  impl->messageConsumed(port);
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
