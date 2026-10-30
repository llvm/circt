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
  utils::ReadyIdSet<uint64_t> readyIds;

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
  //  - Ports' `stateMutex` and `ClientEndState`'s mutex are leaf locks.
  struct ClientSession {
    std::shared_ptr<ix::WebSocket> ws;
    // The set of `to_client` channel ids the client subscribed to, and
    // subscribe requests waiting for the simulator side to connect (channel
    // id -> request id). Both guarded by `subscribedMutex`.
    std::mutex subscribedMutex;
    std::unordered_set<uint64_t> subscribed;
    std::unordered_map<uint64_t, uint64_t> pendingSubscribes;
    // True once `hello` has been answered.
    std::atomic<bool> helloDone = false;
    // Set once the client has disconnected.
    std::atomic<bool> closed = false;
    // Distinguishes this session from earlier ones, so a deferred reply to an
    // earlier client's request can't reach this one.
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

public:
  /// Reply `result: {}` to request `requestId` of session `generation`, if
  /// that session is still the active one. Callable from any thread.
  void sendDeferredResult(uint64_t generation, uint64_t requestId);
  /// The simulator side of to-client channel `channelId` connected: answer
  /// the client's parked `subscribe`, if any. Callable from any thread.
  void completePendingSubscribe(uint64_t channelId);

private:
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
  void handleBinaryFrame(const std::string &data);
  void handleControlFrame(ix::WebSocket &ws, const std::string &text);
  void handleHello(ix::WebSocket &ws, uint64_t requestId, const json &params);
  void handleSubscribe(ix::WebSocket &ws, uint64_t requestId,
                       const json &params);
  void handleUnsubscribe(ix::WebSocket &ws, uint64_t requestId,
                         const json &params);
  void handleConnect(ix::WebSocket &ws, uint64_t requestId, const json &params);
  /// Look up a channel. If it doesn't exist or doesn't have direction `dir`,
  /// reply with an error and return nullopt.
  std::optional<ChannelInfo> findChannel(ix::WebSocket &ws, uint64_t requestId,
                                         uint64_t channelId,
                                         ChannelDirection dir);
  /// Make `channelId` subscribed in `s` and acknowledge `requestId`. Caller
  /// holds `s.subscribedMutex`, and must call the returned `onConnected`
  /// callbacks once it doesn't.
  [[nodiscard]] std::vector<ChannelPort::ConnectCallback>
  subscribeLocked(ClientSession &s, RpcServerWritePort &port,
                  uint64_t channelId, uint64_t requestId);
  void sendResult(ix::WebSocket &ws, uint64_t requestId, const json &result);
  void sendError(ix::WebSocket &ws, uint64_t requestId, const std::string &code,
                 const std::string &message);

  void transportLoop();
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
using ConnectCallbacks = std::vector<ChannelPort::ConnectCallback>;

/// Call every callback in `callbacks` with `error`.
void notifyAll(ConnectCallbacks callbacks, std::exception_ptr error = nullptr) {
  for (auto &cb : callbacks)
    cb(error);
}

/// Tracks whether the client's end of a channel is connected, and holds the
/// simulator side's `onConnected` callbacks until it is. A callback which has
/// fired stays fired if the client later goes away; ones registered later
/// wait for the next client.
///
/// Updates carry the client session's generation. Once a session has been
/// retired (closed), late updates from it are ignored, so a request which was
/// still being handled when its client disconnected can't mark the channel
/// connected again.
class ClientEndState {
public:
  /// Call `cb` once the client end is connected (now, if it already is).
  void whenConnected(ChannelPort::ConnectCallback cb) {
    if (!cb)
      return;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!connected) {
        waiters.push_back(std::move(cb));
        return;
      }
    }
    cb(nullptr);
  }

  /// Record whether session `generation` has the client end connected.
  /// Returns the callbacks which are now due; the caller must call them (with
  /// nullptr) once it holds no locks.
  [[nodiscard]] ConnectCallbacks set(bool isConnected, uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex);
    if (generation <= retiredGeneration)
      return {};
    connected = isConnected;
    ConnectCallbacks ready;
    if (isConnected)
      ready.swap(waiters);
    return ready;
  }

  /// Session `generation` closed.
  void retire(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex);
    // Parenthesized to avoid <windows.h>'s `max` macro.
    retiredGeneration = (std::max)(retiredGeneration, generation);
    connected = false;
  }

  /// The simulator side disconnected: fail any callbacks still waiting.
  void simDisconnected(const std::string &what) {
    ConnectCallbacks failed;
    {
      std::lock_guard<std::mutex> lock(mutex);
      failed.swap(waiters);
    }
    notifyAll(std::move(failed),
              std::make_exception_ptr(std::runtime_error(
                  what + " was disconnected before the client connected")));
  }

private:
  std::mutex mutex;
  bool connected = false;
  uint64_t retiredGeneration = 0;
  ConnectCallbacks waiters;
};

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
///
/// Frames which arrive while the port is not connected are parked here, in
/// order, and handed over when it connects, rather than dropped.
///
/// `connect()`'s `onConnected` is called once the client has connected its
/// end (sent `connect` for this channel).
class RpcServerReadPort : public ReadChannelPort {
public:
  RpcServerReadPort(Type *type, Impl &impl, uint64_t channelId)
      : ReadChannelPort(type), impl(impl), channelId(channelId) {}

  /// Polling-mode connect: force the internal queue to be unbounded
  /// (`bufferSize == 0`) regardless of what the caller asked for.
  void connect(const ConnectOptions &options = {},
               ConnectCallback onConnected = nullptr) override {
    ConnectOptions forced = options;
    forced.bufferSize = 0;
    ReadChannelPort::connect(forced, std::move(onConnected));
    markConnected();
  }
  void connect(ReadCallback callback, const ConnectOptions &options = {},
               ConnectCallback onConnected = nullptr) override {
    ReadChannelPort::connect(std::move(callback), options,
                             std::move(onConnected));
    markConnected();
  }
  using ReadChannelPort::connect;

  /// Client session `generation` closed: `onConnected` callbacks from later
  /// `connect()`s wait for the next client.
  void clientDisconnected(uint64_t generation) { clientEnd.retire(generation); }

  void disconnect() override {
    {
      std::lock_guard<std::mutex> lock(connectMutex);
      connected = false;
    }
    clientEnd.simDisconnected("Channel id " + std::to_string(channelId));
    ReadChannelPort::disconnect();
  }

  /// The client connected its end (`requestId` of session `generation`).
  /// Answer once this port is connected too; immediately if it already is.
  /// Never blocks.
  void clientConnecting(uint64_t generation, uint64_t requestId) {
    notifyAll(clientEnd.set(true, generation));
    {
      std::lock_guard<std::mutex> lock(connectMutex);
      if (!connected) {
        pendingConnects.emplace_back(generation, requestId);
        return;
      }
    }
    impl.sendDeferredResult(generation, requestId);
  }

  /// Hand one inbound frame to the simulation, or park it if the port isn't
  /// connected (or earlier frames are still parked). Returns `false` only if
  /// a connected callback-mode consumer refused the frame, in which case the
  /// caller should drop it.
  bool deliver(const MessageData &data) {
    {
      std::lock_guard<std::mutex> lock(connectMutex);
      if (!connected || !parked.empty()) {
        parked.push_back(data);
        return true;
      }
    }
    if (handOver(data))
      return true;
    // Lost a race with `disconnect()`: park it at the front, since it arrived
    // before anything parked since.
    std::lock_guard<std::mutex> lock(connectMutex);
    if (connected)
      return false;
    parked.push_front(data);
    return true;
  }

protected:
  void connectImpl(const ConnectOptions &,
                   ConnectCallback onConnected) override {
    clientEnd.whenConnected(std::move(onConnected));
  }

private:
  /// Pass one frame to the base class's callback / polling queue.
  bool handOver(const MessageData &data) {
    std::unique_ptr<SegmentedMessageData> msg =
        std::make_unique<MessageData>(data);
    return invokeCallback(msg);
  }

  /// Called after the base class has finished connecting (so inbound frames
  /// will be accepted). Flushes parked frames in order, then answers any
  /// parked `connect` requests.
  void markConnected() {
    std::unique_lock<std::mutex> lock(connectMutex);
    connected = true;
    // `deliver()` keeps appending to `parked` (rather than delivering
    // directly) until it's empty, which preserves ordering while we flush
    // without holding the lock across the callback.
    while (!parked.empty()) {
      MessageData front = parked.front();
      lock.unlock();
      bool ok = handOver(front);
      lock.lock();
      if (!ok)
        break; // Disconnected again (or refused); leave it parked.
      parked.pop_front();
    }
    std::vector<std::pair<uint64_t, uint64_t>> waiting;
    waiting.swap(pendingConnects);
    lock.unlock();
    for (auto [generation, requestId] : waiting)
      impl.sendDeferredResult(generation, requestId);
  }

  Impl &impl;
  uint64_t channelId;

  std::mutex connectMutex;
  bool connected = false;
  // Frames received while not connected, oldest first.
  std::deque<MessageData> parked;
  // Parked `connect` requests: (session generation, request id).
  std::vector<std::pair<uint64_t, uint64_t>> pendingConnects;
  ClientEndState clientEnd;
};

/// Write queue for "to client" channels. Writes go into the per-port TSQueue;
/// the queue's notifier hook rings the transport doorbell so the transport
/// thread wakes up and drains across the WebSocket. DPI threads must never
/// block on I/O, so this path is strictly non-blocking.
///
/// A client's `subscribe` is answered only once this port is connected.
/// `connect()`'s `onConnected` is called once a client has subscribed.
class RpcServerWritePort : public WriteChannelPort {
public:
  RpcServerWritePort(Type *type, Impl &impl, uint64_t channelId)
      : WriteChannelPort(type), channelId(channelId), impl(impl),
        writeQueue([&impl, channelId] { impl.readyIds.markReady(channelId); }) {
  }

  void connect(const ConnectOptions &options = {},
               ConnectCallback onConnected = nullptr) override {
    WriteChannelPort::connect(options, std::move(onConnected));
    simConnected.store(true);
    // Answer any subscribe which arrived first. Done after setting
    // `simConnected` so a concurrent `handleSubscribe` either sees it or
    // parks its request where this finds it.
    impl.completePendingSubscribe(channelId);
  }
  void disconnect() override {
    simConnected.store(false);
    clientEnd.simDisconnected("Channel id " + std::to_string(channelId));
    WriteChannelPort::disconnect();
  }

  bool isSimConnected() const { return simConnected.load(); }
  /// Record whether client session `generation` is subscribed. Called under
  /// that session's `subscribedMutex`, which orders it against other
  /// subscribes and unsubscribes. Returns the `onConnected` callbacks now
  /// due; the caller must call them once it holds no locks.
  [[nodiscard]] ConnectCallbacks setClientSubscribed(bool subscribed,
                                                     uint64_t generation) {
    return clientEnd.set(subscribed, generation);
  }

  uint64_t getChannelId() const { return channelId; }
  uint64_t channelId;
  utils::TSQueue<MessageData> writeQueue;

  /// Pop the next queued message for sending to client session
  /// `generation`. Returns nullopt once that session is retired, so nothing
  /// is sent to a client which has gone away.
  std::optional<MessageData> popForSend(uint64_t generation) {
    std::lock_guard<std::mutex> lock(stateMutex);
    if (generation <= retiredGeneration)
      return std::nullopt;
    return writeQueue.pop();
  }

  /// Client session `generation` closed: ignore anything else which arrives
  /// on its behalf.
  void retireClient(uint64_t generation) {
    {
      std::lock_guard<std::mutex> lock(stateMutex);
      retiredGeneration = (std::max)(retiredGeneration, generation);
    }
    clientEnd.retire(generation);
  }

protected:
  void connectImpl(const ConnectOptions &,
                   ConnectCallback onConnected) override {
    clientEnd.whenConnected(std::move(onConnected));
  }

  // TODO: TSQueue is unbounded so if there's no client subscibed it'll fill up
  // memory. We should add some backpressure mechanism here to avoid that.
  void writeImpl(const MessageData &data) override { writeQueue.push(data); }
  bool tryWriteImpl(const MessageData &data) override {
    writeImpl(data);
    return true;
  }

private:
  Impl &impl;
  std::atomic<bool> simConnected{false};
  ClientEndState clientEnd;
  // Guards `retiredGeneration`. A leaf lock.
  std::mutex stateMutex;
  uint64_t retiredGeneration = 0;
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

void Impl::handleBinaryFrame(const std::string &data) {
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
  // return means a callback-mode consumer refused it; drop the frame.
  if (!port->deliver(payload))
    ctxt.getLogger().warning(
        "cosim", std::format("Dropped {} bytes for channel id {}: refused by "
                             "the simulator-side consumer",
                             payload.getSize(), channelId));
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
  else if (method == "connect")
    handleConnect(ws, requestId, params);
  else
    sendError(ws, requestId, "protocol_error", "Unknown method: " + method);
}

//===----------------------------------------------------------------------===//
// Impl - control methods
//===----------------------------------------------------------------------===//

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

  if (std::shared_ptr<ClientSession> s = getSession())
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

  std::optional<ChannelInfo> info =
      findChannel(ws, requestId, channelId, ChannelDirection::ToClient);
  if (!info)
    return;
  std::shared_ptr<ClientSession> s = getSession();
  if (!s) {
    sendError(ws, requestId, "internal", "No active session");
    return;
  }
  std::vector<ChannelPort::ConnectCallback> due;
  {
    std::lock_guard<std::mutex> subLock(s->subscribedMutex);
    if (s->subscribed.count(channelId) ||
        s->pendingSubscribes.count(channelId)) {
      sendError(ws, requestId, "protocol_error",
                std::format("Channel id {} is already subscribed", channelId));
      return;
    }
    // Must not block here (this is IX's network thread). If the simulator
    // side isn't connected yet, park the request; the port's `connect()`
    // completes it. See `RpcServerWritePort::connect` for why checking under
    // `subscribedMutex` can't miss that.
    if (!info->writePort->isSimConnected()) {
      s->pendingSubscribes[channelId] = requestId;
      return;
    }
    due = subscribeLocked(*s, *info->writePort, channelId, requestId);
  }
  notifyAll(std::move(due));
  // Kick the transport thread: if the port already has queued data (typical
  // for accelerator-startup writes that landed before the client
  // subscribed), this is what flushes it.
  readyIds.markReady(channelId);
}

std::vector<ChannelPort::ConnectCallback>
Impl::subscribeLocked(ClientSession &s, RpcServerWritePort &port,
                      uint64_t channelId, uint64_t requestId) {
  s.subscribed.insert(channelId);
  auto due = port.setClientSubscribed(true, s.generation);
  // Send the subscribe-ack BEFORE waking the transport thread. IXWebSocket
  // queues sends in FIFO order on a given connection, so as long as the ack
  // is enqueued first, any data frames the transport thread emits next will
  // arrive after it. Holding `subscribedMutex` keeps the transport thread
  // from sending on this channel before the ack, too.
  sendResult(*s.ws, requestId, json::object());
  return due;
}

void Impl::completePendingSubscribe(uint64_t channelId) {
  std::shared_ptr<ClientSession> s = getSession();
  if (!s)
    return;
  RpcServerWritePort *port = nullptr;
  {
    std::lock_guard<std::mutex> chLock(channelsMutex);
    auto it = channelById.find(channelId);
    if (it != channelById.end())
      port = it->second.writePort;
  }
  if (!port)
    return;
  std::vector<ChannelPort::ConnectCallback> due;
  {
    std::lock_guard<std::mutex> subLock(s->subscribedMutex);
    auto it = s->pendingSubscribes.find(channelId);
    if (it == s->pendingSubscribes.end())
      return;
    uint64_t requestId = it->second;
    s->pendingSubscribes.erase(it);
    due = subscribeLocked(*s, *port, channelId, requestId);
  }
  notifyAll(std::move(due));
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

  std::shared_ptr<ClientSession> s = getSession();
  if (!s) {
    sendError(ws, requestId, "internal", "No active session");
    return;
  }
  // Holding `subscribedMutex` across the ack keeps the transport thread from
  // sending another data frame on this channel before it.
  {
    std::lock_guard<std::mutex> subLock(s->subscribedMutex);
    if (auto it = s->pendingSubscribes.find(channelId);
        it != s->pendingSubscribes.end()) {
      // Never completed: fail the subscribe, then acknowledge.
      sendError(ws, it->second, "canceled",
                std::format("Subscription to channel id {} canceled by "
                            "unsubscribe",
                            channelId));
      s->pendingSubscribes.erase(it);
      sendResult(ws, requestId, json::object());
      return;
    }
    auto removed = s->subscribed.erase(channelId);
    if (!removed) {
      sendError(ws, requestId, "not_subscribed",
                std::format("Channel id {} is not subscribed", channelId));
      return;
    }
    // Unsubscribing never makes callbacks due.
    if (writePort)
      (void)writePort->setClientSubscribed(false, s->generation);
    sendResult(ws, requestId, json::object());
  }
}

void Impl::handleConnect(ix::WebSocket &ws, uint64_t requestId,
                         const json &params) {
  auto chIdIt = params.find("channel_id");
  if (chIdIt == params.end() || !chIdIt->is_number_unsigned()) {
    sendError(ws, requestId, "protocol_error",
              "connect requires unsigned \"channel_id\"");
    return;
  }
  uint64_t channelId = chIdIt->get<uint64_t>();

  std::optional<ChannelInfo> info =
      findChannel(ws, requestId, channelId, ChannelDirection::ToServer);
  if (!info)
    return;
  std::shared_ptr<ClientSession> s = getSession();
  if (!s) {
    sendError(ws, requestId, "internal", "No active session");
    return;
  }
  // Must not block here: this is IX's network thread, which also carries
  // every other channel's traffic. Park the request on the port instead.
  info->readPort->clientConnecting(s->generation, requestId);
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

void Impl::sendDeferredResult(uint64_t generation, uint64_t requestId) {
  std::shared_ptr<ClientSession> s = getSession();
  if (!s || s->generation != generation)
    return;
  sendResult(*s->ws, requestId, json::object());
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
// Impl - transport thread (push model, `to_client` only)
//===----------------------------------------------------------------------===//

void Impl::transportLoop() {
  while (true) {
    std::unordered_set<uint64_t> ids;
    if (!readyIds.waitDrain(ids))
      return;
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
