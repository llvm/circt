//===- RpcServer.cpp - Run a cosim server ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/backends/RpcServer.h"
#include "esi/Context.h"
#include "esi/Utils.h"

#include "cosim.grpc.pb.h"

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <format>

using namespace esi;
using namespace esi::cosim;

using grpc::CallbackServerContext;
using grpc::Server;
using grpc::ServerUnaryReactor;
using grpc::ServerWriteReactor;
using grpc::Status;
using grpc::StatusCode;

/// Write the port number to a file. Necessary when we are allowed to select our
/// own port. We can't use stdout/stderr because the flushing semantics are
/// undefined (as in `flush()` doesn't work on all simulators).
static void writePort(uint16_t port) {
  // "cosim.cfg" since we may want to include other info in the future.
  FILE *fd = fopen("cosim.cfg", "w");
  fprintf(fd, "port: %u\n", static_cast<unsigned int>(port));
  fclose(fd);
}

namespace {
class RpcServerReadPort;
class RpcServerWritePort;
} // namespace

class esi::cosim::RpcServer::Impl
    : public esi::cosim::ChannelServer::CallbackService {
public:
  Impl(Context &ctxt, int port);
  ~Impl();

  Context &getContext() { return ctxt; }

  //===--------------------------------------------------------------------===//
  // Internal API
  //===--------------------------------------------------------------------===//

  void setSysInfo(int esiVersion,
                  std::optional<GetCycleCountFunc> getCycleCount,
                  std::optional<uint64_t> coreClockFrequencyHz,
                  const std::vector<uint8_t> &compressedManifest) {
    this->compressedManifest = compressedManifest;
    this->esiVersion = esiVersion;
    this->getCycleCount = getCycleCount;
    this->coreClockFrequencyHz = coreClockFrequencyHz;
  }

  ReadChannelPort &registerReadPort(const std::string &name,
                                    const std::string &type);
  WriteChannelPort &registerWritePort(const std::string &name,
                                      const std::string &type);

  void stop(uint32_t timeoutMS = 0);

  int getPort() { return port; }

  //===--------------------------------------------------------------------===//
  // RPC API implementations. See the .proto file for the API documentation.
  //===--------------------------------------------------------------------===//

  ServerUnaryReactor *GetSysInfo(CallbackServerContext *context,
                                 const VoidMessage *,
                                 SysInfo *response) override;
  ServerUnaryReactor *ListChannels(CallbackServerContext *, const VoidMessage *,
                                   ListOfChannels *channelsOut) override;
  ServerWriteReactor<esi::cosim::Message> *
  ConnectToClientChannel(CallbackServerContext *context,
                         const ChannelDesc *request) override;
  ServerUnaryReactor *SendToServer(CallbackServerContext *context,
                                   const esi::cosim::AddressedMessage *request,
                                   esi::cosim::VoidMessage *response) override;

private:
  Context &ctxt;
  int esiVersion;
  std::optional<GetCycleCountFunc> getCycleCount;
  std::optional<uint64_t> coreClockFrequencyHz;
  std::vector<uint8_t> compressedManifest;
  std::map<std::string, std::unique_ptr<RpcServerReadPort>> readPorts;
  std::map<std::string, std::unique_ptr<RpcServerWritePort>> writePorts;
  int port = -1;
  std::unique_ptr<Server> server;
};
using Impl = esi::cosim::RpcServer::Impl;

//===----------------------------------------------------------------------===//
// Read and write ports
//
// Implemented as simple queues which the RPC server writes to and reads from.
//===----------------------------------------------------------------------===//

namespace {
/// Implements a simple read queue. The RPC server will push messages into this
/// as appropriate.
class RpcServerReadPort : public ReadChannelPort {
public:
  RpcServerReadPort(Type *type) : ReadChannelPort(type) {}

  /// Internal call. Push a message FROM the RPC client to the read port.
  void push(MessageData &data) {
    while (!callback(data))
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
};

/// Implements a simple write queue. The RPC server will pull messages from this
/// as appropriate. Note that this could be more performant if a callback is
/// used. This would have more complexity as when a client disconnects the
/// outstanding messages will need somewhere to be held until the next client
/// connects. For now, it's simpler to just have the server poll the queue.
class RpcServerWritePort : public WriteChannelPort {
public:
  RpcServerWritePort(Type *type) : WriteChannelPort(type) {}

  utils::TSQueue<MessageData> writeQueue;

protected:
  void writeImpl(const MessageData &data) override { writeQueue.push(data); }
  bool tryWriteImpl(const MessageData &data) override {
    writeQueue.push(data);
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// RPC server implementations
//===----------------------------------------------------------------------===//

/// Start a server on the given port. -1 means to let the OS pick a port.
Impl::Impl(Context &ctxt, int port) : ctxt(ctxt), esiVersion(-1) {
  grpc::ServerBuilder builder;
  std::string server_address("127.0.0.1:" + std::to_string(port));
  // TODO: use secure credentials. Not so bad for now since we only accept
  // connections on localhost.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(this);
  server = builder.BuildAndStart();
  if (!server)
    throw std::runtime_error("Failed to start server on " + server_address);
  writePort(port);
  this->port = port;
  ctxt.getLogger().info("cosim", "Server listening on 127.0.0.1:" +
                                     std::to_string(port));
}

void Impl::stop(uint32_t timeoutMS) {
  // Disconnect all the ports.
  for (auto &[name, port] : readPorts)
    port->disconnect();
  for (auto &[name, port] : writePorts)
    port->disconnect();

  // Shutdown the server and wait for it to finish.
  if (timeoutMS > 0)
    server->Shutdown(gpr_time_add(
        gpr_now(GPR_CLOCK_REALTIME),
        gpr_time_from_millis(static_cast<int>(timeoutMS), GPR_TIMESPAN)));
  else
    server->Shutdown();

  server->Wait();
  server = nullptr;
}

Impl::~Impl() {
  if (server)
    stop();
}

ReadChannelPort &Impl::registerReadPort(const std::string &name,
                                        const std::string &type) {
  auto port = new RpcServerReadPort(new Type(type));
  readPorts.emplace(name, port);
  return *port;
}
WriteChannelPort &Impl::registerWritePort(const std::string &name,
                                          const std::string &type) {
  auto port = new RpcServerWritePort(new Type(type));
  writePorts.emplace(name, port);
  return *port;
}

ServerUnaryReactor *Impl::GetSysInfo(CallbackServerContext *context,
                                     const VoidMessage *, SysInfo *response) {
  response->set_esi_version(esiVersion);
  if (getCycleCount)
    response->set_cycle_count((*getCycleCount)());
  if (coreClockFrequencyHz)
    response->set_core_clock_frequency_hz(*coreClockFrequencyHz);
  response->set_compressed_manifest(compressedManifest.data(),
                                    compressedManifest.size());
  ServerUnaryReactor *reactor = context->DefaultReactor();
  reactor->Finish(Status::OK);
  return reactor;
}

/// Load the list of channels into the response and fire it off.
ServerUnaryReactor *Impl::ListChannels(CallbackServerContext *context,
                                       const VoidMessage *,
                                       ListOfChannels *channelsOut) {
  for (auto &[name, port] : readPorts) {
    auto *channel = channelsOut->add_channels();
    channel->set_name(name);
    channel->set_type(port->getType()->getID());
    channel->set_dir(ChannelDesc::Direction::ChannelDesc_Direction_TO_SERVER);
  }
  for (auto &[name, port] : writePorts) {
    auto *channel = channelsOut->add_channels();
    channel->set_name(name);
    channel->set_type(port->getType()->getID());
    channel->set_dir(ChannelDesc::Direction::ChannelDesc_Direction_TO_CLIENT);
  }

  // The default reactor is basically to just finish the RPC call as if we're
  // implementing the RPC function as a blocking call.
  auto reactor = context->DefaultReactor();
  reactor->Finish(Status::OK);
  return reactor;
}

namespace {
/// When a client connects to a read port (on its end, a write port on this
/// end), construct one of these to poll the corresponding write port on this
/// side and forward the messages.
class RpcServerWriteReactor : public ServerWriteReactor<esi::cosim::Message> {
public:
  RpcServerWriteReactor(RpcServerWritePort *writePort)
      : writePort(writePort), sentSuccessfully(SendStatus::UnknownStatus),
        shutdown(false), onDoneCalled(false) {
    myThread = std::thread(&RpcServerWriteReactor::threadLoop, this);
  }

  // gRPC manages the lifecycle of this object. OnDone() is called when gRPC is
  // completely done with this reactor. We must wait for our thread to finish
  // before deleting. See:
  // https://github.com/grpc/grpc/blob/4795c5e69b25e8c767b498bea784da0ef8c96fd5/examples/cpp/route_guide/route_guide_callback_server.cc#L120
  void OnDone() override {
    // Signal shutdown and wake up any waiting threads.
    {
      std::scoped_lock<std::mutex> lock(sentMutex);
      shutdown = true;
      onDoneCalled = true;
    }
    sentSuccessfullyCV.notify_one();
    onDoneCV.notify_one();

    // Wait for the thread to finish before self-deleting.
    if (myThread.joinable())
      myThread.join();

    delete this;
  }

  void OnWriteDone(bool ok) override {
    std::scoped_lock<std::mutex> lock(sentMutex);
    sentSuccessfully = ok ? SendStatus::Success : SendStatus::Failure;
    sentSuccessfullyCV.notify_one();
  }

  void OnCancel() override {
    std::scoped_lock<std::mutex> lock(sentMutex);
    shutdown = true;
    sentSuccessfully = SendStatus::Disconnect;
    sentSuccessfullyCV.notify_one();
  }

private:
  /// The polling loop.
  void threadLoop();
  /// The polling thread.
  std::thread myThread;

  /// Assoicated write port on this side. (Read port on the client side.)
  RpcServerWritePort *writePort;

  /// Mutex to protect the sentSuccessfully flag and shutdown state.
  std::mutex sentMutex;
  enum SendStatus { UnknownStatus, Success, Failure, Disconnect };
  volatile SendStatus sentSuccessfully;
  std::condition_variable sentSuccessfullyCV;

  std::atomic<bool> shutdown;

  /// Condition variable to wait for OnDone to be called.
  bool onDoneCalled;
  std::condition_variable onDoneCV;
};

} // namespace

void RpcServerWriteReactor::threadLoop() {
  while (!shutdown && sentSuccessfully != SendStatus::Disconnect) {
    // TODO: adapt this to a new notification mechanism which is forthcoming.
    if (!writePort || writePort->writeQueue.empty()) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
      continue;
    }

    // This lambda will get called with the message at the front of the queue.
    // If the send is successful, return true to pop it. We don't know, however,
    // if the message was sent successfully in this thread. It's only when the
    // `OnWriteDone` method is called by gRPC that we know. Use locking and
    // condition variables to orchestrate this confirmation.
    writePort->writeQueue.pop([this](const MessageData &data) -> bool {
      if (shutdown)
        return false;

      esi::cosim::Message msg;
      msg.set_data(reinterpret_cast<const char *>(data.getBytes()),
                   data.getSize());

      // Get a lock, reset the flag, start sending the message, and wait for the
      // write to complete or fail. Be mindful of the shutdown flag.
      std::unique_lock<std::mutex> lock(sentMutex);
      sentSuccessfully = SendStatus::UnknownStatus;
      StartWrite(&msg);
      sentSuccessfullyCV.wait(lock, [&]() {
        return shutdown || sentSuccessfully != SendStatus::UnknownStatus;
      });
      bool ret = sentSuccessfully == SendStatus::Success;
      lock.unlock();
      return ret;
    });
  }

  // Call Finish to signal gRPC that we're done. gRPC will then call OnDone().
  Finish(Status::OK);
}

/// When a client sends a message to a read port (write port on this end), start
/// streaming messages until the client calls uncle and requests a cancellation.
ServerWriteReactor<esi::cosim::Message> *
Impl::ConnectToClientChannel(CallbackServerContext *context,
                             const ChannelDesc *request) {
  getContext().getLogger().debug("cosim", "connect to client channel");
  auto it = writePorts.find(request->name());
  if (it == writePorts.end()) {
    auto reactor = new RpcServerWriteReactor(nullptr);
    reactor->Finish(Status(StatusCode::NOT_FOUND, "Unknown channel"));
    return reactor;
  }
  return new RpcServerWriteReactor(it->second.get());
}

/// When a client sends a message to a write port (a read port on this end),
/// simply locate the associated port, and write that message into its queue.
ServerUnaryReactor *
Impl::SendToServer(CallbackServerContext *context,
                   const esi::cosim::AddressedMessage *request,
                   esi::cosim::VoidMessage *response) {
  auto reactor = context->DefaultReactor();
  auto it = readPorts.find(request->channel_name());
  if (it == readPorts.end()) {
    reactor->Finish(Status(StatusCode::NOT_FOUND, "Unknown channel"));
    return reactor;
  }

  std::string msgDataString = request->message().data();
  MessageData data(reinterpret_cast<const uint8_t *>(msgDataString.data()),
                   msgDataString.size());
  try {
    ctxt.getLogger().debug(
        "cosim",
        std::format("Channel '{}': Received message; pushing data to read port",
                    request->channel_name()));
    it->second->push(data);
  } catch (const std::exception &e) {
    ctxt.getLogger().error(
        "cosim",
        std::format("Channel '{}': Error pushing message to read port: {}",
                    request->channel_name(), e.what()));
    reactor->Finish(
        Status(StatusCode::INTERNAL, "Error pushing message to port"));
    return reactor;
  }

  reactor->Finish(Status::OK);
  return reactor;
}

//===----------------------------------------------------------------------===//
// RpcServer pass throughs to the actual implementations above.
//===----------------------------------------------------------------------===//
RpcServer::RpcServer(Context &ctxt) : ctxt(ctxt) {}
RpcServer::~RpcServer() = default;

void RpcServer::setSysInfo(int esiVersion,
                           std::optional<GetCycleCountFunc> getCycleCount,
                           std::optional<uint64_t> coreClockFrequencyHz,
                           const std::vector<uint8_t> &compressedManifest) {
  if (!impl)
    throw std::runtime_error("Server not running");

  impl->setSysInfo(esiVersion, getCycleCount, coreClockFrequencyHz,
                   compressedManifest);
}

ReadChannelPort &RpcServer::registerReadPort(const std::string &name,
                                             const std::string &type) {
  if (!impl)
    throw std::runtime_error("Server not running");
  return impl->registerReadPort(name, type);
}

WriteChannelPort &RpcServer::registerWritePort(const std::string &name,
                                               const std::string &type) {
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
