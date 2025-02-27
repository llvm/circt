//===- RpcServer.cpp - Run a cosim server ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/backends/RpcServer.h"
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
  Impl(int port);
  ~Impl();

  //===--------------------------------------------------------------------===//
  // Internal API
  //===--------------------------------------------------------------------===//

  void setManifest(int esiVersion,
                   const std::vector<uint8_t> &compressedManifest) {
    this->compressedManifest = compressedManifest;
    this->esiVersion = esiVersion;
  }

  ReadChannelPort &registerReadPort(const std::string &name,
                                    const std::string &type);
  WriteChannelPort &registerWritePort(const std::string &name,
                                      const std::string &type);

  void stop();

  //===--------------------------------------------------------------------===//
  // RPC API implementations. See the .proto file for the API documentation.
  //===--------------------------------------------------------------------===//

  ServerUnaryReactor *GetManifest(CallbackServerContext *context,
                                  const VoidMessage *,
                                  Manifest *response) override;
  ServerUnaryReactor *ListChannels(CallbackServerContext *, const VoidMessage *,
                                   ListOfChannels *channelsOut) override;
  ServerWriteReactor<esi::cosim::Message> *
  ConnectToClientChannel(CallbackServerContext *context,
                         const ChannelDesc *request) override;
  ServerUnaryReactor *SendToServer(CallbackServerContext *context,
                                   const esi::cosim::AddressedMessage *request,
                                   esi::cosim::VoidMessage *response) override;

private:
  int esiVersion;
  std::vector<uint8_t> compressedManifest;
  std::map<std::string, std::unique_ptr<RpcServerReadPort>> readPorts;
  std::map<std::string, std::unique_ptr<RpcServerWritePort>> writePorts;

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
  void write(const MessageData &data) override { writeQueue.push(data); }
  bool tryWrite(const MessageData &data) override {
    writeQueue.push(data);
    return true;
  }

  utils::TSQueue<MessageData> writeQueue;
};
} // namespace

//===----------------------------------------------------------------------===//
// RPC server implementations
//===----------------------------------------------------------------------===//

/// Start a server on the given port. -1 means to let the OS pick a port.
Impl::Impl(int port) : esiVersion(-1) {
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
  std::cout << "Server listening on 127.0.0.1:" << port << std::endl;
}

void Impl::stop() {
  // Disconnect all the ports.
  for (auto &[name, port] : readPorts)
    port->disconnect();
  for (auto &[name, port] : writePorts)
    port->disconnect();

  // Shutdown the server and wait for it to finish.
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
  port->connect();
  return *port;
}
WriteChannelPort &Impl::registerWritePort(const std::string &name,
                                          const std::string &type) {
  auto port = new RpcServerWritePort(new Type(type));
  writePorts.emplace(name, port);
  port->connect();
  return *port;
}

ServerUnaryReactor *Impl::GetManifest(CallbackServerContext *context,
                                      const VoidMessage *, Manifest *response) {
  response->set_esi_version(esiVersion);
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
        shutdown(false) {
    myThread = std::thread(&RpcServerWriteReactor::threadLoop, this);
  }
  ~RpcServerWriteReactor() {
    shutdown = true;
    // Wake up the potentially sleeping thread.
    sentSuccessfullyCV.notify_one();
    if (myThread.joinable())
      myThread.join();
  }

  // Deleting 'this' from within a callback is safe since this is how gRPC tells
  // us that it's released the reference. This pattern lets gRPC manage this
  // object. (Though a shared pointer would be better.) It was actually copied
  // from one of the gRPC examples:
  // https://github.com/grpc/grpc/blob/4795c5e69b25e8c767b498bea784da0ef8c96fd5/examples/cpp/route_guide/route_guide_callback_server.cc#L120
  // The alternative is to have something else (e.g. Impl) manage this object
  // and have this method tell it that gRPC is done with it and it should be
  // deleted. As of now, there's no specific need for that and it adds
  // additional complexity. If there is at some point in the future, change
  // this.
  void OnDone() override { delete this; }
  void OnWriteDone(bool ok) override {
    std::scoped_lock<std::mutex> lock(sentMutex);
    sentSuccessfully = ok ? SendStatus::Success : SendStatus::Failure;
    sentSuccessfullyCV.notify_one();
  }
  void OnCancel() override {
    std::scoped_lock<std::mutex> lock(sentMutex);
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

  /// Mutex to protect the sentSuccessfully flag.
  std::mutex sentMutex;
  enum SendStatus { UnknownStatus, Success, Failure, Disconnect };
  volatile SendStatus sentSuccessfully;
  std::condition_variable sentSuccessfullyCV;

  std::atomic<bool> shutdown;
};

} // namespace

void RpcServerWriteReactor::threadLoop() {
  while (!shutdown && sentSuccessfully != SendStatus::Disconnect) {
    // TODO: adapt this to a new notification mechanism which is forthcoming.
    if (writePort->writeQueue.empty())
      std::this_thread::sleep_for(std::chrono::microseconds(100));

    // This lambda will get called with the message at the front of the queue.
    // If the send is successful, return true to pop it. We don't know, however,
    // if the message was sent successfully in this thread. It's only when the
    // `OnWriteDone` method is called by gRPC that we know. Use locking and
    // condition variables to orchestrate this confirmation.
    writePort->writeQueue.pop([this](const MessageData &data) -> bool {
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
  Finish(Status::OK);
}

/// When a client sends a message to a read port (write port on this end), start
/// streaming messages until the client calls uncle and requests a cancellation.
ServerWriteReactor<esi::cosim::Message> *
Impl::ConnectToClientChannel(CallbackServerContext *context,
                             const ChannelDesc *request) {
  printf("connect to client channel\n");
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
  it->second->push(data);
  reactor->Finish(Status::OK);
  return reactor;
}

//===----------------------------------------------------------------------===//
// RpcServer pass throughs to the actual implementations above.
//===----------------------------------------------------------------------===//
RpcServer::~RpcServer() {
  if (impl)
    delete impl;
}
void RpcServer::setManifest(int esiVersion,
                            const std::vector<uint8_t> &compressedManifest) {
  impl->setManifest(esiVersion, compressedManifest);
}
ReadChannelPort &RpcServer::registerReadPort(const std::string &name,
                                             const std::string &type) {
  return impl->registerReadPort(name, type);
}
WriteChannelPort &RpcServer::registerWritePort(const std::string &name,
                                               const std::string &type) {
  return impl->registerWritePort(name, type);
}
void RpcServer::run(int port) { impl = new Impl(port); }
void RpcServer::stop() {
  assert(impl && "Server not running");
  impl->stop();
}
