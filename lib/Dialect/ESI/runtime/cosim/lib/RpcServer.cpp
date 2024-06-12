//===- RpcServer.cpp - Run a cosim server ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "esi/cosim/RpcServer.h"
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
using grpc::ServerBuilder;
using grpc::Status;

/// Write the port number to a file. Necessary when we allow 'EzRpcServer' to
/// select its own port. We can't use stdout/stderr because the flushing
/// semantics are undefined (as in `flush()` doesn't work on all simulators).
static void writePort(uint16_t port) {
  // "cosim.cfg" since we may want to include other info in the future.
  FILE *fd = fopen("cosim.cfg", "w");
  fprintf(fd, "port: %u\n", (unsigned int)port);
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

  void setManifest(int esiVersion, std::vector<uint8_t> compressedManifest) {
    this->compressedManifest = std::move(compressedManifest);
    this->esiVersion = esiVersion;
  }

  ReadChannelPort &registerReadPort(const std::string &name,
                                    const std::string &type);
  WriteChannelPort &registerWritePort(const std::string &name,
                                      const std::string &type);

  void stop();

  //===--------------------------------------------------------------------===//
  // RPC API implementations.
  //===--------------------------------------------------------------------===//

  grpc::ServerUnaryReactor *GetManifest(CallbackServerContext *context,
                                        const VoidMessage *,
                                        Manifest *response) override {
    printf("GetManifest\n");
    fflush(stdout);
    response->set_esi_version(esiVersion);
    response->set_compressed_manifest(compressedManifest.data(),
                                      compressedManifest.size());
    auto reactor = context->DefaultReactor();
    reactor->Finish(Status::OK);
    return reactor;
  }

  grpc::ServerUnaryReactor *ListChannels(CallbackServerContext *,
                                         const VoidMessage *,
                                         ListOfChannels *channelsOut) override;

  grpc::ServerWriteReactor<esi::cosim::Message> *
  ConnectToClientChannel(CallbackServerContext *context,
                         const ChannelDesc *request) override;
  grpc::ServerUnaryReactor *
  SendToServer(CallbackServerContext *context,
               const esi::cosim::AddressedMessage *request,
               esi::cosim::VoidMessage *response) override;

private:
  int esiVersion;
  std::vector<uint8_t> compressedManifest;
  std::map<std::string, RpcServerReadPort *> readPorts;
  std::map<std::string, RpcServerWritePort *> writePorts;

  std::unique_ptr<Server> server;
};
using Impl = esi::cosim::RpcServer::Impl;

RpcServer::~RpcServer() {
  if (impl)
    delete impl;
}

void RpcServer::setManifest(int esiVersion,
                            std::vector<uint8_t> compressedManifest) {
  impl->setManifest(esiVersion, std::move(compressedManifest));
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

namespace {
class RpcServerReadPort : public ReadChannelPort {
public:
  RpcServerReadPort(Type *type) : ReadChannelPort(type) {}

  bool read(MessageData &data) override {
    std::optional<MessageData> msg = readQueue.pop();
    if (!msg)
      return false;
    data = std::move(*msg);
    return true;
  }
  void gotMessage(MessageData &data) { readQueue.push(std::move(data)); }

private:
  utils::TSQueue<MessageData> readQueue;
};

class RpcServerWritePort : public WriteChannelPort {
public:
  RpcServerWritePort(Type *type) : WriteChannelPort(type) {}

  void write(const MessageData &data) override {
    writeQueue.push(data);
    printf("pushed message\n");
  }

  utils::TSQueue<MessageData> writeQueue;
};

class RpcServerWriteReactor
    : public grpc::ServerWriteReactor<esi::cosim::Message> {
public:
  RpcServerWriteReactor(RpcServerWritePort *writePort)
      : writePort(writePort), sentSuccessfully(false), shutdown(false) {
    myThread = std::thread(&RpcServerWriteReactor::threadLoop, this);
  }
  void OnDone() override { delete this; }
  void OnWriteDone(bool ok) override {
    printf("on write done\n");
    std::scoped_lock<std::mutex> lock(msgMutex);
    sentSuccessfully = ok;
    sentSuccessfullyCV.notify_one();
  }
  void OnCancel() override {
    printf("on cancel\n");
    std::scoped_lock<std::mutex> lock(msgMutex);
    sentSuccessfully = false;
    sentSuccessfullyCV.notify_one();
  }

  void threadLoop();

  RpcServerWritePort *writePort;
  std::thread myThread;

  std::mutex msgMutex;
  esi::cosim::Message msg;
  std::condition_variable sentSuccessfullyCV;
  std::atomic<bool> sentSuccessfully;
  std::atomic<bool> shutdown;
};

} // namespace

Impl::Impl(int port) : esiVersion(-1) {
  ServerBuilder builder;
  std::string server_address("127.0.0.1:" + std::to_string(port));
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(this);
  server = builder.BuildAndStart();
  if (!server)
    throw std::runtime_error("Failed to start server on " + server_address);
  writePort(port);
  std::cout << "Server listening on 127.0.0.1:" << port << std::endl;
}

Impl::~Impl() {
  for (auto &port : readPorts)
    delete port.second;
  for (auto &port : writePorts)
    delete port.second;
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

void Impl::stop() {
  server->Shutdown();
  server->Wait();
}

grpc::ServerUnaryReactor *Impl::ListChannels(CallbackServerContext *context,
                                             const VoidMessage *,
                                             ListOfChannels *channelsOut) {
  for (auto [name, port] : readPorts) {
    auto *channel = channelsOut->add_channels();
    channel->set_name(name);
    channel->set_type(port->getType()->getID());
    channel->set_dir(ChannelDesc::Direction::ChannelDesc_Direction_TO_SERVER);
  }
  for (auto [name, port] : writePorts) {
    auto *channel = channelsOut->add_channels();
    channel->set_name(name);
    channel->set_type(port->getType()->getID());
    channel->set_dir(ChannelDesc::Direction::ChannelDesc_Direction_TO_CLIENT);
  }
  auto reactor = context->DefaultReactor();
  reactor->Finish(Status::OK);
  return reactor;
}

grpc::ServerWriteReactor<esi::cosim::Message> *
Impl::ConnectToClientChannel(CallbackServerContext *context,
                             const ChannelDesc *request) {
  printf("connect to client channel\n");
  auto it = writePorts.find(request->name());
  if (it == writePorts.end()) {
    auto reactor = new RpcServerWriteReactor(nullptr);
    reactor->Finish(Status(grpc::StatusCode::NOT_FOUND, "Unknown channel"));
    return reactor;
  }
  return new RpcServerWriteReactor(it->second);
}

void RpcServerWriteReactor::threadLoop() {
  printf("thread loop\n");
  while (!shutdown) {
    // TODO: adapt this to a new notification mechanism which is forthcoming.
    if (writePort->writeQueue.empty())
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    else
      printf("queue not empty\n");

    writePort->writeQueue.pop([this](const MessageData &data) -> bool {
      printf("attempting to send message\n");
      std::unique_lock<std::mutex> lock(msgMutex);
      msg.set_data(reinterpret_cast<const char *>(data.getBytes()),
                   data.getSize());
      sentSuccessfully = false;
      StartWrite(&msg);
      sentSuccessfullyCV.wait(lock);
      bool ret = sentSuccessfully;
      lock.unlock();
      printf("pop'd message\n");
      return ret;
    });
  }
}

grpc::ServerUnaryReactor *
Impl::SendToServer(CallbackServerContext *context,
                   const esi::cosim::AddressedMessage *request,
                   esi::cosim::VoidMessage *response) {
  auto reactor = context->DefaultReactor();
  auto it = readPorts.find(request->channel_name());
  if (it == readPorts.end()) {
    reactor->Finish(Status(grpc::StatusCode::NOT_FOUND, "Unknown channel"));
    return reactor;
  }

  std::string msgDataString = request->message().data();
  MessageData data(reinterpret_cast<const uint8_t *>(msgDataString.data()),
                   msgDataString.size());
  it->second->gotMessage(data);
  reactor->Finish(Status::OK);
  return reactor;
}
