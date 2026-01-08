//===- RpcClient.cpp - ESI Cosim RPC client implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT
// (lib/dialect/ESI/runtime/cpp/lib/backends/RpcClient.cpp).
//
//===----------------------------------------------------------------------===//

#include "esi/backends/RpcClient.h"
#include "esi/Utils.h"

#include "cosim.grpc.pb.h"

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <condition_variable>
#include <mutex>
#include <thread>

using namespace esi;
using namespace esi::backends::cosim;

using grpc::ClientContext;
using grpc::Status;

static void checkStatus(Status s, const std::string &msg) {
  if (!s.ok())
    throw std::runtime_error(msg + ". Code " + to_string(s.error_code()) +
                             ": " + s.error_message() + " (" +
                             s.error_details() + ")");
}

//===----------------------------------------------------------------------===//
// ReadChannelConnectionImpl - gRPC streaming reader implementation
//===----------------------------------------------------------------------===//

namespace {
class ReadChannelConnectionImpl
    : public RpcClient::ReadChannelConnection,
      public grpc::ClientReadReactor<::esi::cosim::Message> {
public:
  ReadChannelConnectionImpl(::esi::cosim::ChannelServer::Stub *stub,
                            const ::esi::cosim::ChannelDesc &desc,
                            RpcClient::ReadCallback callback)
      : stub(stub), grpcDesc(desc), callback(std::move(callback)),
        context(nullptr), done(false) {}

  ~ReadChannelConnectionImpl() override { disconnect(); }

  void start() {
    context = new ClientContext();
    stub->async()->ConnectToClientChannel(context, &grpcDesc, this);
    StartCall();
    StartRead(&incomingMessage);
  }

  void OnReadDone(bool ok) override {
    if (!ok)
      // This happens when we are disconnecting since we are canceling the call.
      return;

    // Read the delivered message and push it onto the queue.
    const std::string &messageString = incomingMessage.data();
    MessageData data(reinterpret_cast<const uint8_t *>(messageString.data()),
                     messageString.size());

    while (!callback(data))
      // Blocking here could cause deadlocks in specific situations.
      // TODO: Implement a way to handle this better.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Initiate the next read.
    StartRead(&incomingMessage);
  }

  // Called by gRPC when the RPC is fully complete (after cancel or error).
  void OnDone(const grpc::Status & /*status*/) override {
    std::lock_guard<std::mutex> lock(doneMutex);
    done = true;
    doneCV.notify_all();
  }

  void disconnect() override {
    if (!context)
      return;

    // Initiate cancellation.
    context->TryCancel();

    // Wait for gRPC to signal completion via OnDone().
    std::unique_lock<std::mutex> lock(doneMutex);
    doneCV.wait(lock, [this]() { return done; });

    // Now it's safe to clean up.
    delete context;
    context = nullptr;
  }

private:
  ::esi::cosim::ChannelServer::Stub *stub;
  ::esi::cosim::ChannelDesc grpcDesc;
  RpcClient::ReadCallback callback;
  ClientContext *context;
  ::esi::cosim::Message incomingMessage;

  // Synchronization for waiting on gRPC completion.
  std::mutex doneMutex;
  std::condition_variable doneCV;
  bool done;
};
} // namespace

//===----------------------------------------------------------------------===//
// RpcClient::Impl - internal implementation class
//===----------------------------------------------------------------------===//

class RpcClient::Impl {
public:
  Impl(const std::string &hostname, uint16_t port) {
    auto channel = grpc::CreateChannel(hostname + ":" + std::to_string(port),
                                       grpc::InsecureChannelCredentials());
    stub = ::esi::cosim::ChannelServer::NewStub(channel);
  }

  ::esi::cosim::ChannelServer::Stub *getStub() const { return stub.get(); }

  ::esi::cosim::Manifest getManifest() const {
    ::esi::cosim::Manifest response;
    // To get around the a race condition where the manifest may not be set yet,
    // loop until it is. TODO: fix this with the DPI API change.
    do {
      ClientContext context;
      ::esi::cosim::VoidMessage arg;
      Status s = stub->GetManifest(&context, arg, &response);
      checkStatus(s, "Failed to get manifest");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (response.esi_version() < 0);
    return response;
  }

  bool getChannelDesc(const std::string &channelName,
                      ::esi::cosim::ChannelDesc &desc) const {
    ClientContext context;
    ::esi::cosim::VoidMessage arg;
    ::esi::cosim::ListOfChannels response;
    Status s = stub->ListChannels(&context, arg, &response);
    checkStatus(s, "Failed to list channels");
    for (const auto &channel : response.channels())
      if (channel.name() == channelName) {
        desc = channel;
        return true;
      }
    return false;
  }

  std::vector<RpcClient::ChannelDesc> listChannels() const {
    ClientContext context;
    ::esi::cosim::VoidMessage arg;
    ::esi::cosim::ListOfChannels response;
    Status s = stub->ListChannels(&context, arg, &response);
    checkStatus(s, "Failed to list channels");

    std::vector<RpcClient::ChannelDesc> result;
    result.reserve(response.channels_size());
    for (const auto &grpcDesc : response.channels()) {
      RpcClient::ChannelDesc desc;
      desc.name = grpcDesc.name();
      desc.type = grpcDesc.type();
      if (grpcDesc.dir() ==
          ::esi::cosim::ChannelDesc::Direction::ChannelDesc_Direction_TO_SERVER)
        desc.dir = RpcClient::ChannelDirection::ToServer;
      else
        desc.dir = RpcClient::ChannelDirection::ToClient;
      result.push_back(std::move(desc));
    }
    return result;
  }

  void writeToServer(const std::string &channelName, const MessageData &data) {
    ClientContext context;
    ::esi::cosim::AddressedMessage grpcMsg;
    grpcMsg.set_channel_name(channelName);
    grpcMsg.mutable_message()->set_data(data.getBytes(), data.getSize());
    ::esi::cosim::VoidMessage response;
    grpc::Status sendStatus = stub->SendToServer(&context, grpcMsg, &response);
    if (!sendStatus.ok())
      throw std::runtime_error("Failed to write to channel '" + channelName +
                               "': " + std::to_string(sendStatus.error_code()) +
                               " " + sendStatus.error_message() +
                               ". Details: " + sendStatus.error_details());
  }

  std::unique_ptr<RpcClient::ReadChannelConnection>
  connectClientReceiver(const std::string &channelName,
                        RpcClient::ReadCallback callback) {
    ::esi::cosim::ChannelDesc grpcDesc;
    if (!getChannelDesc(channelName, grpcDesc))
      throw std::runtime_error("Could not find channel '" + channelName + "'");

    auto connection = std::make_unique<ReadChannelConnectionImpl>(
        stub.get(), grpcDesc, std::move(callback));
    connection->start();
    return connection;
  }

private:
  std::unique_ptr<::esi::cosim::ChannelServer::Stub> stub;
};

//===----------------------------------------------------------------------===//
// RpcClient
//===----------------------------------------------------------------------===//

RpcClient::RpcClient(const std::string &hostname, uint16_t port)
    : impl(std::make_unique<Impl>(hostname, port)) {}

RpcClient::~RpcClient() = default;

uint32_t RpcClient::getEsiVersion() const {
  return impl->getManifest().esi_version();
}

std::vector<uint8_t> RpcClient::getCompressedManifest() const {
  ::esi::cosim::Manifest response = impl->getManifest();
  std::string compressedManifestStr = response.compressed_manifest();
  return std::vector<uint8_t>(compressedManifestStr.begin(),
                              compressedManifestStr.end());
}

bool RpcClient::getChannelDesc(const std::string &channelName,
                               ChannelDesc &desc) const {
  ::esi::cosim::ChannelDesc grpcDesc;
  if (!impl->getChannelDesc(channelName, grpcDesc))
    return false;

  desc.name = grpcDesc.name();
  desc.type = grpcDesc.type();
  if (grpcDesc.dir() ==
      ::esi::cosim::ChannelDesc::Direction::ChannelDesc_Direction_TO_SERVER)
    desc.dir = ChannelDirection::ToServer;
  else
    desc.dir = ChannelDirection::ToClient;
  return true;
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
