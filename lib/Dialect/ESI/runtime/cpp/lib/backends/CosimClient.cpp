//===- CosimClient.cpp - ESI Cosim gRPC client implementation -------------===//
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
// (lib/dialect/ESI/runtime/cpp/lib/backends/CosimClient.cpp).
//
//===----------------------------------------------------------------------===//

#include "esi/backends/CosimClient.h"
#include "esi/Utils.h"

#include "cosim.grpc.pb.h"

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <thread>

using namespace esi;
using namespace esi::backends::cosim;

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

static void checkStatus(Status s, const std::string &msg) {
  if (!s.ok())
    throw std::runtime_error(msg + ". Code " + to_string(s.error_code()) +
                             ": " + s.error_message() + " (" +
                             s.error_details() + ")");
}

//===----------------------------------------------------------------------===//
// CosimClientImpl - internal implementation class
//===----------------------------------------------------------------------===//

namespace esi {
namespace backends {
namespace cosim {

class CosimClientImpl {
public:
  CosimClientImpl(const std::string &hostname, uint16_t port) {
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

private:
  std::unique_ptr<::esi::cosim::ChannelServer::Stub> stub;
};

} // namespace cosim
} // namespace backends
} // namespace esi

//===----------------------------------------------------------------------===//
// CosimClient
//===----------------------------------------------------------------------===//

CosimClient::CosimClient(const std::string &hostname, uint16_t port)
    : impl(std::make_unique<CosimClientImpl>(hostname, port)) {}

CosimClient::~CosimClient() = default;

uint32_t CosimClient::getEsiVersion() const {
  return impl->getManifest().esi_version();
}

std::vector<uint8_t> CosimClient::getCompressedManifest() const {
  ::esi::cosim::Manifest response = impl->getManifest();
  std::string compressedManifestStr = response.compressed_manifest();
  return std::vector<uint8_t>(compressedManifestStr.begin(),
                              compressedManifestStr.end());
}

bool CosimClient::getChannelDesc(const std::string &channelName,
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

std::vector<CosimClient::ChannelDesc> CosimClient::listChannels() const {
  ClientContext context;
  ::esi::cosim::VoidMessage arg;
  ::esi::cosim::ListOfChannels response;
  Status s = impl->getStub()->ListChannels(&context, arg, &response);
  checkStatus(s, "Failed to list channels");

  std::vector<ChannelDesc> result;
  result.reserve(response.channels_size());
  for (const auto &grpcDesc : response.channels()) {
    ChannelDesc desc;
    desc.name = grpcDesc.name();
    desc.type = grpcDesc.type();
    if (grpcDesc.dir() ==
        ::esi::cosim::ChannelDesc::Direction::ChannelDesc_Direction_TO_SERVER)
      desc.dir = ChannelDirection::ToServer;
    else
      desc.dir = ChannelDirection::ToClient;
    result.push_back(std::move(desc));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// WriteCosimChannelPort::Impl
//===----------------------------------------------------------------------===//

class WriteCosimChannelPort::Impl {
public:
  Impl(AcceleratorConnection &conn, CosimClient &client,
       const CosimClient::ChannelDesc &desc, std::string name)
      : conn(conn), client(client), desc(desc), name(std::move(name)) {}

  void connectImpl() {
    if (desc.dir != CosimClient::ChannelDirection::ToServer)
      throw std::runtime_error("Channel '" + name +
                               "' is not a to server channel");
  }

  void writeImpl(const MessageData &data) {
    // Add trace logging before sending the message.
    conn.getLogger().trace(
        [this,
         &data](std::string &subsystem, std::string &msg,
                std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "cosim_write";
          msg = "Writing message to channel '" + name + "'";
          details = std::make_unique<std::map<std::string, std::any>>();
          (*details)["channel"] = name;
          (*details)["data_size"] = data.getSize();
          (*details)["message_data"] = data.toHex();
        });

    ClientContext context;
    ::esi::cosim::AddressedMessage grpcMsg;
    grpcMsg.set_channel_name(name);
    grpcMsg.mutable_message()->set_data(data.getBytes(), data.getSize());
    ::esi::cosim::VoidMessage response;
    grpc::Status sendStatus =
        client.getImpl()->getStub()->SendToServer(&context, grpcMsg, &response);
    if (!sendStatus.ok())
      throw std::runtime_error("Failed to write to channel '" + name +
                               "': " + std::to_string(sendStatus.error_code()) +
                               " " + sendStatus.error_message() +
                               ". Details: " + sendStatus.error_details());
  }

  AcceleratorConnection &conn;
  CosimClient &client;
  CosimClient::ChannelDesc desc;
  std::string name;
};

WriteCosimChannelPort::WriteCosimChannelPort(
    AcceleratorConnection &conn, CosimClient &client,
    const CosimClient::ChannelDesc &desc, const Type *type, std::string name)
    : WriteChannelPort(type),
      impl(std::make_unique<Impl>(conn, client, desc, std::move(name))) {}

WriteCosimChannelPort::~WriteCosimChannelPort() = default;

void WriteCosimChannelPort::connectImpl(
    const ChannelPort::ConnectOptions &options) {
  impl->connectImpl();
}

void WriteCosimChannelPort::writeImpl(const MessageData &data) {
  impl->writeImpl(data);
}

bool WriteCosimChannelPort::tryWriteImpl(const MessageData &data) {
  impl->writeImpl(data);
  return true;
}

//===----------------------------------------------------------------------===//
// ReadCosimChannelPort::Impl
//===----------------------------------------------------------------------===//

class ReadCosimChannelPort::Impl
    : public grpc::ClientReadReactor<::esi::cosim::Message> {
public:
  Impl(ReadCosimChannelPort &port, AcceleratorConnection &conn,
       CosimClient &client, const CosimClient::ChannelDesc &desc,
       std::string name)
      : port(port), conn(conn), client(client), desc(desc),
        name(std::move(name)), context(nullptr) {}

  ~Impl() { disconnect(); }

  void connectImpl() {
    if (desc.dir != CosimClient::ChannelDirection::ToClient)
      throw std::runtime_error("Channel '" + name +
                               "' is not a to client channel");

    // Initiate a stream of messages from the server.
    if (context)
      return;
    context = new ClientContext();

    // We need to get the raw gRPC ChannelDesc for the async call.
    ::esi::cosim::ChannelDesc grpcDesc;
    if (!client.getImpl()->getChannelDesc(name, grpcDesc))
      throw std::runtime_error("Could not find channel '" + name + "'");

    client.getImpl()->getStub()->async()->ConnectToClientChannel(
        context, &grpcDesc, this);
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

    // Add trace logging for the received message.
    conn.getLogger().trace(
        [this,
         &data](std::string &subsystem, std::string &msg,
                std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "cosim_read";
          msg = "Received message from channel '" + name + "'";
          details = std::make_unique<std::map<std::string, std::any>>();
          (*details)["channel"] = name;
          (*details)["data_size"] = data.getSize();
          (*details)["message_data"] = data.toHex();
        });

    while (!port.callback(data))
      // Blocking here could cause deadlocks in specific situations.
      // TODO: Implement a way to handle this better.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Log the message consumption.
    conn.getLogger().trace(
        [this](std::string &subsystem, std::string &msg,
               std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "cosim_read";
          msg = "Message from channel '" + name + "' consumed";
        });

    // Initiate the next read.
    StartRead(&incomingMessage);
  }

  void disconnect() {
    Logger &logger = conn.getLogger();
    logger.debug("cosim_read", "Disconnecting channel " + name);
    if (!context)
      return;
    context->TryCancel();
    // Don't delete the context since gRPC still hold a reference to it.
    // TODO: figure out how to delete it.
  }

  ReadCosimChannelPort &port;
  AcceleratorConnection &conn;
  CosimClient &client;
  CosimClient::ChannelDesc desc;
  std::string name;
  ClientContext *context;
  ::esi::cosim::Message incomingMessage;
};

ReadCosimChannelPort::ReadCosimChannelPort(AcceleratorConnection &conn,
                                           CosimClient &client,
                                           const CosimClient::ChannelDesc &desc,
                                           const Type *type, std::string name)
    : ReadChannelPort(type),
      impl(std::make_unique<Impl>(*this, conn, client, desc, std::move(name))) {
}

ReadCosimChannelPort::~ReadCosimChannelPort() = default;

void ReadCosimChannelPort::connectImpl(
    const ChannelPort::ConnectOptions &options) {
  impl->connectImpl();
}

void ReadCosimChannelPort::disconnect() {
  impl->disconnect();
  ReadChannelPort::disconnect();
}
