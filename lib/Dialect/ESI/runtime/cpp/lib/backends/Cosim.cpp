//===- Cosim.cpp - Connection to ESI simulation via capnp RPC -------------===//
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
// (lib/dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp).
//
//===----------------------------------------------------------------------===//

#include "esi/backends/Cosim.h"
#include "esi/Services.h"
#include "esi/Utils.h"

#include "cosim.grpc.pb.h"

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <fstream>
#include <iostream>
#include <set>

using namespace esi;
using namespace esi::cosim;
using namespace esi::services;
using namespace esi::backends::cosim;

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;

static void checkStatus(Status s, const std::string &msg) {
  if (!s.ok())
    throw std::runtime_error(msg + ". Code " + to_string(s.error_code()) +
                             ": " + s.error_message() + " (" +
                             s.error_details() + ")");
}

/// Hack around C++ not having a way to forward declare a nested class.
struct esi::backends::cosim::CosimAccelerator::StubContainer {
  StubContainer(std::unique_ptr<ChannelServer::Stub> stub)
      : stub(std::move(stub)) {}
  std::unique_ptr<ChannelServer::Stub> stub;

  /// Get the type ID for a channel name.
  bool getChannelDesc(const std::string &channelName,
                      esi::cosim::ChannelDesc &desc);
};
using StubContainer = esi::backends::cosim::CosimAccelerator::StubContainer;

/// Parse the connection std::string and instantiate the accelerator. Support
/// the traditional 'host:port' syntax and a path to 'cosim.cfg' which is output
/// by the cosimulation when it starts (which is useful when it chooses its own
/// port).
std::unique_ptr<AcceleratorConnection>
CosimAccelerator::connect(Context &ctxt, std::string connectionString) {
  std::string portStr;
  std::string host = "localhost";

  size_t colon;
  if ((colon = connectionString.find(':')) != std::string::npos) {
    portStr = connectionString.substr(colon + 1);
    host = connectionString.substr(0, colon);
  } else if (connectionString.ends_with("cosim.cfg")) {
    std::ifstream cfg(connectionString);
    std::string line, key, value;

    while (getline(cfg, line))
      if ((colon = line.find(":")) != std::string::npos) {
        key = line.substr(0, colon);
        value = line.substr(colon + 1);
        if (key == "port")
          portStr = value;
        else if (key == "host")
          host = value;
      }

    if (portStr.size() == 0)
      throw std::runtime_error("port line not found in file");
  } else if (connectionString == "env") {
    char *hostEnv = getenv("ESI_COSIM_HOST");
    if (hostEnv)
      host = hostEnv;
    else
      host = "localhost";
    char *portEnv = getenv("ESI_COSIM_PORT");
    if (portEnv)
      portStr = portEnv;
    else
      throw std::runtime_error("ESI_COSIM_PORT environment variable not set");
  } else {
    throw std::runtime_error("Invalid connection std::string '" +
                             connectionString + "'");
  }
  uint16_t port = stoul(portStr);
  auto conn = make_unique<CosimAccelerator>(ctxt, host, port);

  // Using the MMIO manifest method is really only for internal debugging, so it
  // doesn't need to be part of the connection string.
  char *manifestMethod = getenv("ESI_COSIM_MANIFEST_MMIO");
  if (manifestMethod != nullptr)
    conn->setManifestMethod(ManifestMethod::MMIO);

  return conn;
}

/// Construct and connect to a cosim server.
CosimAccelerator::CosimAccelerator(Context &ctxt, std::string hostname,
                                   uint16_t port)
    : AcceleratorConnection(ctxt) {
  // Connect to the simulation.
  auto channel = grpc::CreateChannel(hostname + ":" + std::to_string(port),
                                     grpc::InsecureChannelCredentials());
  rpcClient = new StubContainer(ChannelServer::NewStub(channel));
}
CosimAccelerator::~CosimAccelerator() {
  if (rpcClient)
    delete rpcClient;
  channels.clear();
}

namespace {
class CosimSysInfo : public SysInfo {
public:
  CosimSysInfo(ChannelServer::Stub *rpcClient) : rpcClient(rpcClient) {}

  uint32_t getEsiVersion() const override {
    ::esi::cosim::Manifest response = getManifest();
    return response.esi_version();
  }

  std::vector<uint8_t> getCompressedManifest() const override {
    ::esi::cosim::Manifest response = getManifest();
    std::string compressedManifestStr = response.compressed_manifest();
    return std::vector<uint8_t>(compressedManifestStr.begin(),
                                compressedManifestStr.end());
  }

private:
  ::esi::cosim::Manifest getManifest() const {
    ::esi::cosim::Manifest response;
    // To get around the a race condition where the manifest may not be set yet,
    // loop until it is. TODO: fix this with the DPI API change.
    do {
      ClientContext context;
      VoidMessage arg;
      Status s = rpcClient->GetManifest(&context, arg, &response);
      checkStatus(s, "Failed to get manifest");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (response.esi_version() < 0);
    return response;
  }

  esi::cosim::ChannelServer::Stub *rpcClient;
};
} // namespace

namespace {
/// Cosim client implementation of a write channel port.
class WriteCosimChannelPort : public WriteChannelPort {
public:
  WriteCosimChannelPort(ChannelServer::Stub *rpcClient, const ChannelDesc &desc,
                        const Type *type, std::string name)
      : WriteChannelPort(type), rpcClient(rpcClient), desc(desc), name(name) {}
  ~WriteCosimChannelPort() = default;

  void connectImpl() override {
    if (desc.type() != getType()->getID())
      throw std::runtime_error("Channel '" + name +
                               "' has wrong type. Expected " +
                               getType()->getID() + ", got " + desc.type());
    if (desc.dir() != ChannelDesc::Direction::ChannelDesc_Direction_TO_SERVER)
      throw std::runtime_error("Channel '" + name +
                               "' is not a to server channel");
    assert(desc.name() == name);
  }

  /// Send a write message to the server.
  void write(const MessageData &data) override {
    ClientContext context;
    AddressedMessage msg;
    msg.set_channel_name(name);
    msg.mutable_message()->set_data(data.getBytes(), data.getSize());
    VoidMessage response;
    grpc::Status sendStatus = rpcClient->SendToServer(&context, msg, &response);
    if (!sendStatus.ok())
      throw std::runtime_error("Failed to write to channel '" + name +
                               "': " + std::to_string(sendStatus.error_code()) +
                               " " + sendStatus.error_message() +
                               ". Details: " + sendStatus.error_details());
  }

protected:
  ChannelServer::Stub *rpcClient;
  /// The channel description as provided by the server.
  ChannelDesc desc;
  /// The name of the channel from the manifest.
  std::string name;
};
} // namespace

namespace {
/// Cosim client implementation of a read channel port. Since gRPC read protocol
/// streams messages back, this implementation is quite complex.
class ReadCosimChannelPort
    : public ReadChannelPort,
      public grpc::ClientReadReactor<esi::cosim::Message> {
public:
  ReadCosimChannelPort(ChannelServer::Stub *rpcClient, const ChannelDesc &desc,
                       const Type *type, std::string name)
      : ReadChannelPort(type), rpcClient(rpcClient), desc(desc), name(name),
        context(nullptr) {}
  virtual ~ReadCosimChannelPort() { disconnect(); }

  void connectImpl() override {
    // Sanity checking.
    if (desc.type() != getType()->getID())
      throw std::runtime_error("Channel '" + name +
                               "' has wrong type. Expected " +
                               getType()->getID() + ", got " + desc.type());
    if (desc.dir() != ChannelDesc::Direction::ChannelDesc_Direction_TO_CLIENT)
      throw std::runtime_error("Channel '" + name +
                               "' is not a to server channel");
    assert(desc.name() == name);

    // Initiate a stream of messages from the server.
    context = std::make_unique<ClientContext>();
    rpcClient->async()->ConnectToClientChannel(context.get(), &desc, this);
    StartCall();
    StartRead(&incomingMessage);
  }

  /// Gets called when there's a new message from the server. It'll be stored in
  /// `incomingMessage`.
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

  /// Disconnect this channel from the server.
  void disconnect() override {
    if (!context)
      return;
    context->TryCancel();
    context.reset();
    ReadChannelPort::disconnect();
  }

protected:
  ChannelServer::Stub *rpcClient;
  /// The channel description as provided by the server.
  ChannelDesc desc;
  /// The name of the channel from the manifest.
  std::string name;

  std::unique_ptr<ClientContext> context;
  /// Storage location for the incoming message.
  esi::cosim::Message incomingMessage;
};

} // namespace

std::map<std::string, ChannelPort &>
CosimAccelerator::requestChannelsFor(AppIDPath idPath,
                                     const BundleType *bundleType) {
  std::map<std::string, ChannelPort &> channelResults;

  // Find the client details for the port at 'fullPath'.
  auto f = clientChannelAssignments.find(idPath);
  if (f == clientChannelAssignments.end())
    return channelResults;
  const std::map<std::string, std::string> &channelAssignments = f->second;

  // Each channel in a bundle has a separate cosim endpoint. Find them all.
  for (auto [name, dir, type] : bundleType->getChannels()) {
    auto f = channelAssignments.find(name);
    if (f == channelAssignments.end())
      throw std::runtime_error("Could not find channel assignment for '" +
                               idPath.toStr() + "." + name + "'");
    std::string channelName = f->second;

    // Get the endpoint, which may or may not exist. Construct the port.
    // Everything is validated when the client calls 'connect()' on the port.
    ChannelDesc chDesc;
    if (!rpcClient->getChannelDesc(channelName, chDesc))
      throw std::runtime_error("Could not find channel '" + channelName +
                               "' in cosimulation");

    ChannelPort *port;
    if (BundlePort::isWrite(dir)) {
      port = new WriteCosimChannelPort(rpcClient->stub.get(), chDesc, type,
                                       channelName);
    } else {
      port = new ReadCosimChannelPort(rpcClient->stub.get(), chDesc, type,
                                      channelName);
    }
    channels.emplace(port);
    channelResults.emplace(name, *port);
  }
  return channelResults;
}

/// Get the channel description for a channel name. Iterate through the list
/// each time. Since this will only be called a small number of times on a small
/// list, it's not worth doing anything fancy.
bool StubContainer::getChannelDesc(const std::string &channelName,
                                   ChannelDesc &desc) {
  ClientContext context;
  VoidMessage arg;
  ListOfChannels response;
  Status s = stub->ListChannels(&context, arg, &response);
  checkStatus(s, "Failed to list channels");
  for (const auto &channel : response.channels())
    if (channel.name() == channelName) {
      desc = channel;
      return true;
    }
  return false;
}

namespace {
class CosimMMIO : public MMIO {
public:
  CosimMMIO(Context &ctxt, StubContainer *rpcClient) {
    // We have to locate the channels ourselves since this service might be used
    // to retrieve the manifest.
    ChannelDesc readArg, readResp;
    if (!rpcClient->getChannelDesc("__cosim_mmio_read.arg", readArg) ||
        !rpcClient->getChannelDesc("__cosim_mmio_read.result", readResp))
      throw std::runtime_error("Could not find MMIO channels");

    const esi::Type *i32Type = getType(ctxt, new UIntType(readArg.type(), 32));
    const esi::Type *i64Type = getType(ctxt, new UIntType(readResp.type(), 64));

    // Get ports, create the function, then connect to it.
    readArgPort = std::make_unique<WriteCosimChannelPort>(
        rpcClient->stub.get(), readArg, i32Type, "__cosim_mmio_read.arg");
    readRespPort = std::make_unique<ReadCosimChannelPort>(
        rpcClient->stub.get(), readResp, i64Type, "__cosim_mmio_read.result");
    readMMIO.reset(FuncService::Function::get(AppID("__cosim_mmio_read"),
                                              *readArgPort, *readRespPort));
    readMMIO->connect();
  }

  // Call the read function and wait for a response.
  uint64_t read(uint32_t addr) const override {
    auto arg = MessageData::from(addr);
    std::future<MessageData> result = readMMIO->call(arg);
    result.wait();
    return *result.get().as<uint64_t>();
  }

  void write(uint32_t addr, uint64_t data) override {
    // TODO: this.
    throw std::runtime_error("Cosim MMIO write not implemented");
  }

private:
  const esi::Type *getType(Context &ctxt, esi::Type *type) {
    if (auto t = ctxt.getType(type->getID())) {
      delete type;
      return *t;
    }
    ctxt.registerType(type);
    return type;
  }
  std::unique_ptr<WriteCosimChannelPort> readArgPort;
  std::unique_ptr<ReadCosimChannelPort> readRespPort;
  std::unique_ptr<FuncService::Function> readMMIO;
};

} // namespace

Service *CosimAccelerator::createService(Service::Type svcType,
                                         AppIDPath idPath, std::string implName,
                                         const ServiceImplDetails &details,
                                         const HWClientDetails &clients) {
  // Compute our parents idPath path.
  AppIDPath prefix = std::move(idPath);
  if (prefix.size() > 0)
    prefix.pop_back();

  if (implName == "cosim") {
    // Get the channel assignments for each client.
    for (auto client : clients) {
      AppIDPath fullClientPath = prefix + client.relPath;
      std::map<std::string, std::string> channelAssignments;
      for (auto assignment : std::any_cast<std::map<std::string, std::any>>(
               client.implOptions.at("channel_assignments")))
        channelAssignments[assignment.first] =
            std::any_cast<std::string>(assignment.second);
      clientChannelAssignments[fullClientPath] = std::move(channelAssignments);
    }
  }

  if (svcType == typeid(services::MMIO)) {
    return new CosimMMIO(getCtxt(), rpcClient);
  } else if (svcType == typeid(SysInfo)) {
    switch (manifestMethod) {
    case ManifestMethod::Cosim:
      return new CosimSysInfo(rpcClient->stub.get());
    case ManifestMethod::MMIO:
      return new MMIOSysInfo(getService<services::MMIO>());
    }
  } else if (svcType == typeid(CustomService) && implName == "cosim") {
    return new CustomService(idPath, details, clients);
  }
  return nullptr;
}

void CosimAccelerator::setManifestMethod(ManifestMethod method) {
  manifestMethod = method;
}

REGISTER_ACCELERATOR("cosim", backends::cosim::CosimAccelerator);
