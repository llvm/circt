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

using namespace std;

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

static void checkStatus(Status s, string msg) {
  if (!s.ok())
    throw runtime_error(msg + ". Code " + to_string(s.error_code()) + ": " +
                        s.error_message() + " (" + s.error_details() + ")");
}

struct esi::backends::cosim::CosimAccelerator::StubContainer {
  StubContainer(std::unique_ptr<ChannelServer::Stub> stub)
      : stub(std::move(stub)) {}
  std::unique_ptr<ChannelServer::Stub> stub;
};

/// Parse the connection string and instantiate the accelerator. Support the
/// traditional 'host:port' syntax and a path to 'cosim.cfg' which is output by
/// the cosimulation when it starts (which is useful when it chooses its own
/// port).
unique_ptr<AcceleratorConnection>
CosimAccelerator::connect(Context &ctxt, string connectionString) {
  string portStr;
  string host = "localhost";

  size_t colon;
  if ((colon = connectionString.find(':')) != string::npos) {
    portStr = connectionString.substr(colon + 1);
    host = connectionString.substr(0, colon);
  } else if (connectionString.ends_with("cosim.cfg")) {
    ifstream cfg(connectionString);
    string line, key, value;

    while (getline(cfg, line))
      if ((colon = line.find(":")) != string::npos) {
        key = line.substr(0, colon);
        value = line.substr(colon + 1);
        if (key == "port")
          portStr = value;
        else if (key == "host")
          host = value;
      }

    if (portStr.size() == 0)
      throw runtime_error("port line not found in file");
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
      throw runtime_error("ESI_COSIM_PORT environment variable not set");
  } else {
    throw runtime_error("Invalid connection string '" + connectionString + "'");
  }
  uint16_t port = stoul(portStr);
  return make_unique<CosimAccelerator>(ctxt, host, port);
}

/// Construct and connect to a cosim server.
CosimAccelerator::CosimAccelerator(Context &ctxt, string hostname,
                                   uint16_t port)
    : AcceleratorConnection(ctxt) {
  // Connect to the simulation.
  auto channel = grpc::CreateChannel(hostname + ":" + to_string(port),
                                     grpc::InsecureChannelCredentials());
  rpcClient = std::make_unique<StubContainer>(ChannelServer::NewStub(channel));
}
CosimAccelerator::~CosimAccelerator() { channels.clear(); }

// TODO: Fix MMIO!
// namespace {
// class CosimMMIO : public MMIO {
// public:
//   CosimMMIO(esi::cosim::LowLevel *lowLevel) : lowLevel(lowLevel) {}

//   // Push the read request into the LowLevel capnp bridge and wait for the
//   // response.
//   uint32_t read(uint32_t addr) const override {
//     lowLevel->readReqs.push(addr);

//     std::optional<std::pair<uint64_t, uint8_t>> resp;
//     while (resp = lowLevel->readResps.pop(), !resp.has_value())
//       std::this_thread::sleep_for(std::chrono::microseconds(10));
//     if (resp->second != 0)
//       throw runtime_error("MMIO read error" + to_string(resp->second));
//     return resp->first;
//   }

//   // Push the write request into the LowLevel capnp bridge and wait for the
//   ack
//   // or error.
//   void write(uint32_t addr, uint32_t data) override {
//     lowLevel->writeReqs.push(make_pair(addr, data));

//     std::optional<uint8_t> resp;
//     while (resp = lowLevel->writeResps.pop(), !resp.has_value())
//       std::this_thread::sleep_for(std::chrono::microseconds(10));
//     if (*resp != 0)
//       throw runtime_error("MMIO write error" + to_string(*resp));
//   }

// private:
//   esi::cosim::LowLevel *lowLevel;
// };
// } // namespace

namespace {
class CosimSysInfo : public SysInfo {
public:
  CosimSysInfo(
      const std::unique_ptr<esi::cosim::ChannelServer::Stub> &rpcClient)
      : rpcClient(rpcClient) {}

  uint32_t getEsiVersion() const override {
    ::esi::cosim::Manifest response = getManifest();
    return response.esi_version();
  }

  vector<uint8_t> getCompressedManifest() const override {
    ::esi::cosim::Manifest response = getManifest();
    std::string compressedManifestStr = response.compressed_manifest();
    return std::vector<uint8_t>(compressedManifestStr.begin(),
                                compressedManifestStr.end());
  }

private:
  ::esi::cosim::Manifest getManifest() const {
    ::esi::cosim::Manifest response;
    do {
      ClientContext context;
      VoidMessage arg;
      Status s = rpcClient->GetManifest(&context, arg, &response);
      checkStatus(s, "Failed to get manifest");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (response.esi_version() < 0);
    return response;
  }

  const std::unique_ptr<esi::cosim::ChannelServer::Stub> &rpcClient;
};
} // namespace

namespace {
class WriteCosimChannelPort : public WriteChannelPort {
public:
  WriteCosimChannelPort(ChannelServer::Stub *rpcClient, const ChannelDesc &desc,
                        const Type *type, string name)
      : WriteChannelPort(type), rpcClient(rpcClient), desc(desc), name(name) {}
  ~WriteCosimChannelPort() = default;

  void connect() override {
    WriteChannelPort::connect();
    if (desc.type() != getType()->getID())
      throw runtime_error("Channel '" + name + "' has wrong type. Expected " +
                          getType()->getID() + ", got " + desc.type());
    if (desc.dir() != ChannelDesc::Direction::ChannelDesc_Direction_TO_SERVER)
      throw runtime_error("Channel '" + name + "' is not a to server channel");
    assert(desc.name() == name);
  }

  void write(const MessageData &data) override {
    ClientContext context;
    AddressedMessage msg;
    msg.set_channel_name(name);
    msg.mutable_message()->set_data(data.getBytes(), data.getSize());
    VoidMessage response;
    grpc::Status sendStatus = rpcClient->SendToServer(&context, msg, &response);
    if (!sendStatus.ok())
      throw runtime_error("Failed to write to channel '" + name +
                          "': " + sendStatus.error_message() +
                          ". Details: " + sendStatus.error_details());
  }

protected:
  ChannelServer::Stub *rpcClient;
  ChannelDesc desc;
  string name;
};
} // namespace

namespace {
class ReadCosimChannelPort
    : public ReadChannelPort,
      public grpc::ClientReadReactor<esi::cosim::Message> {
public:
  ReadCosimChannelPort(ChannelServer::Stub *rpcClient, const ChannelDesc &desc,
                       const Type *type, string name)
      : ReadChannelPort(type), rpcClient(rpcClient), desc(desc), name(name),
        context(nullptr) {}
  virtual ~ReadCosimChannelPort() { disconnect(); }

  virtual void connect() override {
    if (desc.type() != getType()->getID())
      throw runtime_error("Channel '" + name + "' has wrong type. Expected " +
                          getType()->getID() + ", got " + desc.type());
    if (desc.dir() != ChannelDesc::Direction::ChannelDesc_Direction_TO_CLIENT)
      throw runtime_error("Channel '" + name + "' is not a to server channel");
    assert(desc.name() == name);

    context = new ClientContext();
    rpcClient->async()->ConnectToClientChannel(context, &desc, this);
    StartCall();
    StartRead(&incomingMessage);
  }
  void OnReadDone(bool ok) override {
    if (!ok)
      return;
    const std::string &messageString = incomingMessage.data();
    MessageData data(reinterpret_cast<const uint8_t *>(messageString.data()),
                     messageString.size());
    messageQueue.push(data);
    StartRead(&incomingMessage);
  }
  void disconnect() override {
    if (!context)
      return;
    context->TryCancel();
    delete context;
    context = nullptr;
  }
  bool read(MessageData &) override;

protected:
  ChannelServer::Stub *rpcClient;
  ChannelDesc desc;
  string name;

  ClientContext *context;
  esi::cosim::Message incomingMessage;
  esi::utils::TSQueue<MessageData> messageQueue;
};

} // namespace

bool ReadCosimChannelPort::read(MessageData &data) {
  std::optional<MessageData> msg = messageQueue.pop();
  if (!msg.has_value())
    return false;
  data = *msg;
  return true;
}

map<string, ChannelPort &>
CosimAccelerator::requestChannelsFor(AppIDPath idPath,
                                     const BundleType *bundleType) {
  map<string, ChannelPort &> channelResults;

  // Find the client details for the port at 'fullPath'.
  auto f = clientChannelAssignments.find(idPath);
  if (f == clientChannelAssignments.end())
    return channelResults;
  const map<string, string> &channelAssignments = f->second;

  // Each channel in a bundle has a separate cosim endpoint. Find them all.
  for (auto [name, dir, type] : bundleType->getChannels()) {
    auto f = channelAssignments.find(name);
    if (f == channelAssignments.end())
      throw runtime_error("Could not find channel assignment for '" +
                          idPath.toStr() + "." + name + "'");
    string channelName = f->second;

    // Get the endpoint, which may or may not exist. Construct the port.
    // Everything is validated when the client calls 'connect()' on the port.
    ChannelDesc chDesc;
    if (!getChannelDesc(channelName, chDesc))
      throw runtime_error("Could not find channel '" + channelName +
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

bool CosimAccelerator::getChannelDesc(const string &channelName,
                                      ChannelDesc &desc) {
  ClientContext context;
  VoidMessage arg;
  ListOfChannels response;
  Status s = rpcClient->stub->ListChannels(&context, arg, &response);
  checkStatus(s, "Failed to list channels");
  for (const auto &channel : response.channels())
    if (channel.name() == channelName) {
      desc = channel;
      return true;
    }
  return false;
}

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
      map<string, string> channelAssignments;
      for (auto assignment : any_cast<map<string, any>>(
               client.implOptions.at("channel_assignments")))
        channelAssignments[assignment.first] =
            any_cast<string>(assignment.second);
      clientChannelAssignments[fullClientPath] = std::move(channelAssignments);
    }
  }

  if (svcType == typeid(services::MMIO)) {
    // return new CosimMMIO(rpcClient->getLowLevel());
  } else if (svcType == typeid(SysInfo)) {
    switch (manifestMethod) {
    case ManifestMethod::Cosim:
      return new CosimSysInfo(rpcClient->stub);
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
