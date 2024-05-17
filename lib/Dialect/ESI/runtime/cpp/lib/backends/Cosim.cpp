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

#include "cosim/CapnpThreads.h"

#include <fstream>
#include <iostream>
#include <set>

using namespace std;

using namespace esi;
using namespace esi::services;
using namespace esi::backends::cosim;

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
  rpcClient = std::make_unique<esi::cosim::RpcClient>();
  rpcClient->run(hostname, port);
}

namespace {
class CosimMMIO : public MMIO {
public:
  CosimMMIO(esi::cosim::LowLevel *lowLevel) : lowLevel(lowLevel) {}

  // Push the read request into the LowLevel capnp bridge and wait for the
  // response.
  uint32_t read(uint32_t addr) const override {
    lowLevel->readReqs.push(addr);

    std::optional<std::pair<uint64_t, uint8_t>> resp;
    while (resp = lowLevel->readResps.pop(), !resp.has_value())
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    if (resp->second != 0)
      throw runtime_error("MMIO read error" + to_string(resp->second));
    return resp->first;
  }

  // Push the write request into the LowLevel capnp bridge and wait for the ack
  // or error.
  void write(uint32_t addr, uint32_t data) override {
    lowLevel->writeReqs.push(make_pair(addr, data));

    std::optional<uint8_t> resp;
    while (resp = lowLevel->writeResps.pop(), !resp.has_value())
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    if (*resp != 0)
      throw runtime_error("MMIO write error" + to_string(*resp));
  }

private:
  esi::cosim::LowLevel *lowLevel;
};
} // namespace

namespace {
class CosimSysInfo : public SysInfo {
public:
  CosimSysInfo(const std::unique_ptr<esi::cosim::RpcClient> &rpcClient)
      : rpcClient(rpcClient) {}

  uint32_t getEsiVersion() const override {
    unsigned int esiVersion;
    std::vector<uint8_t> compressedManifest;
    if (!rpcClient->getCompressedManifest(esiVersion, compressedManifest))
      throw runtime_error("Could not get ESI version from cosim");
    return esiVersion;
  }

  vector<uint8_t> getCompressedManifest() const override {
    unsigned int esiVersion;
    std::vector<uint8_t> compressedManifest;
    if (!rpcClient->getCompressedManifest(esiVersion, compressedManifest))
      throw runtime_error("Could not get ESI version from cosim");
    return compressedManifest;
  }

private:
  const std::unique_ptr<esi::cosim::RpcClient> &rpcClient;
};
} // namespace

namespace {
class WriteCosimChannelPort : public WriteChannelPort {
public:
  WriteCosimChannelPort(esi::cosim::Endpoint *ep, const Type *type, string name)
      : WriteChannelPort(type), ep(ep), name(name) {}
  virtual ~WriteCosimChannelPort() = default;

  // TODO: Replace this with a request to connect to the capnp thread.
  virtual void connect() override {
    if (!ep)
      throw runtime_error("Could not find channel '" + name +
                          "' in cosimulation");
    if (ep->getSendTypeId() == "")
      throw runtime_error("Channel '" + name + "' is not a read channel");
    if (ep->getSendTypeId() != getType()->getID())
      throw runtime_error("Channel '" + name + "' has wrong type. Expected " +
                          getType()->getID() + ", got " + ep->getSendTypeId());
    ep->setInUse();
  }
  virtual void disconnect() override {
    if (ep)
      ep->returnForUse();
  }
  virtual void write(const MessageData &) override;

protected:
  esi::cosim::Endpoint *ep;
  string name;
};
} // namespace

void WriteCosimChannelPort::write(const MessageData &data) {
  ep->pushMessageToSim(make_unique<esi::MessageData>(data));
}

namespace {
class ReadCosimChannelPort : public ReadChannelPort {
public:
  ReadCosimChannelPort(esi::cosim::Endpoint *ep, const Type *type, string name)
      : ReadChannelPort(type), ep(ep), name(name) {}
  virtual ~ReadCosimChannelPort() = default;

  // TODO: Replace this with a request to connect to the capnp thread.
  virtual void connect() override {
    if (!ep)
      throw runtime_error("Could not find channel '" + name +
                          "' in cosimulation");
    if (ep->getRecvTypeId() == "")
      throw runtime_error("Channel '" + name + "' is not a read channel");
    if (ep->getRecvTypeId() != getType()->getID())
      throw runtime_error("Channel '" + name + "' has wrong type. Expected " +
                          getType()->getID() + ", got " + ep->getRecvTypeId());
    ep->setInUse();
  }
  virtual void disconnect() override {
    if (ep)
      ep->returnForUse();
  }
  virtual bool read(MessageData &) override;

protected:
  esi::cosim::Endpoint *ep;
  string name;
};

} // namespace

bool ReadCosimChannelPort::read(MessageData &data) {
  esi::cosim::Endpoint::MessageDataPtr msg;
  if (!ep->getMessageToClient(msg))
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
    esi::cosim::Endpoint *ep = rpcClient->getEndpoint(channelName);
    ChannelPort *port;
    if (BundlePort::isWrite(dir))
      port = new WriteCosimChannelPort(ep, type, channelName);
    else
      port = new ReadCosimChannelPort(ep, type, channelName);
    channels.emplace(port);
    channelResults.emplace(name, *port);
  }
  return channelResults;
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
    return new CosimMMIO(rpcClient->getLowLevel());
  } else if (svcType == typeid(SysInfo)) {
    switch (manifestMethod) {
    case ManifestMethod::Cosim:
      return new CosimSysInfo(rpcClient);
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
