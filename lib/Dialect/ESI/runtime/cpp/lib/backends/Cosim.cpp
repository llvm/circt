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

#include "CosimDpi.capnp.h"
#include <capnp/ez-rpc.h>

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

namespace {
class CosimChannelPort;
}

struct esi::backends::cosim::CosimAccelerator::Impl {
  capnp::EzRpcClient rpcClient;
  kj::WaitScope &waitScope;
  CosimDpiServer::Client cosim;
  EsiLowLevel::Client lowLevel;

  // We own all channels connected to rpcClient since their lifetime is tied to
  // rpcClient.
  set<unique_ptr<ChannelPort>> channels;

  // Map from client path to channel assignments for that client.
  map<AppIDPath, map<string, string>> clientChannelAssignments;

  Impl(string hostname, uint16_t port)
      : rpcClient(hostname, port), waitScope(rpcClient.getWaitScope()),
        cosim(rpcClient.getMain<CosimDpiServer>()), lowLevel(nullptr) {
    auto llReq = cosim.openLowLevelRequest();
    auto llPromise = llReq.send();
    lowLevel = llPromise.wait(waitScope).getLowLevel();
  }
  ~Impl();

  /// Request the host side channel ports for a particular instance (identified
  /// by the AppID path). For convenience, provide the bundle type and direction
  /// of the bundle port.
  std::map<std::string, ChannelPort &> requestChannelsFor(AppIDPath,
                                                          const BundleType *);
};

/// Construct and connect to a cosim server.
CosimAccelerator::CosimAccelerator(Context &ctxt, string hostname,
                                   uint16_t port)
    : AcceleratorConnection(ctxt) {
  impl = make_unique<Impl>(hostname, port);
}

namespace {
class CosimMMIO : public MMIO {
public:
  CosimMMIO(EsiLowLevel::Client &llClient, kj::WaitScope &waitScope)
      : llClient(llClient), waitScope(waitScope) {}

  uint32_t read(uint32_t addr) const override {
    auto req = llClient.readMMIORequest();
    req.setAddress(addr);
    return req.send().wait(waitScope).getData();
  }
  void write(uint32_t addr, uint32_t data) override {
    auto req = llClient.writeMMIORequest();
    req.setAddress(addr);
    req.setData(data);
    req.send().wait(waitScope);
  }

private:
  EsiLowLevel::Client &llClient;
  kj::WaitScope &waitScope;
};
} // namespace

namespace {
class CosimSysInfo : public SysInfo {
public:
  CosimSysInfo(CosimDpiServer::Client &client, kj::WaitScope &waitScope)
      : client(client), waitScope(waitScope) {}

  uint32_t getEsiVersion() const override {
    auto maniResp =
        client.getCompressedManifestRequest().send().wait(waitScope);
    return maniResp.getVersion();
  }

  vector<uint8_t> getCompressedManifest() const override {
    auto maniResp =
        client.getCompressedManifestRequest().send().wait(waitScope);
    capnp::Data::Reader data = maniResp.getCompressedManifest();
    return vector<uint8_t>(data.begin(), data.end());
  }

private:
  CosimDpiServer::Client &client;
  kj::WaitScope &waitScope;
};
} // namespace

namespace {
/// Parent class for read and write channel ports.
class CosimChannelPort {
public:
  CosimChannelPort(CosimAccelerator::Impl &impl, string name)
      : impl(impl), name(name), isConnected(false), ep(nullptr) {}
  virtual ~CosimChannelPort() {}

  void connect();
  void disconnect();
  void write(const MessageData &);
  bool read(MessageData &);

protected:
  CosimAccelerator::Impl &impl;
  string name;
  bool isConnected;
  EsiDpiEndpoint::Client ep;
};
} // namespace

void CosimChannelPort::write(const MessageData &data) {
  if (!isConnected)
    throw runtime_error("Cannot write to a channel port that is not connected");

  auto req = ep.sendFromHostRequest();
  req.setMsg(capnp::Data::Reader(data.getBytes(), data.getSize()));
  req.send().wait(impl.waitScope);
}

bool CosimChannelPort::read(MessageData &data) {
  auto req = ep.recvToHostRequest();
  auto resp = req.send().wait(impl.waitScope);
  if (!resp.getHasData())
    return false;
  capnp::Data::Reader msg = resp.getResp();
  size_t size = msg.size();
  std::vector<uint8_t> vec(size);
  memcpy(vec.data(), msg.begin(), size);
  data = MessageData(vec);
  return true;
}

esi::backends::cosim::CosimAccelerator::Impl::~Impl() {
  // Make sure all channels are disconnected before rpcClient gets deconstructed
  // or it'll throw an exception.
  for (auto &ccp : channels)
    ccp->disconnect();
}

void CosimChannelPort::connect() {
  if (isConnected)
    return;

  // Linear search through the list of cosim endpoints. Slow, but good enough as
  // connect isn't expected to be called often.
  auto listResp = impl.cosim.listRequest().send().wait(impl.waitScope);
  for (auto iface : listResp.getIfaces())
    if (iface.getEndpointID() == name) {
      auto openReq = impl.cosim.openRequest();
      openReq.setIface(iface);
      auto openResp = openReq.send().wait(impl.waitScope);
      ep = openResp.getEndpoint();
      isConnected = true;
      return;
    }
  throw runtime_error("Could not find channel '" + name + "' in cosimulation");
}

void CosimChannelPort::disconnect() {
  if (!isConnected)
    return;
  ep.closeRequest().send().wait(impl.waitScope);
  ep = nullptr;
  isConnected = false;
}

namespace {
class WriteCosimChannelPort : public WriteChannelPort {
public:
  WriteCosimChannelPort(CosimAccelerator::Impl &impl, const Type *type,
                        string name)
      : WriteChannelPort(type),
        cosim(make_unique<CosimChannelPort>(impl, name)) {}

  virtual ~WriteCosimChannelPort() = default;

  virtual void connect() override { cosim->connect(); }
  virtual void disconnect() override { cosim->disconnect(); }
  virtual void write(const MessageData &) override;

protected:
  std::unique_ptr<CosimChannelPort> cosim;
};
} // namespace

void WriteCosimChannelPort::write(const MessageData &data) {
  cosim->write(data);
}

namespace {
class ReadCosimChannelPort : public ReadChannelPort {
public:
  ReadCosimChannelPort(CosimAccelerator::Impl &impl, const Type *type,
                       string name)
      : ReadChannelPort(type), cosim(new CosimChannelPort(impl, name)) {}

  virtual ~ReadCosimChannelPort() = default;

  virtual void connect() override { cosim->connect(); }
  virtual void disconnect() override { cosim->disconnect(); }
  virtual bool read(MessageData &) override;

protected:
  std::unique_ptr<CosimChannelPort> cosim;
};

} // namespace

bool ReadCosimChannelPort::read(MessageData &data) { return cosim->read(data); }

map<string, ChannelPort &>
CosimAccelerator::Impl::requestChannelsFor(AppIDPath idPath,
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

    ChannelPort *port;
    if (BundlePort::isWrite(dir))
      port = new WriteCosimChannelPort(*this, type, channelName);
    else
      port = new ReadCosimChannelPort(*this, type, channelName);
    channels.emplace(port);
    channelResults.emplace(name, *port);
  }
  return channelResults;
}

map<string, ChannelPort &>
CosimAccelerator::requestChannelsFor(AppIDPath idPath,
                                     const BundleType *bundleType) {
  return impl->requestChannelsFor(idPath, bundleType);
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
      impl->clientChannelAssignments[fullClientPath] =
          std::move(channelAssignments);
    }
  }

  if (svcType == typeid(services::MMIO)) {
    return new CosimMMIO(impl->lowLevel, impl->waitScope);
  } else if (svcType == typeid(SysInfo)) {
    switch (manifestMethod) {
    case ManifestMethod::Cosim:
      return new CosimSysInfo(impl->cosim, impl->waitScope);
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
