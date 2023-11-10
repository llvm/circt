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
#include "esi/StdServices.h"

#include "CosimDpi.capnp.h"
#include <capnp/ez-rpc.h>

#include <fstream>
#include <iostream>

using namespace std;

using namespace esi;
using namespace esi::services;
using namespace esi::backends::cosim;

/// Parse the connection string and instantiate the accelerator. Support the
/// traditional 'host:port' syntax and a path to 'cosim.cfg' which is output by
/// the cosimulation when it starts (which is useful when it chooses its own
/// port).
unique_ptr<Accelerator> CosimAccelerator::connect(string connectionString) {
  string portStr;
  string host = "localhost";

  size_t colon;
  if ((colon = connectionString.find(':')) != string::npos) {
    portStr = connectionString.substr(colon + 1);
    host = connectionString.substr(0, colon);
  } else {
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
  }
  uint16_t port = stoul(portStr);
  return make_unique<CosimAccelerator>(host, port);
}

struct esi::backends::cosim::CosimAccelerator::Impl {
  capnp::EzRpcClient rpcClient;
  kj::WaitScope &waitScope;
  CosimDpiServer::Client cosim;
  EsiLowLevel::Client lowLevel;

  Impl(string hostname, uint16_t port)
      : rpcClient(hostname, port), waitScope(rpcClient.getWaitScope()),
        cosim(rpcClient.getMain<CosimDpiServer>()), lowLevel(nullptr) {
    auto llReq = cosim.openLowLevelRequest();
    auto llPromise = llReq.send();
    lowLevel = llPromise.wait(waitScope).getLowLevel();
  }
};

/// Construct and connect to a cosim server.
// TODO: Implement this.
CosimAccelerator::CosimAccelerator(string hostname, uint16_t port) {
  impl = make_unique<Impl>(hostname, port);
}

namespace {
class CosimMMIO : public MMIO {
public:
  CosimMMIO(EsiLowLevel::Client &llClient, kj::WaitScope &waitScope)
      : llClient(llClient), waitScope(waitScope) {}

  uint64_t read(uint32_t addr) const override {
    auto req = llClient.readMMIORequest();
    req.setAddress(addr);
    return req.send().wait(waitScope).getData();
  }
  void write(uint32_t addr, uint64_t data) override {
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
class CosimCustomService : public services::CustomService {
public:
  using CustomService::CustomService;

  virtual map<string, ChannelPort &>
  requestChannelsFor(AppIDPath, const BundleType &,
                     BundlePort::Direction portDir) override {
    return {};
  }
};
} // namespace

Service *CosimAccelerator::createService(Service::Type svcType, AppIDPath id,
                                         const ServiceImplDetails &details,
                                         const HWClientDetails &clients) {
  if (svcType == typeid(MMIO))
    return new CosimMMIO(impl->lowLevel, impl->waitScope);
  else if (svcType == typeid(SysInfo))
    // return new MMIOSysInfo(getService<MMIO>());
    return new CosimSysInfo(impl->cosim, impl->waitScope);
  else if (svcType == typeid(CustomService))
    return new CosimCustomService(id, details, clients);
  return nullptr;
}

REGISTER_ACCELERATOR("cosim", backends::cosim::CosimAccelerator);
