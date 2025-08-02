//===- StdServices.cpp - implementations of std services ------------------===//
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

#include "esi/Services.h"
#include "esi/Accelerator.h"
#include "esi/Engines.h"

#include "zlib.h"

#include <cassert>
#include <stdexcept>

using namespace esi;
using namespace esi::services;

Service *Service::getChildService(Service::Type service, AppIDPath id,
                                  std::string implName,
                                  ServiceImplDetails details,
                                  HWClientDetails clients) {
  return conn.getService(service, id, implName, details, clients);
}

std::string SysInfo::getServiceSymbol() const { return "__builtin_SysInfo"; }

// Allocate 10MB for the uncompressed manifest. This should be plenty.
constexpr uint32_t MAX_MANIFEST_SIZE = 10 << 20;
/// Get the compressed manifest, uncompress, and return it.
std::string SysInfo::getJsonManifest() const {
  std::vector<uint8_t> compressed = getCompressedManifest();
  std::vector<Bytef> dst(MAX_MANIFEST_SIZE);
  uLongf dstSize = MAX_MANIFEST_SIZE;
  int rc =
      uncompress(dst.data(), &dstSize, compressed.data(), compressed.size());
  if (rc != Z_OK)
    throw std::runtime_error("zlib uncompress failed with rc=" +
                             std::to_string(rc));
  return std::string(reinterpret_cast<char *>(dst.data()), dstSize);
}

//===----------------------------------------------------------------------===//
// MMIO class implementations.
//===----------------------------------------------------------------------===//

MMIO::MMIO(AcceleratorConnection &conn, const AppIDPath &idPath,
           const HWClientDetails &clients)
    : Service(conn) {
  AppIDPath idParent = idPath.parent();
  for (const HWClientDetail &client : clients) {
    auto offsetIter = client.implOptions.find("offset");
    if (offsetIter == client.implOptions.end())
      throw std::runtime_error("MMIO client missing 'offset' option");
    Constant offset = std::any_cast<Constant>(offsetIter->second);
    uint64_t offsetVal = std::any_cast<uint64_t>(offset.value);
    if (offsetVal >= 1ull << 32)
      throw std::runtime_error("MMIO client offset mustn't exceed 32 bits");

    auto sizeIter = client.implOptions.find("size");
    if (sizeIter == client.implOptions.end())
      throw std::runtime_error("MMIO client missing 'size' option");
    Constant size = std::any_cast<Constant>(sizeIter->second);
    uint64_t sizeVal = std::any_cast<uint64_t>(size.value);
    if (sizeVal >= 1ull << 32)
      throw std::runtime_error("MMIO client size mustn't exceed 32 bits");
    AppIDPath absPath = idParent + client.relPath;
    regions[absPath] = RegionDescriptor{(uint32_t)offsetVal, (uint32_t)sizeVal};
  }
}

std::string MMIO::getServiceSymbol() const {
  return std::string(MMIO::StdName);
}
BundlePort *MMIO::getPort(AppIDPath id, const BundleType *type) const {
  auto regionIter = regions.find(id);
  if (regionIter == regions.end())
    return nullptr;
  return new MMIORegion(id.back(), const_cast<MMIO *>(this),
                        regionIter->second);
}

namespace {
class MMIOPassThrough : public MMIO {
public:
  MMIOPassThrough(const HWClientDetails &clients, const AppIDPath &idPath,
                  MMIO *parent)
      : MMIO(parent->getConnection(), idPath, clients), parent(parent) {}
  uint64_t read(uint32_t addr) const override { return parent->read(addr); }
  void write(uint32_t addr, uint64_t data) override {
    parent->write(addr, data);
  }

private:
  MMIO *parent;
};
} // namespace

Service *MMIO::getChildService(Service::Type service, AppIDPath id,
                               std::string implName, ServiceImplDetails details,
                               HWClientDetails clients) {
  if (service != typeid(MMIO))
    return Service::getChildService(service, id, implName, details, clients);
  return new MMIOPassThrough(clients, id, this);
}

//===----------------------------------------------------------------------===//
// MMIO Region service port class implementations.
//===----------------------------------------------------------------------===//

MMIO::MMIORegion::MMIORegion(AppID id, MMIO *parent, RegionDescriptor desc)
    : ServicePort(id, nullptr, {}), parent(parent), desc(desc) {}
uint64_t MMIO::MMIORegion::read(uint32_t addr) const {
  if (addr >= desc.size)
    throw std::runtime_error("MMIO read out of bounds: " + toHex(addr));
  return parent->read(desc.base + addr);
}
void MMIO::MMIORegion::write(uint32_t addr, uint64_t data) {
  if (addr >= desc.size)
    throw std::runtime_error("MMIO write out of bounds: " + toHex(addr));
  parent->write(desc.base + addr, data);
}

MMIOSysInfo::MMIOSysInfo(const MMIO *mmio)
    : SysInfo(mmio->getConnection()), mmio(mmio) {}

uint32_t MMIOSysInfo::getEsiVersion() const {
  uint64_t reg;
  if ((reg = mmio->read(MetadataOffset)) != MagicNumber)
    throw std::runtime_error("Invalid magic number: " + toHex(reg));
  return mmio->read(MetadataOffset + 8);
}

std::vector<uint8_t> MMIOSysInfo::getCompressedManifest() const {
  uint64_t version = getEsiVersion();
  if (version != 0)
    throw std::runtime_error("Unsupported ESI header version: " +
                             std::to_string(version));
  uint64_t manifestPtr = mmio->read(MetadataOffset + 0x10);
  uint64_t size = mmio->read(manifestPtr);
  uint64_t numWords = (size + 7) / 8;
  std::vector<uint64_t> manifestWords(numWords);
  for (size_t i = 0; i < numWords; ++i)
    manifestWords[i] = mmio->read(manifestPtr + 8 + (i * 8));

  std::vector<uint8_t> manifest;
  for (size_t i = 0; i < size; ++i) {
    uint64_t word = manifestWords[i / 8];
    manifest.push_back(word >> (8 * (i % 8)));
  }
  return manifest;
}

std::string HostMem::getServiceSymbol() const { return "__builtin_HostMem"; }

CustomService::CustomService(AppIDPath idPath, AcceleratorConnection &conn,
                             const ServiceImplDetails &details,
                             const HWClientDetails &clients)
    : Service(conn), id(idPath) {
  if (auto f = details.find("service"); f != details.end()) {
    serviceSymbol = std::any_cast<std::string>(f->second);
    // Strip off initial '@'.
    serviceSymbol = serviceSymbol.substr(1);
  }
}

BundlePort *CustomService::getPort(AppIDPath id, const BundleType *type) const {
  return new BundlePort(id.back(), type,
                        conn.getEngineMapFor(id).requestPorts(id, type));
}

FuncService::FuncService(AppIDPath idPath, AcceleratorConnection &conn,
                         ServiceImplDetails details, HWClientDetails clients)
    : Service(conn) {

  if (auto f = details.find("service"); f != details.end())
    // Strip off initial '@'.
    symbol = std::any_cast<std::string>(f->second).substr(1);
}

std::string FuncService::getServiceSymbol() const { return symbol; }

BundlePort *FuncService::getPort(AppIDPath id, const BundleType *type) const {
  return new Function(id.back(), type,
                      conn.getEngineMapFor(id).requestPorts(id, type));
}

FuncService::Function *FuncService::Function::get(AppID id, BundleType *type,
                                                  WriteChannelPort &arg,
                                                  ReadChannelPort &result) {
  return new Function(
      id, type, {{std::string("arg"), arg}, {std::string("result"), result}});
  return nullptr;
}

void FuncService::Function::connect() {
  if (connected)
    throw std::runtime_error("Function is already connected");
  if (channels.size() != 2)
    throw std::runtime_error("FuncService must have exactly two channels");
  arg = &getRawWrite("arg");
  arg->connect();
  result = &getRawRead("result");
  result->connect();
  connected = true;
}

std::future<MessageData>
FuncService::Function::call(const MessageData &argData) {
  if (!connected)
    throw std::runtime_error("Function must be 'connect'ed before calling");
  std::scoped_lock<std::mutex> lock(callMutex);
  arg->write(argData);
  return result->readAsync();
}

CallService::CallService(AcceleratorConnection &acc, AppIDPath idPath,
                         ServiceImplDetails details)
    : Service(acc) {
  if (auto f = details.find("service"); f != details.end())
    // Strip off initial '@'.
    symbol = std::any_cast<std::string>(f->second).substr(1);
}

std::string CallService::getServiceSymbol() const { return symbol; }

BundlePort *CallService::getPort(AppIDPath id, const BundleType *type) const {
  return new Callback(conn, id.back(), type,
                      conn.getEngineMapFor(id).requestPorts(id, type));
}

CallService::Callback::Callback(AcceleratorConnection &acc, AppID id,
                                const BundleType *type, PortMap channels)
    : ServicePort(id, type, channels), acc(acc) {}

CallService::Callback *CallService::Callback::get(AcceleratorConnection &acc,
                                                  AppID id,
                                                  const BundleType *type,
                                                  WriteChannelPort &result,
                                                  ReadChannelPort &arg) {
  return new Callback(acc, id, type, {{"arg", arg}, {"result", result}});
}

void CallService::Callback::connect(
    std::function<MessageData(const MessageData &)> callback, bool quick) {
  if (channels.size() != 2)
    throw std::runtime_error("CallService must have exactly two channels");
  result = &getRawWrite("result");
  result->connect();
  arg = &getRawRead("arg");
  if (quick) {
    // If it's quick, we can just call the callback directly.
    arg->connect([this, callback](MessageData argMsg) -> bool {
      MessageData resultMsg = callback(std::move(argMsg));
      this->result->write(std::move(resultMsg));
      return true;
    });
  } else {
    // If it's not quick, we need to use the service thread.
    arg->connect();
    acc.getServiceThread()->addListener(
        {arg}, [this, callback](ReadChannelPort *, MessageData argMsg) -> void {
          MessageData resultMsg = callback(std::move(argMsg));
          this->result->write(std::move(resultMsg));
        });
  }
}

TelemetryService::TelemetryService(AppIDPath idPath,
                                   AcceleratorConnection &conn,
                                   ServiceImplDetails details,
                                   HWClientDetails clients)
    : Service(conn) {}

std::string TelemetryService::getServiceSymbol() const {
  return std::string(TelemetryService::StdName);
}

BundlePort *TelemetryService::getPort(AppIDPath id,
                                      const BundleType *type) const {
  auto *port = new Telemetry(id.back(), type,
                             conn.getEngineMapFor(id).requestPorts(id, type));
  telemetryPorts.insert(std::make_pair(id, port));
  return port;
}

TelemetryService::Telemetry::Telemetry(AppID id, const BundleType *type,
                                       PortMap channels)
    : ServicePort(id, type, channels) {}

TelemetryService::Telemetry *
TelemetryService::Telemetry::get(AppID id, BundleType *type,
                                 WriteChannelPort &get, ReadChannelPort &data) {
  return new Telemetry(id, type, {{"get", get}, {"data", data}});
}

/// Connect to a particular telemetry port. The bundle should have two channels
/// -- get and data. Get should have type 'i0' and data can be anything.
void TelemetryService::Telemetry::connect() {
  if (channels.size() != 2)
    throw std::runtime_error("TelemetryService must have exactly two channels");
  get_req = &getRawWrite("get");
  // TODO: There are problems with DMA'ing i0. As a workaround, sometimes i1 is
  // used. When these issues are fixed, re-enable this check. There may also be
  // a problem with the void type.
  // if (!dynamic_cast<const VoidType *>(get_req->getType()))
  //   throw std::runtime_error("TelemetryService get channel must be void");
  get_req->connect();
  data = &getRawRead("data");
  data->connect();
}

std::future<MessageData> TelemetryService::Telemetry::read() {
  if (!get_req)
    throw std::runtime_error("TelemetryService get channel not connected");
  // TODO: This is a hack to get around the fact that we can't send a void
  // message. We need to send something, so we send a single byte whose value
  // doesn't matter.
  std::vector<uint8_t> empty = {1};
  get_req->write(MessageData(empty));
  return data->readAsync();
}

Service *ServiceRegistry::createService(AcceleratorConnection *acc,
                                        Service::Type svcType, AppIDPath id,
                                        std::string implName,
                                        ServiceImplDetails details,
                                        HWClientDetails clients) {
  // TODO: Add a proper registration mechanism.
  if (svcType == typeid(FuncService))
    return new FuncService(id, *acc, details, clients);
  if (svcType == typeid(CallService))
    return new CallService(*acc, id, details);
  if (svcType == typeid(TelemetryService))
    return new TelemetryService(id, *acc, details, clients);
  if (svcType == typeid(CustomService))
    return new CustomService(id, *acc, details, clients);
  return nullptr;
}

Service::Type ServiceRegistry::lookupServiceType(const std::string &svcName) {
  // TODO: Add a proper registration mechanism.
  if (svcName == "esi.service.std.func")
    return typeid(FuncService);
  if (svcName == "esi.service.std.call")
    return typeid(CallService);
  if (svcName == MMIO::StdName)
    return typeid(MMIO);
  if (svcName == HostMem::StdName)
    return typeid(HostMem);
  if (svcName == TelemetryService::StdName)
    return typeid(TelemetryService);
  return typeid(CustomService);
}
