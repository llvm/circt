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
    const Constant *offset = std::any_cast<Constant>(&offsetIter->second);
    if (!offset)
      throw std::runtime_error(
          "MMIO client 'offset' option must be a constant");
    const uint64_t *offsetVal = std::any_cast<uint64_t>(&offset->value);
    if (!offsetVal)
      throw std::runtime_error(
          "MMIO client 'offset' option must be an integer");
    if (*offsetVal >= 1ull << 32)
      throw std::runtime_error("MMIO client offset mustn't exceed 32 bits");

    auto sizeIter = client.implOptions.find("size");
    if (sizeIter == client.implOptions.end())
      throw std::runtime_error("MMIO client missing 'size' option");
    const Constant *size = std::any_cast<Constant>(&sizeIter->second);
    if (!size)
      throw std::runtime_error("MMIO client 'size' option must be a constant");
    const uint64_t *sizeVal = std::any_cast<uint64_t>(&size->value);
    if (!sizeVal)
      throw std::runtime_error("MMIO client 'size' option must be an integer");
    if (*sizeVal >= 1ull << 32)
      throw std::runtime_error("MMIO client size mustn't exceed 32 bits");
    regions[client.relPath] = RegionDescriptor{
        static_cast<uint32_t>(*offsetVal), static_cast<uint32_t>(*sizeVal)};
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
    : Service(conn), id(idPath), mmio(nullptr) {
  // Compute our parents idPath path.
  AppIDPath prefix = std::move(idPath);
  if (prefix.size() > 0)
    prefix.pop_back();
  for (const HWClientDetail &client : clients) {
    if (client.implOptions.contains("type") &&
        std::any_cast<std::string>(client.implOptions.at("type")) != "mmio")
      continue; // Not an MMIO assignment.
    AppIDPath fullClientPath = prefix + client.relPath;
    auto offsetIter = client.implOptions.find("offset");
    if (offsetIter == client.implOptions.end()) {
      conn.getLogger().warning("Telemetry",
                               "mmio client " + fullClientPath.toStr() +
                                   " missing 'offset' option, skipping");
      continue;
    }
    const Constant *offset = std::any_cast<Constant>(&offsetIter->second);
    if (offset == nullptr) {
      conn.getLogger().warning(
          "Telemetry", "mmio client " + fullClientPath.toStr() +
                           " 'offset' option must be a constant, skipping");
      continue;
    }
    const uint64_t *offsetVal = std::any_cast<uint64_t>(&offset->value);
    if (offsetVal == nullptr) {
      conn.getLogger().warning(
          "Telemetry", "mmio client " + fullClientPath.toStr() +
                           " 'offset' option must be an integer, skipping");
      continue;
    }
    portAddressAssignments.emplace(fullClientPath, *offsetVal);
  }
}

std::string TelemetryService::getServiceSymbol() const {
  return std::string(TelemetryService::StdName);
}

MMIO::MMIORegion *TelemetryService::getMMIORegion() const {
  if (!mmio) {
    AppIDPath lastPath;
    AppIDPath mmioPath = id;
    mmioPath.pop_back();
    mmioPath.push_back(AppID("__telemetry_mmio"));
    auto port = conn.getAccelerator().resolvePort(mmioPath, lastPath);
    if (!port)
      throw std::runtime_error("TelemetryService: could not resolve port " +
                               id.toStr() + ". Got as far as " +
                               lastPath.toStr());
    mmio = dynamic_cast<MMIO::MMIORegion *>(port);
    if (!mmio)
      throw std::runtime_error("TelemetryService: port " + id.toStr() +
                               " is not a MMIO region");
  }
  return mmio;
}

BundlePort *TelemetryService::getPort(AppIDPath id,
                                      const BundleType *type) const {
  auto offsetIter = portAddressAssignments.find(id);
  auto *port = new Metric(id.back(), type, {}, this,
                          offsetIter != portAddressAssignments.end()
                              ? std::optional<uint64_t>(offsetIter->second)
                              : std::nullopt);
  telemetryPorts.insert(std::make_pair(id, port));
  return port;
}

Service *TelemetryService::getChildService(Service::Type service, AppIDPath id,
                                           std::string implName,
                                           ServiceImplDetails details,
                                           HWClientDetails clients) {
  TelemetryService *child = new TelemetryService(id, conn, details, clients);
  children.push_back(child);
  return child;
}

TelemetryService::Metric::Metric(AppID id, const BundleType *type,
                                 PortMap channels,
                                 const TelemetryService *telemetryService,
                                 std::optional<uint64_t> offset)
    : ServicePort(id, type, channels), telemetryService(telemetryService),
      mmio(nullptr), offset(offset) {}

/// Connect to a particular telemetry port. Offset should be non-nullopt.
void TelemetryService::Metric::connect() {
  if (!offset.has_value())
    throw std::runtime_error("Telemetry offset not found for " + id.toString());
  mmio = telemetryService->getMMIORegion();
  assert(mmio && "TelemetryService: MMIO region not found");
}

std::future<MessageData> TelemetryService::Metric::read() {
  return std::async(std::launch::async, [this]() {
    uint64_t data = readInt();
    return MessageData::from(data);
  });
}

uint64_t TelemetryService::Metric::readInt() {
  assert(offset.has_value() &&
         "Telemetry offset must be set. Checked in connect().");
  assert(mmio && "TelemetryService: MMIO region not set");
  return mmio->read(*offset);
}

void TelemetryService::getTelemetryPorts(std::map<AppIDPath, Metric *> &ports) {
  for (const auto &entry : telemetryPorts)
    ports[entry.first] = entry.second;
  for (TelemetryService *child : children)
    child->getTelemetryPorts(ports);
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
