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

#include "zlib.h"

#include <cassert>
#include <stdexcept>

using namespace esi;
using namespace esi::services;

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

std::string MMIO::getServiceSymbol() const { return "__builtin_MMIO"; }

MMIOSysInfo::MMIOSysInfo(const MMIO *mmio) : mmio(mmio) {}

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

CustomService::CustomService(AppIDPath idPath,
                             const ServiceImplDetails &details,
                             const HWClientDetails &clients)
    : id(idPath) {
  if (auto f = details.find("service"); f != details.end()) {
    serviceSymbol = std::any_cast<std::string>(f->second);
    // Strip off initial '@'.
    serviceSymbol = serviceSymbol.substr(1);
  }
}

FuncService::FuncService(AcceleratorConnection *acc, AppIDPath idPath,
                         const std::string &implName,
                         ServiceImplDetails details, HWClientDetails clients) {
  if (auto f = details.find("service"); f != details.end())
    // Strip off initial '@'.
    symbol = std::any_cast<std::string>(f->second).substr(1);
}

std::string FuncService::getServiceSymbol() const { return symbol; }

ServicePort *
FuncService::getPort(AppIDPath id, const BundleType *type,
                     const std::map<std::string, ChannelPort &> &channels,
                     AcceleratorConnection &acc) const {
  return new Function(id.back(), channels);
}

FuncService::Function::Function(
    AppID id, const std::map<std::string, ChannelPort &> &channels)
    : ServicePort(id, channels),
      arg(dynamic_cast<WriteChannelPort &>(channels.at("arg"))),
      result(dynamic_cast<ReadChannelPort &>(channels.at("result"))) {
  assert(channels.size() == 2 && "FuncService must have exactly two channels");
}

FuncService::Function *FuncService::Function::get(AppID id,
                                                  WriteChannelPort &arg,
                                                  ReadChannelPort &result) {
  return new Function(id, {{"arg", arg}, {"result", result}});
}

void FuncService::Function::connect() {
  arg.connect();
  result.connect();
}

std::future<MessageData>
FuncService::Function::call(const MessageData &argData) {
  std::scoped_lock<std::mutex> lock(callMutex);
  arg.write(argData);
  return result.readAsync();
}

CallService::CallService(AcceleratorConnection *acc, AppIDPath idPath,
                         std::string implName, ServiceImplDetails details,
                         HWClientDetails clients) {
  if (auto f = details.find("service"); f != details.end())
    // Strip off initial '@'.
    symbol = std::any_cast<std::string>(f->second).substr(1);
}

std::string CallService::getServiceSymbol() const { return symbol; }

ServicePort *
CallService::getPort(AppIDPath id, const BundleType *type,
                     const std::map<std::string, ChannelPort &> &channels,
                     AcceleratorConnection &acc) const {
  return new Callback(acc, id.back(), channels);
}

CallService::Callback::Callback(
    AcceleratorConnection &acc, AppID id,
    const std::map<std::string, ChannelPort &> &channels)
    : ServicePort(id, channels),
      arg(dynamic_cast<ReadChannelPort &>(channels.at("arg"))),
      result(dynamic_cast<WriteChannelPort &>(channels.at("result"))),
      acc(acc) {
  if (channels.size() != 2)
    throw std::runtime_error("CallService must have exactly two channels");
}

void CallService::Callback::connect(
    std::function<MessageData(const MessageData &)> callback, bool quick) {
  result.connect();
  if (quick) {
    // If it's quick, we can just call the callback directly.
    arg.connect([this, callback](MessageData argMsg) -> bool {
      MessageData resultMsg = callback(std::move(argMsg));
      this->result.write(std::move(resultMsg));
      return true;
    });
  } else {
    // If it's not quick, we need to use the service thread.
    arg.connect();
    acc.getServiceThread()->addListener(
        {&arg},
        [this, callback](ReadChannelPort *, MessageData argMsg) -> void {
          MessageData resultMsg = callback(std::move(argMsg));
          this->result.write(std::move(resultMsg));
        });
  }
}

Service *ServiceRegistry::createService(AcceleratorConnection *acc,
                                        Service::Type svcType, AppIDPath id,
                                        std::string implName,
                                        ServiceImplDetails details,
                                        HWClientDetails clients) {
  // TODO: Add a proper registration mechanism.
  if (svcType == typeid(FuncService))
    return new FuncService(acc, id, implName, details, clients);
  if (svcType == typeid(CallService))
    return new CallService(acc, id, implName, details, clients);
  return nullptr;
}

Service::Type ServiceRegistry::lookupServiceType(const std::string &svcName) {
  // TODO: Add a proper registration mechanism.
  if (svcName == "esi.service.std.func")
    return typeid(FuncService);
  if (svcName == "esi.service.std.call")
    return typeid(CallService);
  return typeid(CustomService);
}
