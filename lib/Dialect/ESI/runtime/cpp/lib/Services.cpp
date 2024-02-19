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

using namespace std;

using namespace esi;
using namespace esi::services;

string SysInfo::getServiceSymbol() const { return "__builtin_SysInfo"; }

// Allocate 10MB for the uncompressed manifest. This should be plenty.
constexpr uint32_t MAX_MANIFEST_SIZE = 10 << 20;
/// Get the compressed manifest, uncompress, and return it.
string SysInfo::getJsonManifest() const {
  vector<uint8_t> compressed = getCompressedManifest();
  vector<Bytef> dst(MAX_MANIFEST_SIZE);
  uLongf dstSize = MAX_MANIFEST_SIZE;
  int rc =
      uncompress(dst.data(), &dstSize, compressed.data(), compressed.size());
  if (rc != Z_OK)
    throw runtime_error("zlib uncompress failed with rc=" + to_string(rc));
  return string(reinterpret_cast<char *>(dst.data()), dstSize);
}

string MMIO::getServiceSymbol() const { return "__builtin_MMIO"; }

MMIOSysInfo::MMIOSysInfo(const MMIO *mmio) : mmio(mmio) {}

uint32_t MMIOSysInfo::getEsiVersion() const {
  uint32_t reg;
  if ((reg = mmio->read(MetadataOffset)) != MagicNumberLo)
    throw runtime_error("Invalid magic number low bits: " + toHex(reg));
  if ((reg = mmio->read(MetadataOffset + 4)) != MagicNumberHi)
    throw runtime_error("Invalid magic number high bits: " + toHex(reg));
  return mmio->read(MetadataOffset + 8);
}

vector<uint8_t> MMIOSysInfo::getCompressedManifest() const {
  uint32_t manifestPtr = mmio->read(MetadataOffset + 12);
  uint32_t size = mmio->read(manifestPtr);
  uint32_t numWords = (size + 3) / 4;
  vector<uint32_t> manifestWords(numWords);
  for (size_t i = 0; i < numWords; ++i)
    manifestWords[i] = mmio->read(manifestPtr + 4 + (i * 4));

  vector<uint8_t> manifest;
  for (size_t i = 0; i < size; ++i) {
    uint32_t word = manifestWords[i / 4];
    manifest.push_back(word >> (8 * (i % 4)));
  }
  return manifest;
}

CustomService::CustomService(AppIDPath idPath,
                             const ServiceImplDetails &details,
                             const HWClientDetails &clients)
    : id(idPath) {
  if (auto f = details.find("service"); f != details.end()) {
    serviceSymbol = any_cast<string>(f->second);
    // Strip off initial '@'.
    serviceSymbol = serviceSymbol.substr(1);
  }
}

FuncService::FuncService(AcceleratorConnection *acc, AppIDPath idPath,
                         std::string implName, ServiceImplDetails details,
                         HWClientDetails clients) {
  if (auto f = details.find("service"); f != details.end())
    // Strip off initial '@'.
    symbol = any_cast<string>(f->second).substr(1);
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
  if (channels.size() != 2)
    throw runtime_error("FuncService must have exactly two channels");
}

void FuncService::Function::connect() {
  arg.connect();
  result.connect();
}

MessageData FuncService::Function::call(const MessageData &argData) {
  arg.write(argData);
  MessageData resultData;
  // TODO: Return a future instead of spin blocking.
  while (!result.read(resultData))
    ;
  return resultData;
}

Service *ServiceRegistry::createService(AcceleratorConnection *acc,
                                        Service::Type svcType, AppIDPath id,
                                        std::string implName,
                                        ServiceImplDetails details,
                                        HWClientDetails clients) {
  // TODO: Add a proper registration mechanism.
  if (svcType == typeid(FuncService))
    return new FuncService(acc, id, implName, details, clients);
  return nullptr;
}

Service::Type ServiceRegistry::lookupServiceType(const std::string &svcName) {
  // TODO: Add a proper registration mechanism.
  if (svcName == "esi.service.std.func")
    return typeid(FuncService);
  return typeid(CustomService);
}
