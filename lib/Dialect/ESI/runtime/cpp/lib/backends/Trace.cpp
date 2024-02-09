//===- Trace.cpp - Implementation of trace backend -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp/lib/).
//
//===----------------------------------------------------------------------===//

#include "esi/backends/Trace.h"

#include "esi/Accelerator.h"
#include "esi/Services.h"
#include "esi/Utils.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

using namespace std;

using namespace esi;
using namespace esi::services;
using namespace esi::backends::trace;

// We only support v1.
constexpr uint32_t ESIVersion = 1;

namespace {
class TraceChannelPort;
}

struct esi::backends::trace::TraceAccelerator::Impl {
  Impl(Mode mode, filesystem::path manifestJson, filesystem::path traceFile)
      : manifestJson(manifestJson), traceFile(traceFile) {
    if (!filesystem::exists(manifestJson))
      throw runtime_error("manifest file '" + manifestJson.string() +
                          "' does not exist");

    if (mode == Write) {
      // Open the trace file for writing.
      traceWrite = new ofstream(traceFile);
      if (!traceWrite->is_open())
        throw runtime_error("failed to open trace file '" + traceFile.string() +
                            "'");
    } else {
      assert(false && "not implemented");
    }
  }

  ~Impl() {
    if (traceWrite) {
      traceWrite->close();
      delete traceWrite;
    }
  }

  Service *createService(Service::Type svcType, AppIDPath idPath,
                         const ServiceImplDetails &details,
                         const HWClientDetails &clients);

  /// Request the host side channel ports for a particular instance (identified
  /// by the AppID path). For convenience, provide the bundle type and direction
  /// of the bundle port.
  std::map<std::string, ChannelPort &> requestChannelsFor(AppIDPath,
                                                          const BundleType *);

  void adoptChannelPort(ChannelPort *port) { channels.emplace_back(port); }

  void write(const AppIDPath &id, const string &portName, const void *data,
             size_t size);

private:
  ofstream *traceWrite;
  filesystem::path manifestJson;
  filesystem::path traceFile;
  vector<unique_ptr<ChannelPort>> channels;
};

void TraceAccelerator::Impl::write(const AppIDPath &id, const string &portName,
                                   const void *data, size_t size) {
  string b64data;
  utils::encodeBase64(data, size, b64data);

  *traceWrite << "write " << id << '.' << portName << ": " << b64data << endl;
}

unique_ptr<AcceleratorConnection>
TraceAccelerator::connect(Context &ctxt, string connectionString) {
  string modeStr;
  string manifestPath;
  string traceFile = "trace.log";

  // Parse the connection string.
  // <mode>:<manifest path>[:<traceFile>]
  regex connPattern("(\\w):([^:]+)(:(\\w+))?");
  smatch match;
  if (regex_search(connectionString, match, connPattern)) {
    modeStr = match[1];
    manifestPath = match[2];
    if (match[3].matched)
      traceFile = match[3];
  } else {
    throw runtime_error("connection string must be of the form "
                        "'<mode>:<manifest path>[:<traceFile>]'");
  }

  // Parse the mode.
  Mode mode;
  if (modeStr == "w")
    mode = Write;
  else
    throw runtime_error("unknown mode '" + modeStr + "'");

  return make_unique<TraceAccelerator>(
      ctxt, mode, filesystem::path(manifestPath), filesystem::path(traceFile));
}

TraceAccelerator::TraceAccelerator(Context &ctxt, Mode mode,
                                   filesystem::path manifestJson,
                                   filesystem::path traceFile)
    : AcceleratorConnection(ctxt) {
  impl = make_unique<Impl>(mode, manifestJson, traceFile);
}

Service *TraceAccelerator::createService(Service::Type svcType,
                                         AppIDPath idPath, std::string implName,
                                         const ServiceImplDetails &details,
                                         const HWClientDetails &clients) {
  return impl->createService(svcType, idPath, details, clients);
}
namespace {
class TraceSysInfo : public SysInfo {
public:
  TraceSysInfo(filesystem::path manifestJson) : manifestJson(manifestJson) {}

  uint32_t getEsiVersion() const override { return ESIVersion; }

  string getJsonManifest() const override {
    // Read in the whole json file and return it.
    ifstream manifest(manifestJson);
    if (!manifest.is_open())
      throw runtime_error("failed to open manifest file '" +
                          manifestJson.string() + "'");
    stringstream buffer;
    buffer << manifest.rdbuf();
    manifest.close();
    return buffer.str();
  }

  vector<uint8_t> getCompressedManifest() const override {
    throw runtime_error("compressed manifest not supported by trace backend");
  }

private:
  filesystem::path manifestJson;
};
} // namespace

namespace {
class WriteTraceChannelPort : public WriteChannelPort {
public:
  WriteTraceChannelPort(TraceAccelerator::Impl &impl, const Type *type,
                        const AppIDPath &id, const string &portName)
      : WriteChannelPort(type), impl(impl), id(id), portName(portName) {}

  virtual void write(const MessageData &data) override {
    impl.write(id, portName, data.getBytes(), data.getSize());
  }

protected:
  TraceAccelerator::Impl &impl;
  AppIDPath id;
  string portName;
};
} // namespace

namespace {
class ReadTraceChannelPort : public ReadChannelPort {
public:
  ReadTraceChannelPort(TraceAccelerator::Impl &impl, const Type *type)
      : ReadChannelPort(type) {}

  virtual bool read(MessageData &data) override;

private:
  size_t numReads = 0;
};
} // namespace

bool ReadTraceChannelPort::read(MessageData &data) {
  if ((++numReads & 0x1) == 1)
    return false;

  std::ptrdiff_t numBits = getType()->getBitWidth();
  if (numBits < 0)
    // TODO: support other types.
    throw runtime_error("unsupported type for read: " + getType()->getID());

  std::ptrdiff_t size = (numBits + 7) / 8;
  std::vector<uint8_t> bytes(size);
  for (std::ptrdiff_t i = 0; i < size; ++i)
    bytes[i] = rand() % 256;
  data = MessageData(bytes);
  return true;
}

namespace {
class TraceCustomService : public CustomService {
public:
  TraceCustomService(TraceAccelerator::Impl &impl, AppIDPath idPath,
                     const ServiceImplDetails &details,
                     const HWClientDetails &clients)
      : CustomService(idPath, details, clients) {}
};
} // namespace

map<string, ChannelPort &>
TraceAccelerator::Impl::requestChannelsFor(AppIDPath idPath,
                                           const BundleType *bundleType) {
  map<string, ChannelPort &> channels;
  for (auto [name, dir, type] : bundleType->getChannels()) {
    ChannelPort *port;
    if (BundlePort::isWrite(dir))
      port = new WriteTraceChannelPort(*this, type, idPath, name);
    else
      port = new ReadTraceChannelPort(*this, type);
    channels.emplace(name, *port);
    adoptChannelPort(port);
  }
  return channels;
}

map<string, ChannelPort &>
TraceAccelerator::requestChannelsFor(AppIDPath idPath,
                                     const BundleType *bundleType) {
  return impl->requestChannelsFor(idPath, bundleType);
}

Service *
TraceAccelerator::Impl::createService(Service::Type svcType, AppIDPath idPath,
                                      const ServiceImplDetails &details,
                                      const HWClientDetails &clients) {
  if (svcType == typeid(SysInfo))
    return new TraceSysInfo(manifestJson);
  if (svcType == typeid(CustomService))
    return new TraceCustomService(*this, idPath, details, clients);
  return nullptr;
}

REGISTER_ACCELERATOR("trace", TraceAccelerator);
