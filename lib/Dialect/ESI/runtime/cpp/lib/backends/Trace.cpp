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

using namespace esi;
using namespace esi::services;
using namespace esi::backends::trace;

// We only support v0.
constexpr uint32_t ESIVersion = 0;

namespace {
class TraceChannelPort;
}

struct esi::backends::trace::TraceAccelerator::Impl {
  Impl(Mode mode, std::filesystem::path manifestJson,
       std::filesystem::path traceFile)
      : manifestJson(manifestJson), traceFile(traceFile) {
    if (!std::filesystem::exists(manifestJson))
      throw std::runtime_error("manifest file '" + manifestJson.string() +
                               "' does not exist");

    if (mode == Write) {
      // Open the trace file for writing.
      traceWrite = new std::ofstream(traceFile);
      if (!traceWrite->is_open())
        throw std::runtime_error("failed to open trace file '" +
                                 traceFile.string() + "'");
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

  void write(const AppIDPath &id, const std::string &portName, const void *data,
             size_t size);

private:
  std::ofstream *traceWrite;
  std::filesystem::path manifestJson;
  std::filesystem::path traceFile;
  std::vector<std::unique_ptr<ChannelPort>> channels;
};

void TraceAccelerator::Impl::write(const AppIDPath &id,
                                   const std::string &portName,
                                   const void *data, size_t size) {
  std::string b64data;
  utils::encodeBase64(data, size, b64data);

  *traceWrite << "write " << id << '.' << portName << ": " << b64data
              << std::endl;
}

std::unique_ptr<AcceleratorConnection>
TraceAccelerator::connect(Context &ctxt, std::string connectionString) {
  std::string modeStr;
  std::string manifestPath;
  std::string traceFile = "trace.log";

  // Parse the connection std::string.
  // <mode>:<manifest path>[:<traceFile>]
  std::regex connPattern("(\\w):([^:]+)(:(\\w+))?");
  std::smatch match;
  if (regex_search(connectionString, match, connPattern)) {
    modeStr = match[1];
    manifestPath = match[2];
    if (match[3].matched)
      traceFile = match[3];
  } else {
    throw std::runtime_error("connection std::string must be of the form "
                             "'<mode>:<manifest path>[:<traceFile>]'");
  }

  // Parse the mode.
  Mode mode;
  if (modeStr == "w")
    mode = Write;
  else
    throw std::runtime_error("unknown mode '" + modeStr + "'");

  return std::make_unique<TraceAccelerator>(ctxt, mode,
                                            std::filesystem::path(manifestPath),
                                            std::filesystem::path(traceFile));
}

TraceAccelerator::TraceAccelerator(Context &ctxt, Mode mode,
                                   std::filesystem::path manifestJson,
                                   std::filesystem::path traceFile)
    : AcceleratorConnection(ctxt) {
  impl = std::make_unique<Impl>(mode, manifestJson, traceFile);
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
  TraceSysInfo(std::filesystem::path manifestJson)
      : manifestJson(manifestJson) {}

  uint32_t getEsiVersion() const override { return ESIVersion; }

  std::string getJsonManifest() const override {
    // Read in the whole json file and return it.
    std::ifstream manifest(manifestJson);
    if (!manifest.is_open())
      throw std::runtime_error("failed to open manifest file '" +
                               manifestJson.string() + "'");
    std::stringstream buffer;
    buffer << manifest.rdbuf();
    manifest.close();
    return buffer.str();
  }

  std::vector<uint8_t> getCompressedManifest() const override {
    throw std::runtime_error(
        "compressed manifest not supported by trace backend");
  }

private:
  std::filesystem::path manifestJson;
};
} // namespace

namespace {
class WriteTraceChannelPort : public WriteChannelPort {
public:
  WriteTraceChannelPort(TraceAccelerator::Impl &impl, const Type *type,
                        const AppIDPath &id, const std::string &portName)
      : WriteChannelPort(type), impl(impl), id(id), portName(portName) {}

  virtual void write(const MessageData &data) override {
    impl.write(id, portName, data.getBytes(), data.getSize());
  }

protected:
  TraceAccelerator::Impl &impl;
  AppIDPath id;
  std::string portName;
};
} // namespace

namespace {
class ReadTraceChannelPort : public ReadChannelPort {
public:
  ReadTraceChannelPort(TraceAccelerator::Impl &impl, const Type *type)
      : ReadChannelPort(type) {}
  ~ReadTraceChannelPort() { disconnect(); }

  void disconnect() override {
    ReadChannelPort::disconnect();
    if (!dataPushThread.joinable())
      return;
    shutdown = true;
    shutdownCV.notify_all();
    dataPushThread.join();
  }

private:
  void connectImpl() override {
    assert(!dataPushThread.joinable() && "already connected");
    shutdown = false;
    dataPushThread = std::thread(&ReadTraceChannelPort::dataPushLoop, this);
  }

  MessageData genMessage() {
    std::ptrdiff_t numBits = getType()->getBitWidth();
    if (numBits < 0)
      // TODO: support other types.
      throw std::runtime_error("unsupported type for read: " +
                               getType()->getID());

    std::ptrdiff_t size = (numBits + 7) / 8;
    std::vector<uint8_t> bytes(size);
    for (std::ptrdiff_t i = 0; i < size; ++i)
      bytes[i] = rand() % 256;
    return MessageData(bytes);
  }

  void dataPushLoop() {
    std::mutex m;
    std::unique_lock<std::mutex> lock(m);
    while (!shutdown) {
      shutdownCV.wait_for(lock, std::chrono::milliseconds(100));
      while (this->callback(genMessage()))
        shutdownCV.wait_for(lock, std::chrono::milliseconds(10));
    }
  }

  std::thread dataPushThread;
  std::condition_variable shutdownCV;
  std::atomic<bool> shutdown;
};
} // namespace

namespace {
class TraceCustomService : public CustomService {
public:
  TraceCustomService(TraceAccelerator::Impl &impl, AppIDPath idPath,
                     const ServiceImplDetails &details,
                     const HWClientDetails &clients)
      : CustomService(idPath, details, clients) {}
};
} // namespace

std::map<std::string, ChannelPort &>
TraceAccelerator::Impl::requestChannelsFor(AppIDPath idPath,
                                           const BundleType *bundleType) {
  std::map<std::string, ChannelPort &> channels;
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

std::map<std::string, ChannelPort &>
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
