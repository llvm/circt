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
class TraceEngine;
} // namespace

TraceAccelerator::Impl &TraceAccelerator::getImpl() { return *impl; }

struct esi::backends::trace::TraceAccelerator::Impl {
  friend class TraceAccelerator;
  Impl(Mode mode, std::filesystem::path manifestJson,
       std::filesystem::path traceFile)
      : manifestJson(manifestJson), traceFile(traceFile) {
    engine = std::make_unique<TraceEngine>(*this);
    if (!std::filesystem::exists(manifestJson))
      throw std::runtime_error("manifest file '" + manifestJson.string() +
                               "' does not exist");

    if (mode == Write) {
      // Open the trace file for writing.
      traceWrite = new std::ofstream(traceFile);
      if (!traceWrite->is_open())
        throw std::runtime_error("failed to open trace file '" +
                                 traceFile.string() + "'");
    } else if (mode == Discard) {
      traceWrite = nullptr;
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

  Service *createService(TraceAccelerator &conn, Service::Type svcType,
                         AppIDPath idPath, const ServiceImplDetails &details,
                         const HWClientDetails &clients);

  void adoptChannelPort(ChannelPort *port) { channels.emplace_back(port); }

  void write(const AppIDPath &id, const std::string &portName, const void *data,
             size_t size, const std::string &prefix = "");
  std::ostream &write(std::string service) {
    assert(traceWrite && "traceWrite is null");
    *traceWrite << "[" << service << "] ";
    return *traceWrite;
  }
  bool isWriteable() { return traceWrite; }

private:
  std::ofstream *traceWrite;
  std::filesystem::path manifestJson;
  std::filesystem::path traceFile;
  std::vector<std::unique_ptr<ChannelPort>> channels;
  std::unique_ptr<TraceEngine> engine;
};

void TraceAccelerator::Impl::write(const AppIDPath &id,
                                   const std::string &portName,
                                   const void *data, size_t size,
                                   const std::string &prefix) {
  if (!isWriteable())
    return;
  std::string b64data;
  utils::encodeBase64(data, size, b64data);

  *traceWrite << prefix << (prefix.empty() ? "w" : "W") << "rite " << id << '.'
              << portName << ": " << b64data << std::endl;
}

std::unique_ptr<AcceleratorConnection>
TraceAccelerator::connect(Context &ctxt, std::string connectionString) {
  std::string modeStr;
  std::string manifestPath;
  std::string traceFile = "trace.log";

  // Parse the connection std::string.
  // <mode>:<manifest path>[:<traceFile>]
  std::regex connPattern("([\\w-]):([^:]+)(:(\\w+))?");
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
  else if (modeStr == "-")
    mode = Discard;
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
TraceAccelerator::~TraceAccelerator() { disconnect(); }

namespace {
class TraceSysInfo : public SysInfo {
public:
  TraceSysInfo(AcceleratorConnection &conn, std::filesystem::path manifestJson)
      : SysInfo(conn), manifestJson(manifestJson) {}

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

  bool tryWrite(const MessageData &data) override {
    impl.write(id, portName, data.getBytes(), data.getSize(), "try");
    return true;
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

private:
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

  bool pollImpl() override { return callback(genMessage()); }
};
} // namespace

namespace {
class TraceEngine : public Engine {
public:
  TraceEngine(TraceAccelerator::Impl &impl) : impl(impl) {}

  std::unique_ptr<ChannelPort> createPort(AppIDPath idPath,
                                          const std::string &channelName,
                                          BundleType::Direction dir,
                                          const Type *type) override {
    std::unique_ptr<ChannelPort> port;
    if (BundlePort::isWrite(dir))
      port = std::make_unique<WriteTraceChannelPort>(impl, type, idPath,
                                                     channelName);
    else
      port = std::make_unique<ReadTraceChannelPort>(impl, type);
    return port;
  }

private:
  TraceAccelerator::Impl &impl;
};
} // namespace

void TraceAccelerator::createEngine(const std::string &dmaEngineName,
                                    AppIDPath idPath,
                                    const ServiceImplDetails &details,
                                    const HWClientDetails &clients) {
  registerEngine(idPath, std::make_unique<TraceEngine>(getImpl()), clients);
}

class TraceMMIO : public MMIO {
public:
  TraceMMIO(TraceAccelerator &conn, const HWClientDetails &clients)
      : MMIO(conn, clients), impl(conn.getImpl()) {}

  virtual uint64_t read(uint32_t addr) const override {
    uint64_t data = rand();
    if (impl.isWriteable())
      impl.write("MMIO") << "[" << std::hex << addr << "] -> " << data
                         << std::endl;
    return data;
  }
  virtual void write(uint32_t addr, uint64_t data) override {
    if (!impl.isWriteable())
      return;
    impl.write("MMIO") << "[" << std::hex << addr << "] <- " << data
                       << std::endl;
  }

private:
  TraceAccelerator::Impl &impl;
};

class TraceHostMem : public HostMem {
public:
  TraceHostMem(TraceAccelerator &conn) : HostMem(conn), impl(conn.getImpl()) {}

  struct TraceHostMemRegion : public HostMemRegion {
    TraceHostMemRegion(std::size_t size, TraceAccelerator::Impl &impl)
        : impl(impl) {
      ptr = malloc(size);
      this->size = size;
    }
    virtual ~TraceHostMemRegion() {
      if (impl.isWriteable())
        impl.write("HostMem") << "free " << ptr << std::endl;
      free(ptr);
    }
    virtual void *getPtr() const override { return ptr; }
    virtual std::size_t getSize() const override { return size; }

  private:
    void *ptr;
    std::size_t size;
    TraceAccelerator::Impl &impl;
  };

  virtual std::unique_ptr<HostMemRegion>
  allocate(std::size_t size, HostMem::Options opts) const override {
    auto ret =
        std::unique_ptr<HostMemRegion>(new TraceHostMemRegion(size, impl));
    if (impl.isWriteable())
      impl.write("HostMem 0x")
          << ret->getPtr() << " allocate " << size
          << " bytes. Writeable: " << opts.writeable
          << ", useLargePages: " << opts.useLargePages << std::endl;
    return ret;
  }
  virtual bool mapMemory(void *ptr, std::size_t size,
                         HostMem::Options opts) const override {

    if (impl.isWriteable())
      impl.write("HostMem")
          << "map 0x" << ptr << " size " << size
          << " bytes. Writeable: " << opts.writeable
          << ", useLargePages: " << opts.useLargePages << std::endl;
    return true;
  }
  virtual void unmapMemory(void *ptr) const override {
    if (impl.isWriteable())
      impl.write("HostMem") << "unmap 0x" << ptr << std::endl;
  }

private:
  TraceAccelerator::Impl &impl;
};

Service *TraceAccelerator::createService(Service::Type svcType,
                                         AppIDPath idPath, std::string implName,
                                         const ServiceImplDetails &details,
                                         const HWClientDetails &clients) {
  if (svcType == typeid(SysInfo))
    return new TraceSysInfo(*this, getImpl().manifestJson);
  if (svcType == typeid(MMIO))
    return new TraceMMIO(*this, clients);
  if (svcType == typeid(HostMem))
    return new TraceHostMem(*this);
  return nullptr;
}

REGISTER_ACCELERATOR("trace", TraceAccelerator);
