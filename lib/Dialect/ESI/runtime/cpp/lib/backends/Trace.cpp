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
#include "esi/Design.h"
#include "esi/StdServices.h"
#include "esi/Utils.h"

#include <fstream>
#include <iostream>
#include <regex>

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

unique_ptr<Accelerator> TraceAccelerator::connect(string connectionString) {
  string modeStr;
  string manifestPath;
  string traceFile = "trace.json";

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

  return make_unique<TraceAccelerator>(mode, filesystem::path(manifestPath),
                                       filesystem::path(traceFile));
}

TraceAccelerator::TraceAccelerator(Mode mode, filesystem::path manifestJson,
                                   filesystem::path traceFile) {
  impl = make_unique<Impl>(mode, manifestJson, traceFile);
}

Service *TraceAccelerator::createService(Service::Type svcType,
                                         AppIDPath idPath,
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
  WriteTraceChannelPort(TraceAccelerator::Impl &impl, const AppIDPath &id,
                        const string &portName)
      : impl(impl), id(id), portName(portName) {}

  virtual void write(const void *data, size_t size) override {
    impl.write(id, portName, data, size);
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
  ReadTraceChannelPort(TraceAccelerator::Impl &impl) {}

  virtual ssize_t read(void *data, size_t maxSize) override;
};
} // namespace

ssize_t ReadTraceChannelPort::read(void *data, size_t maxSize) {
  uint8_t *dataPtr = reinterpret_cast<uint8_t *>(data);
  for (size_t i = 0; i < maxSize; ++i)
    dataPtr[i] = rand() % 256;
  return maxSize;
}

namespace {
class TraceCustomService : public CustomService {
public:
  TraceCustomService(TraceAccelerator::Impl &impl, AppIDPath idPath,
                     const ServiceImplDetails &details,
                     const HWClientDetails &clients)
      : CustomService(idPath, details, clients), impl(impl) {}

  virtual map<string, ChannelPort &>
  requestChannelsFor(AppIDPath idPath, const BundleType &bundleType,
                     BundlePort::Direction svcDir) override {
    map<string, ChannelPort &> channels;
    for (auto [name, dir, type] : bundleType.getChannels()) {
      ChannelPort *port;
      if (BundlePort::isWrite(dir, svcDir))
        port = new WriteTraceChannelPort(impl, idPath, name);
      else
        port = new ReadTraceChannelPort(impl);
      channels.emplace(name, *port);
      impl.adoptChannelPort(port);
    }
    return channels;
  }

private:
  TraceAccelerator::Impl &impl;
};
} // namespace

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
