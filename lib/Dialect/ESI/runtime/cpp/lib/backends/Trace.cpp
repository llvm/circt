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
#include "esi/StdServices.h"

#include <fstream>
#include <iostream>
#include <regex>

using namespace std;

using namespace esi;
using namespace esi::services;
using namespace esi::backends::trace;

// We only support v1.
constexpr uint32_t ESIVersion = 1;

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

  return std::make_unique<TraceAccelerator>(
      mode, filesystem::path(manifestPath), filesystem::path(traceFile));
}

namespace {
class TraceSysInfo : public SysInfo {
public:
  TraceSysInfo(std::filesystem::path manifestJson)
      : manifestJson(manifestJson) {}

  uint32_t esiVersion() const override { return ESIVersion; }

  std::string jsonManifest() const override {
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

  std::vector<uint8_t> compressedManifest() const override {
    throw runtime_error("compressed manifest not supported by trace backend");
  }

private:
  std::filesystem::path manifestJson;
};
} // namespace

struct esi::backends::trace::TraceAccelerator::Impl {
  Impl(Mode mode, std::filesystem::path manifestJson,
       std::filesystem::path traceFile)
      : mode(mode), manifestJson(manifestJson), traceFile(traceFile) {
    if (!filesystem::exists(manifestJson))
      throw runtime_error("manifest file '" + manifestJson.string() +
                          "' does not exist");
  }

  Service *createService(Service::Type svcType);

private:
  Mode mode;
  std::filesystem::path manifestJson;
  std::filesystem::path traceFile;
};

Service *TraceAccelerator::Impl::createService(Service::Type svcType) {
  if (svcType == typeid(SysInfo))
    return new TraceSysInfo(manifestJson);
  return nullptr;
}

TraceAccelerator::TraceAccelerator(Mode mode,
                                   std::filesystem::path manifestJson,
                                   std::filesystem::path traceFile) {
  impl = std::make_unique<Impl>(mode, manifestJson, traceFile);
}

Service *TraceAccelerator::createService(Service::Type svcType) {
  return impl->createService(svcType);
}

REGISTER_ACCELERATOR("trace", TraceAccelerator);
