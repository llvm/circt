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

#include "CosimDpi.capnp.h"
#include <capnp/ez-rpc.h>

#include <fstream>
#include <iostream>

using namespace esi;
using namespace esi::backends::cosim;

namespace esi::backends::cosim {

/// Implement the SysInfo API for cosimulation.
class CosimSysInfo final : public esi::SysInfo {
private:
  friend class CosimAccelerator;
  CosimSysInfo() = default;

public:
  /// Get the ESI version number to check version compatibility.
  uint32_t esiVersion() const override;

  /// Return the JSON-formatted system manifest.
  std::string rawJsonManifest() const override;
};
} // namespace esi::backends::cosim

// For now, just return dummy values since these are not yet supported by the
// hardware.
uint32_t CosimSysInfo::esiVersion() const { return -1; }
std::string CosimSysInfo::rawJsonManifest() const { return ""; }

/// Parse the connection string and instantiate the accelerator. Support the
/// traditional 'host:port' syntax and a path to 'cosim.cfg' which is output by
/// the cosimulation when it starts (which is useful when it chooses its own
/// port).
std::unique_ptr<Accelerator>
CosimAccelerator::connect(std::string connectionString) {
  std::string portStr;
  std::string host = "localhost";

  size_t colon;
  if ((colon = connectionString.find(':')) != std::string::npos) {
    portStr = connectionString.substr(colon + 1);
    host = connectionString.substr(0, colon);
  } else {
    std::ifstream cfg(connectionString);
    std::string line, key, value;

    while (std::getline(cfg, line))
      if ((colon = line.find(":")) != std::string::npos) {
        key = line.substr(0, colon);
        value = line.substr(colon + 1);
        if (key == "port")
          portStr = value;
        else if (key == "host")
          host = value;
      }

    if (portStr.size() == 0)
      throw std::runtime_error("port line not found in file");
  }
  uint16_t port = std::stoul(portStr);
  return std::make_unique<CosimAccelerator>(host, port);
}

/// Construct and connect to a cosim server.
// TODO: Implement this.
CosimAccelerator::CosimAccelerator(std::string hostname, uint16_t port)
    : info(nullptr) {
  std::cout << hostname << ":" << port << std::endl;
}
CosimAccelerator::~CosimAccelerator() {
  if (info)
    delete info;
}

const SysInfo &CosimAccelerator::sysInfo() {
  if (info == nullptr)
    info = new CosimSysInfo();
  return *info;
}

namespace {
/// Register the cosim backend.
struct InitCosim {
  InitCosim() {
    registerBackend("cosim", &backends::cosim::CosimAccelerator::connect);
  }
};
InitCosim initCosim;
} // namespace
