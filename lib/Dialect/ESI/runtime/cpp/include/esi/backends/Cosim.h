//===- Cosim.h - ESI C++ cosimulation backend -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a specialization of the ESI C++ API (backend) for connection into a
// simulation of an ESI system. Currently uses Cap'nProto RPC, but that could
// change. Requires Cap'nProto C++ library.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp).
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_BACKENDS_COSIM_H
#define ESI_BACKENDS_COSIM_H

#include "esi/Accelerator.h"

#include <memory>

namespace esi {
namespace backends {
namespace cosim {

/// Connect to an ESI simulation.
class CosimAccelerator : public esi::AcceleratorConnection {
public:
  struct Impl;

  CosimAccelerator(Context &, std::string hostname, uint16_t port);
  static std::unique_ptr<AcceleratorConnection>
  connect(Context &, std::string connectionString);

  // Different ways to retrieve the manifest in Cosimulation.
  enum ManifestMethod {
    Cosim, // Use the backdoor cosim interface. Default.
    MMIO,  // Use MMIO emulation.
  };
  // Set the way this connection will retrieve the manifest.
  void setManifestMethod(ManifestMethod method);

  /// Request the host side channel ports for a particular instance (identified
  /// by the AppID path). For convenience, provide the bundle type and direction
  /// of the bundle port.
  virtual std::map<std::string, ChannelPort &>
  requestChannelsFor(AppIDPath, const BundleType *) override;

protected:
  virtual Service *createService(Service::Type service, AppIDPath path,
                                 std::string implName,
                                 const ServiceImplDetails &details,
                                 const HWClientDetails &clients) override;

private:
  std::unique_ptr<Impl> impl;
  ManifestMethod manifestMethod = Cosim;
};

} // namespace cosim
} // namespace backends
} // namespace esi

#endif // ESI_BACKENDS_COSIM_H
