//===- Xrt.h - ESI XRT device backend ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a specialization of the ESI C++ API (backend) for connection into
// hardware on an XRT device. Requires XRT C++ library.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp).
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_BACKENDS_XRT_H
#define ESI_BACKENDS_XRT_H

#include "esi/Accelerator.h"

#include <memory>

namespace esi {
namespace backends {
namespace xrt {

/// Connect to an ESI simulation.
class XrtAccelerator : public esi::AcceleratorConnection {
public:
  struct Impl;

  XrtAccelerator(Context &, std::string xclbin, std::string kernelName);
  ~XrtAccelerator();
  static std::unique_ptr<AcceleratorConnection>
  connect(Context &, std::string connectionString);

protected:
  virtual Service *createService(Service::Type service, AppIDPath path,
                                 std::string implName,
                                 const ServiceImplDetails &details,
                                 const HWClientDetails &clients) override;

private:
  std::unique_ptr<Impl> impl;
};

} // namespace xrt
} // namespace backends
} // namespace esi

#endif // ESI_BACKENDS_XRT_H
