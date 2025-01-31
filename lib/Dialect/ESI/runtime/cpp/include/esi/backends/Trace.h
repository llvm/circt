//===- Trace.h - ESI trace backend ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a specialization of the ESI C++ API (backend) for trace-based
// Accelerator interactions. This means that it will have the capability to read
// trace files recorded from interactions with an actual connection. It also has
// a mode wherein it will write to a file (for sends) and produce random data
// (for receives). Both modes are intended for debugging without a simulation.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp/).
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_BACKENDS_COSIM_H
#define ESI_BACKENDS_COSIM_H

#include "esi/Accelerator.h"

#include <filesystem>
#include <memory>

namespace esi {
namespace backends {
namespace trace {

/// Connect to an ESI simulation.
class TraceAccelerator : public esi::AcceleratorConnection {
public:
  enum Mode {
    // Write data sent to the accelerator to the trace file. Produce random
    // garbage data for reads from the accelerator.
    Write,

    // Sent data to the accelerator is compared against the trace file's record.
    // Data read from the accelerator is read from the trace file.
    // TODO: Full trace mode not yet supported.
    // Read

    // Discard all data sent to the accelerator. Disable trace file generation.
    Discard,
  };

  /// Create a trace-based accelerator backend.
  /// \param mode The mode of operation. See Mode.
  /// \param manifestJson The path to the manifest JSON file.
  /// \param traceFile The path to the trace file. For 'Write' mode, this file
  ///   is opened for writing. For 'Read' mode, this file is opened for reading.
  TraceAccelerator(Context &, Mode mode, std::filesystem::path manifestJson,
                   std::filesystem::path traceFile);
  ~TraceAccelerator() override;

  /// Parse the connection string and instantiate the accelerator. Format is:
  /// "<mode>:<manifest path>[:<traceFile>]".
  static std::unique_ptr<AcceleratorConnection>
  connect(Context &, std::string connectionString);

  /// Internal implementation.
  struct Impl;
  Impl &getImpl();

protected:
  void createEngine(const std::string &engineTypeName, AppIDPath idPath,
                    const ServiceImplDetails &details,
                    const HWClientDetails &clients) override;

  virtual Service *createService(Service::Type service, AppIDPath idPath,
                                 std::string implName,
                                 const ServiceImplDetails &details,
                                 const HWClientDetails &clients) override;

private:
  std::unique_ptr<Impl> impl;
};

} // namespace trace
} // namespace backends
} // namespace esi

#endif // ESI_BACKENDS_COSIM_H
