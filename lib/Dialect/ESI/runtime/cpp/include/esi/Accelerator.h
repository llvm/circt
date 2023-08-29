//===- StdServices.h - ESI standard services C++ API ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic ESI APIs.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_ACCELERATOR_H
#define ESI_ACCELERATOR_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace esi {
class SysInfo;

/// An ESI accelerator system.
class Accelerator {
public:
  static std::unique_ptr<Accelerator> connect(std::string backend,
                                              std::string connection);
  virtual ~Accelerator(){};

  virtual const SysInfo &sysInfo() = 0;
};

/// Information about the Accelerator system.
class SysInfo {
public:
  virtual ~SysInfo(){};

  /// Get the ESI version number to check version compatibility.
  virtual uint32_t esiVersion() const = 0;

  /// Return the JSON-formatted system manifest.
  virtual std::string rawJsonManifest() const = 0;
};

using BackendCreate = std::function<std::unique_ptr<Accelerator>(std::string)>;
void registerBackend(std::string name, BackendCreate create);

} // namespace esi

#endif // ESI_ACCELERATOR_H
