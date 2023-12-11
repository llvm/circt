//===- Manifest.h - Metadata on the accelerator -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manifest parsing and API creation.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_MANIFEST_H
#define ESI_MANIFEST_H

#include "esi/Common.h"
#include "esi/Types.h"

#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace esi {

// Forward declarations.
class AcceleratorConnection;
class Accelerator;

/// Class to parse a manifest. It also constructs the dynamic API for the
/// accelerator.
class Manifest {
public:
  class Impl;

  Manifest(const Manifest &) = delete;
  Manifest(const std::string &jsonManifest);

  uint32_t getApiVersion() const;
  // Modules which have designer specified metadata.
  std::vector<ModuleInfo> getModuleInfos() const;

  // Build a dynamic design hierarchy from the manifest.
  std::unique_ptr<Accelerator>
  buildAccelerator(AcceleratorConnection &acc) const;

  /// Get a Type from the manifest based on its ID. Types are uniqued here.
  std::optional<std::reference_wrapper<const Type>> getType(Type::ID id) const;

  /// The Type Table is an ordered list of types. The offset can be used to
  /// compactly and uniquely within a design. It does not include all of the
  /// types in a design -- just the ones listed in the 'types' section of the
  /// manifest.
  const std::vector<std::reference_wrapper<const Type>> &getTypeTable() const;

private:
  std::shared_ptr<Impl> impl;
};

} // namespace esi

std::ostream &operator<<(std::ostream &, const esi::ModuleInfo &);
std::ostream &operator<<(std::ostream &, const esi::AppID &);

#endif // ESI_MANIFEST_H
