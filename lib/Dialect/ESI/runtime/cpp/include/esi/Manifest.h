//===- Manifest.h - Metadata on the accelerator -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_MANIFEST_H
#define ESI_MANIFEST_H

#include <any>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace esi {

//===----------------------------------------------------------------------===//
// Common accelerator description types.
//===----------------------------------------------------------------------===//

struct AppID {
  const std::string name;
  const std::optional<uint32_t> idx;
};
using AppIDPath = std::vector<AppID>;

struct ModuleInfo {
  const std::optional<std::string> name;
  const std::optional<std::string> summary;
  const std::optional<std::string> version;
  const std::optional<std::string> repo;
  const std::optional<std::string> commitHash;
  const std::map<std::string, std::any> extra;
};

struct ServicePort {
  std::string name;
  std::string portName;
};

//===----------------------------------------------------------------------===//
// Manifest parsing and API creation.
//===----------------------------------------------------------------------===//

// Forward declarations.
namespace internal {
class ManifestProxy;
} // namespace internal
class Accelerator;
class Design;

/// Class to parse a manifest. It also constructs the dynamic API for the
/// accelerator.
class Manifest {
public:
  Manifest(const std::string &jsonManifest);
  ~Manifest();

  uint32_t apiVersion() const;
  // Modules which have designer specified metadata.
  std::vector<ModuleInfo> moduleInfos() const;

  // Build a dynamic design hierarchy from the manifest.
  std::unique_ptr<Design> buildDesign(Accelerator &acc) const;

private:
  internal::ManifestProxy &manifest;
};

} // namespace esi

std::ostream &operator<<(std::ostream &, const esi::ModuleInfo &);
std::ostream &operator<<(std::ostream &, const esi::AppID &);

#endif // ESI_MANIFEST_H
