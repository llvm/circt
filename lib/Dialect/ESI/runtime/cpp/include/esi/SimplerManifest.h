//===- SimplerManifest.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manifest parsing and API creation. Simplified version.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_SIMPLER_MANIFEST_H
#define ESI_SIMPLER_MANIFEST_H

#include "esi/Common.h"
#include "esi/Context.h"
#include "esi/Types.h"

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace esi {

// Forward declarations.
class AcceleratorConnection;
class Accelerator;

/// Class to parse a manifest. It also constructs the dynamic API for the
/// accelerator.
class SimplerManifest {
public:
  class Impl;

  SimplerManifest(const SimplerManifest &) = delete;
  SimplerManifest(Context &ctxt, const std::string &jsonManifest);
  ~SimplerManifest();

  Accelerator *buildAccelerator(AcceleratorConnection &acc) const;

private:
  std::unique_ptr<Impl> impl;
};

} // namespace esi

#endif // ESI_SIMPLER_MANIFEST_H
