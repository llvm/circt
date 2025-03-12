//===- FIRParser.h - .fir to FIRRTL dialect parser --------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the .fir file parser.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRPARSER_H
#define CIRCT_DIALECT_FIRRTL_FIRPARSER_H

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Support/LLVM.h"
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class LocationAttr;
class TimingScope;
} // namespace mlir

namespace circt {
namespace firrtl {

struct FIRParserOptions {
  /// Specify how @info locators should be handled.
  enum class InfoLocHandling {
    /// If this is set to true, the @info locators are ignored, and the
    /// locations are set to the location in the .fir file.
    IgnoreInfo,
    /// Prefer @info locators, fallback to .fir locations.
    PreferInfo,
    /// Attach both @info locators (when present) and .fir locations.
    FusedInfo
  };

  InfoLocHandling infoLocatorHandling = InfoLocHandling::PreferInfo;

  /// The number of annotation files that were specified on the command line.
  /// This, provides structure to the buffers in the source manager.
  unsigned numAnnotationFiles;
  bool scalarizePublicModules = false;
  bool scalarizeInternalModules = false;
  bool scalarizeExtModules = false;
  std::vector<std::string> enableLayers;
  std::vector<std::string> disableLayers;
  std::optional<LayerSpecialization> defaultLayerSpecialization;
  std::vector<std::string> selectInstanceChoice;
};

mlir::OwningOpRef<mlir::ModuleOp> importFIRFile(llvm::SourceMgr &sourceMgr,
                                                mlir::MLIRContext *context,
                                                mlir::TimingScope &ts,
                                                FIRParserOptions options = {});

// Decode a source locator string `spelling`, returning a pair indicating that
// the `spelling` was correct and an optional location attribute.  The
// `skipParsing` option can be used to short-circuit parsing and just do
// validation of the `spelling`.  This require both an Identifier and a
// FileLineColLoc to use for caching purposes and context as the cache may be
// updated with a new identifier.
//
// This utility exists because source locators can exist outside of normal
// "parsing".  E.g., these can show up in annotations or in Object Model 2.0
// JSON.
//
// TODO: This API is super wacky and should be streamlined to hide the
// caching.
std::pair<bool, std::optional<mlir::LocationAttr>>
maybeStringToLocation(llvm::StringRef spelling, bool skipParsing,
                      mlir::StringAttr &locatorFilenameCache,
                      FileLineColLoc &fileLineColLocCache,
                      MLIRContext *context);

void registerFromFIRFileTranslation();

/// The FIRRTL specification version.
struct FIRVersion {
  constexpr FIRVersion(uint16_t major, uint16_t minor, uint16_t patch)
      : major{major}, minor{minor}, patch{patch} {}

  explicit constexpr operator uint64_t() const {
    return uint64_t(major) << 32 | uint64_t(minor) << 16 | uint64_t(patch);
  }

  constexpr bool operator<(FIRVersion rhs) const {
    return uint64_t(*this) < uint64_t(rhs);
  }

  constexpr bool operator>(FIRVersion rhs) const {
    return uint64_t(*this) > uint64_t(rhs);
  }

  constexpr bool operator<=(FIRVersion rhs) const {
    return uint64_t(*this) <= uint64_t(rhs);
  }

  constexpr bool operator>=(FIRVersion rhs) const {
    return uint64_t(*this) >= uint64_t(rhs);
  }

  uint16_t major;
  uint16_t minor;
  uint16_t patch;
};

/// The current minimum version of FIRRTL that the parser supports.
constexpr FIRVersion minimumFIRVersion(2, 0, 0);

/// The next version of FIRRTL that is not yet released.
///
/// Features use this version if they have been landed on the main branch of
/// `chipsalliance/firrtl-spec`, but have not been part of a release yet. Once a
/// new version of the spec is released, all uses of `nextFIRVersion` in the
/// parser are replaced with the concrete version `{x, y, z}`, and this
/// declaration here is bumped to the next probable version number.
constexpr FIRVersion nextFIRVersion(4, 3, 0);

/// A marker for parser features that are currently missing from the spec.
///
/// Features use this version if they have _not_ been added to the documentation
/// in the `chipsalliance/firrtl-spec` repository. This allows us to distinguish
/// features that are released in the next version of the spec and features that
/// are still missing from the spec.
constexpr FIRVersion missingSpecFIRVersion = nextFIRVersion;

/// The version of FIRRTL that the exporter produces. This is always the next
/// version, since it contains any new developments.
constexpr FIRVersion exportFIRVersion = nextFIRVersion;

template <typename T>
T &operator<<(T &os, FIRVersion version) {
  return os << version.major << "." << version.minor << "." << version.patch;
}

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRPARSER_H
