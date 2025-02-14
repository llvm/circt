//===- BLIFParser.h - .blif to BLIF dialect parser --------------*- C++ -*-===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the .blif file parser.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_BLIF_BLIFPARSER_H
#define CIRCT_DIALECT_BLIF_BLIFPARSER_H

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
namespace blif {

struct BLIFParserOptions {
  /// Specify how @info locators should be handled.
  enum class InfoLocHandling {
    /// If this is set to true, the @info locators are ignored, and the
    /// locations are set to the location in the .BLIF file.
    IgnoreInfo,
    /// Prefer @info locators, fallback to .BLIF locations.
    PreferInfo,
    /// Attach both @info locators (when present) and .BLIF locations.
    FusedInfo
  };

  InfoLocHandling infoLocatorHandling = InfoLocHandling::PreferInfo;

  /// parse strict blif instead of extended blif
  bool strictBLIF = false;
};

mlir::OwningOpRef<mlir::ModuleOp>
importBLIFFile(llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context,
               mlir::TimingScope &ts, BLIFParserOptions options = {});

void registerFromBLIFFileTranslation();

} // namespace blif
} // namespace circt

#endif // CIRCT_DIALECT_BLIF_BLIFPARSER_H
