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

#include "circt/Support/LLVM.h"

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
  /// If this is set to true, the @info locators are ignored, and the locations
  /// are set to the location in the .fir file.
  bool ignoreInfoLocators = false;
  /// The number of annotation files that were specified on the command line.
  /// This, along with numOMIRFiles provides structure to the buffers in the
  /// source manager.
  unsigned numAnnotationFiles;
  bool scalarizeTopModule = false;
  bool scalarizeExtModules = false;
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

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_FIRPARSER_H
