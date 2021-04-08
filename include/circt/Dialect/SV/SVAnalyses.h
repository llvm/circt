//===- SVAnalyses.h - SV analysis passes ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the SV analysis passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVANALYSES_H
#define CIRCT_DIALECT_SV_SVANALYSES_H

#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/StringSet.h"

namespace mlir {
class Operation;
class ModuleOp;
} // namespace mlir

namespace circt {
namespace sv {

/// A lookup table for legalized and legalized output names.
///
/// This analysis establishes a mapping from the name of modules, interfaces,
/// and various other operations to a legalized version that is properly
/// uniquified and does not collide with any keywords.
struct LegalNamesAnalysis {
  LegalNamesAnalysis(mlir::Operation *op);

  /// Return the legalized name for an operation or assert if there is none.
  StringRef getOperationName(Operation *op) const;
  /// Return the legalized name for an argument to an operation or assert if
  /// there is none.
  StringRef getArgName(Operation *op, size_t argNum) const;
  /// Return the legalized name for a result from an operation or assert if
  /// there is none.
  StringRef getResultName(Operation *op, size_t resultNum) const;

private:
  /// Mapping from operations to their legalized name. Used for module, extern
  /// module, and interface operations.
  llvm::DenseMap<Operation *, StringRef> operationNames;

  /// Mapping from operation arguments to their legalized name. Used for module
  /// input ports.
  llvm::DenseMap<std::pair<Operation *, size_t>, StringRef> argNames;

  /// Mapping from operation results to their legalized name. Used for module
  /// output ports.
  llvm::DenseMap<std::pair<Operation *, size_t>, StringRef> resultNames;

  /// Set of used names, to ensure uniqueness.
  llvm::StringSet<> usedNames;

  /// Numeric suffix used as uniquification agent when resolving conflicts.
  size_t nextGeneratedNameID = 0;

  StringRef legalizeOperation(Operation *op, StringAttr name);
  StringRef legalizeArg(Operation *op, size_t argNum, StringAttr name);
  StringRef legalizeResult(Operation *op, size_t resultNum, StringAttr name);

  void analyzeModulePorts(Operation *module);
  void analyze(mlir::ModuleOp op);
  void analyze(rtl::RTLModuleOp op);
  void analyze(rtl::RTLModuleExternOp op);
  void analyze(sv::InterfaceOp op);
};

} // namespace sv
} // namespace circt

#endif // CIRCT_DIALECT_SV_SVANALYSES_H
