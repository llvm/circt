//===- NLATable.h - Non-Local Anchor Table----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FIRRTL NLATable.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_FIRRTL_NLATABLE_H
#define CIRCT_DIALECT_FIRRTL_NLATABLE_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator.h"

namespace circt {
namespace firrtl {

/// This table tracks nlas and what modules participate in them.
///
/// To use this class, retrieve a cached copy from the analysis manager:
///   auto &nlaTable = getAnalysis<NLTATable>(getOperation());
class NLATable {

public:
  /// Create a new NLA table of a circuit.  This must be called on a FIRRTL
  /// CircuitOp or MLIR ModuleOp.
  explicit NLATable(Operation *operation);

  /// Lookup all NLAs an operation participates in
  ArrayRef<NonLocalAnchor> lookup(Operation *op);

  /// Lookup all NLAs an operation participates in
  ArrayRef<NonLocalAnchor> lookup(StringAttr name);

  /// Resolve a symbol to an NLA
  NonLocalAnchor getNLA(StringAttr name);

  /// Resolve a symbol to a Module
  FModuleLike getModule(StringAttr name);

  //===-------------------------------------------------------------------------
  // Methods to keep an NLATable up to date.
  //
  // These methods are not thread safe.  Make sure that modifications are
  // properly synchronized or performed in a serial context.  When the
  // NLATable is used as an analysis, this is only safe when the pass is
  // on a CircuitOp.

  // Stop tracking a module
  void eraseModule(StringAttr name);

  // Move NLA 'name' from oldModule to newModule, updating the nla and updating
  // the tracking.
  void updateModuleInNLA(StringAttr name, StringAttr oldModule,
                         StringAttr newModule);
  void updateModuleInNLA(NonLocalAnchor nlaOp, StringAttr oldModule,
                         StringAttr newModule);

  // Rename a module, this updates the name to module tracking and the name to
  // NLA tracking.
  void renameModule(StringAttr oldModName, StringAttr newModName);

private:
  NLATable(const NLATable &) = delete;

  /// This maps each operation to its graph node.
  llvm::DenseMap<StringAttr, SmallVector<NonLocalAnchor, 4>> nodeMap;
  llvm::DenseMap<StringAttr, Operation *> symToOp;
};

} // namespace firrtl
} // namespace circt
#endif // CIRCT_DIALECT_FIRRTL_NLATABLE_H
