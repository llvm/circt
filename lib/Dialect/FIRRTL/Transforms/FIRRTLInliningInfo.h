//===- FIRRTLInliningInfo.h - Inlining classification -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InliningInfo helper.
// It computes the per-module facts the inliner needs before rewriting:
// 1. Whether a module is marked inline or flatten.
// 2. Whether it can be reached through a module being flattened.
// 3. Whether it survives inlining at all.
//
// It also records the parents-before-children order the rewrite walk uses.
//
// A private helper of the ModuleInliner pass, not a registered analysis:
// computing the facts validates annotations and can fail with diagnostics,
// which an analysis constructor cannot surface.
// Construct it directly and check the result of run().
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_FIRRTL_TRANSFORMS_FIRRTLINLININGINFO_H
#define DIALECT_FIRRTL_TRANSFORMS_FIRRTLINLININGINFO_H

#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace firrtl {

class InliningInfo {
public:
  struct ModuleInfo {
    /// The module carries an inline annotation.
    bool hasInline : 1;

    /// The module carries a flatten annotation.
    bool hasFlatten : 1;

    /// Some instantiation path reaches this module through one being flattened.
    /// This is a context-insensitive over-approximation (MOP):
    /// any parent flattening it sets the bit, not all parents.
    bool underFlatten : 1;

    /// The dual of underFlatten:
    /// there exists a path along which this module isn't flattened away.
    /// Analysis-internal stepping stone for `isLive`; consumers want that.
    /// (isLive = this, minus regular modules absorbed by their inline mark.)
    bool hasUnflattenedPath : 1;

    /// Will this module exist in the final result?
    bool isLive : 1;

    /// Derived named predicates over the facts above, used in the walk.

    /// This module's regular children remain instantiated:
    /// some path keeps it instantiated and it is not flattening them away.
    bool keepsChildrenInstantiated() const {
      return hasUnflattenedPath && !hasFlatten;
    }

    /// Some copy of this module's body may be flattened,
    /// by its own annotation or a flattening ancestor.
    bool mayBeFlattened() const { return underFlatten || hasFlatten; }

    // No default member initialization of bitfield members until C++20.
    ModuleInfo()
        : hasInline(false), hasFlatten(false), underFlatten(false),
          hasUnflattenedPath(false), isLive(false) {}
  };

  using ModuleInfoMap = DenseMap<Operation *, ModuleInfo>;

  InliningInfo(CircuitOp circuit, InstanceGraph &instanceGraph,
               mlir::SymbolTable &symbolTable)
      : circuit(circuit), instanceGraph(instanceGraph),
        symbolTable(symbolTable) {}

  /// Compute the per-module facts, leaving the IR untouched on failure.
  /// Rejected annotations (e.g., inline/flatten on a non-regular module)
  /// fail with a diagnostic; partially-honored ones draw a warning.
  ///
  /// Closed world:
  /// - Liveness sees instance-graph uses plus symbol uses on circuit-level ops.
  /// - Instance-graph uses include instance_choice targets and alternatives,
  ///   seeded live even from a dead holder.
  /// - A module referenced only inside a module body by a non-instance op
  ///   is invisible and may be erased under that reference.
  ///
  /// Contract (correctness precondition):
  /// Body-to-module references are only from instance operations.
  LogicalResult run();

  const ModuleInfoMap &getModuleInfoMap() const { return modInfoMap; }

  /// Regular modules in inverse post-order: parents strictly before children.
  /// Recorded during the propagation walk so a consumer need not recompute it.
  /// Frozen with the facts, it is the order this analysis was computed over.
  ArrayRef<FModuleOp> getIPOModules() const { return ipoModules; }

  /// Print the computed facts in a stable, test-friendly format.
  void print(raw_ostream &os) const;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  CircuitOp circuit;
  InstanceGraph &instanceGraph;
  mlir::SymbolTable &symbolTable;

  ModuleInfoMap modInfoMap;
  SmallVector<FModuleOp, 16> ipoModules;
};

} // namespace firrtl
} // namespace circt

#endif // DIALECT_FIRRTL_TRANSFORMS_FIRRTLINLININGINFO_H
