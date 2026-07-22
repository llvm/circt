//===- FIRRTLInliningInfo.h - Inlining classification -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InliningInfo helper.
// It computes, per module, the facts the FIRRTL module inliner needs before
// it rewrites anything: whether the module is marked inline or flatten,
// whether it can be reached through a module that is being flattened, and
// whether it survives inlining at all.
// It also records the parents-before-children module order the inliner's
// rewrite walk uses.
//
// This is a private helper of the ModuleInliner pass, not a registered
// AnalysisManager analysis: computing the facts validates the annotations and
// can fail with diagnostics, which an analysis constructor cannot surface.
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

    /// Some instantiation path to this module passes through a module being
    /// flattened.
    /// This is a context-insensitive over-approximation: any parent
    /// flattening it sets the bit, not all parents.
    bool underFlatten : 1;

    /// Some instantiation path reaches this module that no flattened
    /// ancestor absorbs (underFlatten's dual).
    /// Analysis-internal stepping stone for `isLive`; consumers want that.
    /// (isLive = this, minus regular modules absorbed by their inline mark.)
    bool hasUnflattenedPath : 1;

    /// Will this module exist in the final result?
    bool isLive : 1;

    /// Derived predicates over the facts above, named for the walk that
    /// consumes them.

    /// This module's regular children remain instantiated: some path keeps
    /// this module instantiated and it is not flattening them away.
    bool keepsChildrenInstantiated() const {
      return hasUnflattenedPath && !hasFlatten;
    }

    /// Some copy of this module's body may be flattened -- by its own
    /// annotation or a flattening ancestor.
    bool mayBeFlattened() const { return underFlatten || hasFlatten; }

    // Can't have default member initialization of bitfield members until
    // C++20.
    ModuleInfo()
        : hasInline(false), hasFlatten(false), underFlatten(false),
          hasUnflattenedPath(false), isLive(false) {}
  };

  using ModuleInfoMap = DenseMap<Operation *, ModuleInfo>;

  InliningInfo(CircuitOp circuit, InstanceGraph &instanceGraph,
               mlir::SymbolTable &symbolTable)
      : circuit(circuit), instanceGraph(instanceGraph),
        symbolTable(symbolTable) {}

  /// Compute the per-module facts.  Fails (with a diagnostic) on annotations
  /// the inliner rejects, e.g. inline/flatten on a non-regular module; warns
  /// on annotations it can only partially honor.  On failure the IR is
  /// untouched.
  ///
  /// Closed world: liveness sees instance-graph uses plus symbol uses on
  /// circuit-level ops.  Instance-graph uses include instance_choice targets
  /// and alternatives, seeded live even from a dead holder.  A module
  /// referenced only from inside another module's body by a non-instance op
  /// is invisible and may be erased under that reference.  In-contract for
  /// the stock pipeline, where body references to modules are
  /// instance-mediated before LowerXMR, and strictly broader than the old
  /// inliner's scan -- but a closed world, and this line is its boundary.
  LogicalResult run();

  const ModuleInfoMap &getModuleInfoMap() const { return modInfoMap; }

  /// All regular modules in inverse post-order: parents strictly before
  /// children.
  /// Recorded during the propagation walk so a consumer need not recompute
  /// it.
  /// Valid as long as the instance graph is not mutated after run().
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
