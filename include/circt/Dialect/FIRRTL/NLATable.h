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
#include "llvm/ADT/SetOperations.h"
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

  /// Lookup all NLAs an operation participates in.
  ArrayRef<NonLocalAnchor> lookup(Operation *op);

  /// Lookup all NLAs an operation participates in.
  ArrayRef<NonLocalAnchor> lookup(StringAttr name);

  /// Resolve a symbol to an NLA.
  NonLocalAnchor getNLA(StringAttr name);

  /// Resolve a symbol to a Module.
  FModuleLike getModule(StringAttr name);

  /// Insert a new NLA.
  void insert(NonLocalAnchor nlaOp);

  /// Remove the NLA from the analysis.
  void erase(NonLocalAnchor nlaOp);

  /// Compute the NLAs that are common between the two modules, \p mod1 and \p
  /// mod2 and insert them into the set \p common.
  ///  The set of NLAs that an instance op participates in is the set of common
  ///  NLAs between the parent module and the instance target. This can be used
  ///  to get the set of NLAs that an InstanceOp participates in, instead of
  ///  recording them on the op in the IR.
  void commonNLAs(StringAttr mod1, StringAttr mod2,
                  DenseSet<NonLocalAnchor> &common) {
    auto mod1NLAs = lookup(mod1);
    auto mod2NLAs = lookup(mod2);
    common.insert(mod1NLAs.begin(), mod1NLAs.end());
    DenseSet<NonLocalAnchor> set2(mod2NLAs.begin(), mod2NLAs.end());
    llvm::set_intersect(common, set2);
  }

  /// Get the instances that the InstanceOp participates in.
  void getInstanceNLAs(InstanceOp inst, DenseSet<NonLocalAnchor> &nlas) {
    commonNLAs(inst->getParentOfType<FModuleOp>().getNameAttr(),
               inst.moduleNameAttr().getAttr(), nlas);
  }

  //===-------------------------------------------------------------------------
  // Methods to keep an NLATable up to date.
  //
  // These methods are not thread safe.  Make sure that modifications are
  // properly synchronized or performed in a serial context.  When the
  // NLATable is used as an analysis, this is only safe when the pass is
  // on a CircuitOp.

  /// Record a new NLA operation.
  void addNLA(NonLocalAnchor nla);

  /// Stop tracking a module.
  void eraseModule(StringAttr name);

  /// Move NLA \p name from \p oldModule to \p newModule, updating the nla and
  /// updating the tracking.
  void updateModuleInNLA(StringAttr name, StringAttr oldModule,
                         StringAttr newModule);

  /// Move NLA \p nlaOp from \p oldModule to \p newModule, updating the nla and
  /// updating the tracking.
  void updateModuleInNLA(NonLocalAnchor nlaOp, StringAttr oldModule,
                         StringAttr newModule);

  /// Rename a module, this updates the name to module tracking and the name to
  /// NLA tracking.
  void renameModule(StringAttr oldModName, StringAttr newModName);

  /// Replace the module \p oldModName with \p newModName in the namepath of any
  /// NLA. Since the module is being updated, the symbols inside the module
  /// should also be renamed. Use the rename map \p innerSymRenameMap to update
  /// the inner_sym names in the namepath.
  void renameModuleAndInnerRef(
      StringAttr newModName, StringAttr oldModName,
      const DenseMap<StringAttr, StringAttr> &innerSymRenameMap);

  /// Remove the NLA from the Module.
  void removeNLAfromModule(NonLocalAnchor nla, StringAttr mod);

  /// Remove all the nlas in the set from the module.
  void removeNLAsfromModule(const DenseSet<NonLocalAnchor> &nlas,
                            StringAttr mod);

private:
  NLATable(const NLATable &) = delete;

  /// Map modules to the NLA's that target them.
  llvm::DenseMap<StringAttr, SmallVector<NonLocalAnchor, 4>> nodeMap;

  /// Map symbol names to module and NLA operations.
  llvm::DenseMap<StringAttr, Operation *> symToOp;
};

} // namespace firrtl
} // namespace circt
#endif // CIRCT_DIALECT_FIRRTL_NLATABLE_H
