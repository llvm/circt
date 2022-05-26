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
  /// CircuitOp or MLIR ModuleOp. To esnure that the analysis does not return
  /// stale data while a pass is running, it should be kept up-to-date when
  /// modules are added or renamed and NLAs are updated.
  explicit NLATable(Operation *operation);

  /// Lookup all NLAs an operation participates in. This returns a reference to
  /// the internal record, so make a copy before making any update to the
  /// NLATable.
  ArrayRef<NonLocalAnchor> lookup(Operation *op);

  /// Lookup all NLAs an operation participates in. This returns a reference to
  /// the internal record, so make a copy before making any update to the
  /// NLATable.
  ArrayRef<NonLocalAnchor> lookup(StringAttr name);

  /// Resolve a symbol to an NLA.
  NonLocalAnchor getNLA(StringAttr name);

  /// Resolve a symbol to a Module.
  FModuleLike getModule(StringAttr name);

  /// Insert a new NLA. This updates two internal records,
  /// 1. Update the map for the `nlaOp` name to the Operation.
  /// 2. For each module in the NLA namepath, insert the NLA into the list of
  /// NonlocalAnchors that participate in the corresponding module. This does
  /// not update the module name to module op map, if any potentially new module
  /// in the namepath does not already exist in the record.
  void insert(NonLocalAnchor nlaOp);

  /// Remove the NLA from the analysis. This updates two internal records,
  /// 1. Remove the NLA name to the operation map entry.
  /// 2. For each module in the namepath of the NLA, remove the entry from the
  /// list of NLAs that the module participates in.
  void erase(NonLocalAnchor nlaOp);

  /// Compute the NLAs that are common between the two modules, `mod1` and
  /// `mod2` and insert them into the set `common`.
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

  /// Get the NLAs that the InstanceOp participates in, insert it to the
  /// DenseSet `nlas`.
  void getInstanceNLAs(InstanceOp inst, DenseSet<NonLocalAnchor> &nlas) {
    auto instSym = inst.inner_symAttr();
    // If there is no inner sym on the InstanceOp, then it doesnot participate
    // in any NLA.
    if (!instSym)
      return;
    auto mod = inst->getParentOfType<FModuleOp>().getNameAttr();
    // Get the NLAs that are common between the parent module and the target
    // module. This should contain the NLAs that this InstanceOp participates
    // in.
    commonNLAs(inst->getParentOfType<FModuleOp>().getNameAttr(),
               inst.moduleNameAttr().getAttr(), nlas);
    // Handle the case when there are more than one Instances for the same
    // target module. Getting the `commonNLA`, in that case is not enough,
    // remove the NLAs that donot have the InstanceOp as the innerSym.
    for (auto nla : llvm::make_early_inc_range(nlas)) {
      if (!nla.hasInnerSym(mod, instSym))
        nlas.erase(nla);
    }
  }

  /// Get the NLAs that the module `modName` particiaptes in, and insert them
  /// into the DenseSet `nlas`.
  void getNLAsInModule(StringAttr modName, DenseSet<NonLocalAnchor> &nlas) {
    for (auto nla : lookup(modName))
      nlas.insert(nla);
  }

  //===-------------------------------------------------------------------------
  // Methods to keep an NLATable up to date.
  //
  // These methods are not thread safe.  Make sure that modifications are
  // properly synchronized or performed in a serial context.  When the
  // NLATable is used as an analysis, this is only safe when the pass is
  // on a CircuitOp.

  /// Record a new NLA operation. Duplicate of `insert'.
  /// TODO: Remove this.
  void addNLA(NonLocalAnchor nla);

  /// Record a new FModuleLike operation.
  void addModule(FModuleLike mod);

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

  /// Remove all the nlas in the set `nlas` from the module.
  void removeNLAsfromModule(const DenseSet<NonLocalAnchor> &nlas,
                            StringAttr mod);

  /// Add the nla to the module. This ensures that the list of NLAs that the
  /// module participates in is updated. This will be required if `mod` is added
  /// to the namepath of `nla`.
  void addNLAtoModule(NonLocalAnchor nla, StringAttr mod) {
    nodeMap[mod].push_back(nla);
  }

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
