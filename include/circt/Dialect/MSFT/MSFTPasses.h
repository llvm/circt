//===- MSFTPasses.h - Common code for passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_MSFTPASSES_H
#define CIRCT_DIALECT_MSFT_MSFTPASSES_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/MSFT/MSFTOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace msft {

/// A set of methods which are broadly useful in a number of dialects.
struct PassCommon {
protected:
  SymbolCache topLevelSyms;
  DenseMap<Operation *, SmallVector<igraph::InstanceOpInterface, 1>>
      moduleInstantiations;

  // Find all the modules and use the partial order of the instantiation DAG
  // to sort them. If we use this order when "bubbling" up operations, we
  // guarantee one-pass completeness. As a side-effect, populate the module to
  // instantiation sites mapping.
  //
  // Assumption (unchecked): there is not a cycle in the instantiation graph.
  void getAndSortModules(ModuleOp topMod,
                         SmallVectorImpl<hw::HWModuleLike> &mods);
  void getAndSortModulesVisitor(hw::HWModuleLike mod,
                                SmallVectorImpl<hw::HWModuleLike> &mods,
                                DenseSet<Operation *> &modsSeen);
};

#define GEN_PASS_DECL
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"

std::unique_ptr<mlir::Pass> createLowerInstancesPass();
std::unique_ptr<mlir::Pass> createLowerConstructsPass();
std::unique_ptr<mlir::Pass> createExportTclPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"

} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_MSFTPASSES_H
