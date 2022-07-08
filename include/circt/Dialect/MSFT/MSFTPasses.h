//===- MSFTPasses.h - Common code for passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MSFT_MSFTPASSES_H
#define CIRCT_DIALECT_MSFT_MSFTPASSES_H

#include "circt/Dialect/MSFT/MSFTOps.h"

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"

namespace circt {
namespace msft {
void registerMSFTPasses();

/// A set of methods which are broadly useful in a number of dialects.
struct PassCommon {
protected:
  SymbolCache topLevelSyms;
  DenseMap<Operation *, SmallVector<hw::HWInstanceLike, 1>>
      moduleInstantiations;

  LogicalResult verifyInstances(ModuleOp topMod);

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
} // namespace msft
} // namespace circt

#endif // CIRCT_DIALECT_MSFT_MSFTPASSES_H
