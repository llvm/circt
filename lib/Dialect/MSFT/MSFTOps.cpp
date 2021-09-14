//===- MSFTOps.cpp - Implement MSFT dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSFT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *InstanceOp::getReferencedModule() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  assert(topLevelModuleOp &&
         "This instance references a non-existent ModuleOp");

  return topLevelModuleOp.lookupSymbol(moduleName());
}

StringAttr InstanceOp::getResultName(size_t idx) {
  auto *module = getReferencedModule();
  if (!module)
    return {};

  return hw::getModuleResultNameAttr(module, idx);
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Provide default names for instance results.
  std::string name = instanceName().str() + ".";
  size_t baseNameLen = name.size();

  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    name.resize(baseNameLen);
    StringAttr resNameAttr = getResultName(i);
    if (resNameAttr)
      name += resNameAttr.getValue().str();
    else
      name += std::to_string(i);
    setNameFn(getResult(i), name);
  }
}

#define GET_OP_CLASSES
#include "circt/Dialect/MSFT/MSFT.cpp.inc"
