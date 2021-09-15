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
  assert(topLevelModuleOp && "Required to have a ModuleOp parent.");
  return topLevelModuleOp.lookupSymbol(moduleName());
}

StringAttr InstanceOp::getResultName(size_t idx) {
  return hw::getModuleResultNameAttr(getReferencedModule(), idx);
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

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module = symbolTable.lookupNearestSymbolFrom(*this, moduleNameAttr());
  if (module == nullptr)
    return emitError("Cannot find module definition '") << moduleName() << "'";

  // It must be some sort of module.
  if (!hw::isAnyModule(module))
    return emitError("symbol reference '")
           << moduleName() << "' isn't a module";
  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/MSFT/MSFT.cpp.inc"
