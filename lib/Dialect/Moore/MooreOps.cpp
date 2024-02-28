//===- MooreOps.cpp - Implement the Moore operations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/IR/Builders.h"

using namespace circt;
using namespace circt::moore;

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module =
      symbolTable.lookupNearestSymbolFrom(*this, getModuleNameAttr());
  if (module == nullptr)
    return emitError("unknown symbol name '") << getModuleName() << "'";

  // It must be some sort of module.
  if (!isa<SVModuleOp>(module))
    return emitError("symbol '")
           << getModuleName()
           << "' must reference a 'moore.module', but got a '"
           << module->getName() << "' instead";

  return success();
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void VariableOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), getName());
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Domain domain = Domain::TwoValued;
  unsigned size = 0;
  for (auto operand : operands) {
    auto type = operand.getType().cast<UnpackedType>().getSimpleBitVector();
    if (type.domain == Domain::FourValued)
      domain = Domain::FourValued;
    size += type.size;
  }
  results.push_back(
      SimpleBitVectorType(domain, Sign::Unsigned, size).getType(context));
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Moore/Moore.cpp.inc"
#include "circt/Dialect/Moore/MooreEnums.cpp.inc"
