//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helper functions for analyzing and extracting type
// information from MLIR TableGen operation definitions for RTG.
//
//===----------------------------------------------------------------------===//

#include "RTGInstructionUtils.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Trait.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir::tblgen;
using namespace circt;
using namespace circt::tblgen::rtg;

//===----------------------------------------------------------------------===//
// Decorator Functions
//===----------------------------------------------------------------------===//

static bool hasRegisterEffect(const mlir::tblgen::Operator &op,
                              unsigned argIndex, StringRef effectName) {
  for (auto decorator : op.getArgDecorators(argIndex)) {
    if (auto *effect = dyn_cast<RegisterEffect>(&decorator)) {
      if (effect->getEffectName() == effectName)
        return true;
    }
  }
  return false;
}

bool tblgen::rtg::isSourceRegister(const mlir::tblgen::Operator &op,
                                   unsigned argIndex) {
  return hasRegisterEffect(op, argIndex, "SourceReg");
}

bool tblgen::rtg::isDestinationRegister(const mlir::tblgen::Operator &op,
                                        unsigned argIndex) {
  return hasRegisterEffect(op, argIndex, "DestReg");
}

//===----------------------------------------------------------------------===//
// Type Checking Functions
//===----------------------------------------------------------------------===//

static bool checkType(const Record &typeRec, StringRef name) {
  return typeRec.getName() == name || typeRec.isSubClassOf(name);
}

// NOLINTNEXTLINE(misc-no-recursion)
void tblgen::rtg::classifyOperandType(const Record &typeRec,
                                      OperandTypeSet &kinds) {
  // Look through OpVariable's to their constraint.
  if (typeRec.isSubClassOf("OpVariable"))
    return classifyOperandType(*typeRec.getValueAsDef("constraint"), kinds);

  if (checkType(typeRec, "LabelType"))
    return kinds.appendAndUnique(Label());
  if (checkType(typeRec, "ImmediateType"))
    return kinds.appendAndUnique(AnyImmediate());
  if (checkType(typeRec, "ImmediateOfWidth"))
    return kinds.appendAndUnique(Immediate(typeRec.getValueAsInt("immWidth")));
  if (checkType(typeRec, "MemoryType"))
    return kinds.appendAndUnique(AnyMemory());
  if (checkType(typeRec, "MemoryTypeWithAddressWidth"))
    return kinds.appendAndUnique(Memory(typeRec.getValueAsInt("addressWidth")));
  if (checkType(typeRec, "RegisterTypeBase"))
    return kinds.appendAndUnique(
        Register(typeRec.getValueAsString("pythonName"),
                 typeRec.getValueAsString("pythonTypeName"),
                 typeRec.getValueAsInt("binaryEncodingWidth")));

  if (checkType(typeRec, "AnyTypeOf")) {
    auto allowedTypes = typeRec.getValueAsListOfDefs("allowedTypes");
    for (const Record *allowedType : allowedTypes)
      classifyOperandType(*allowedType, kinds);
  }
}

const NamedTypeConstraint *tblgen::rtg::findOperandByName(const Operator &op,
                                                          StringRef name) {
  for (auto arg : op.getArgs()) {
    if (auto *operand = dyn_cast<NamedTypeConstraint *>(arg)) {
      if (operand->name == name)
        return operand;
    }
  }
  return nullptr;
}

bool tblgen::rtg::hasOperatorInterface(const Operator &op,
                                       StringRef interfaceName) {
  for (const auto &trait : op.getTraits()) {
    if (auto *iTrait = dyn_cast<InterfaceTrait>(&trait)) {
      std::string traitName = iTrait->getFullyQualifiedTraitName();
      if (traitName.find(interfaceName.str()) != std::string::npos)
        return true;
    }
  }
  return false;
}
