//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements generation of RTG instruction-related methods such as
// RegisterAllocationOpInterface methods based on SourceReg and DestReg
// decorators.
//
//===----------------------------------------------------------------------===//

#include "RTGInstructionFormat.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Decorators
//===----------------------------------------------------------------------===//

namespace {
// Helper class to represent a register effect decorator.
class RegisterEffect : public Operator::VariableDecorator {
public:
  StringRef getEffectName() const { return def->getValueAsString("effect"); }

  static bool classof(const Operator::VariableDecorator *var) {
    return var->getDef().isSubClassOf("RegisterEffect");
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

// Generate the `isSourceRegister` and `isDestinationRegister` methods for an
// operation based on its `SourceReg` and `DestReg` decorators.
static void genRegisterAllocationMethodsForOp(const Record *opDef,
                                              raw_ostream &os) {
  Operator op(opDef);

  // Check if this op implements RegisterAllocationOpInterface
  bool hasInterface = false;
  for (const auto &trait : op.getTraits()) {
    if (auto *iTrait = dyn_cast<InterfaceTrait>(&trait)) {
      std::string traitName = iTrait->getFullyQualifiedTraitName();
      if (traitName.find("circt::rtg::RegisterAllocationOpInterface") !=
          std::string::npos) {
        hasInterface = true;
        break;
      }
    }
  }

  if (!hasInterface)
    return;

  // Collect source and destination register indices
  SmallVector<unsigned, 4> sourceIndices;
  SmallVector<unsigned, 4> destIndices;

  for (auto [i, arg] : llvm::enumerate(op.getArgs())) {
    auto *operand = dyn_cast<NamedTypeConstraint *>(arg);
    if (!operand)
      continue;

    for (auto decorator : op.getArgDecorators(i)) {
      if (RegisterEffect *effect = dyn_cast<RegisterEffect>(&decorator)) {
        StringRef effectName = effect->getEffectName();
        if (effectName == "SourceReg") {
          sourceIndices.push_back(i);
        } else if (effectName == "DestReg") {
          destIndices.push_back(i);
        }
      }
    }
  }

  if (sourceIndices.empty() && destIndices.empty() && op.getNumArgs() > 0)
    return;

  StringRef className = op.getCppClassName();

  // Generate isSourceRegister method
  os << "bool " << className << "::isSourceRegister(unsigned index) {\n";
  if (sourceIndices.empty()) {
    os << "  return false;\n";
  } else {
    os << "  return ";
    for (unsigned i = 0; i < sourceIndices.size(); ++i) {
      if (i > 0)
        os << " || ";
      os << "index == " << sourceIndices[i];
    }
    os << ";\n";
  }
  os << "}\n\n";

  // Generate isDestinationRegister method
  os << "bool " << className << "::isDestinationRegister(unsigned index) {\n";
  if (destIndices.empty()) {
    os << "  return false;\n";
  } else {
    os << "  return ";
    for (unsigned i = 0; i < destIndices.size(); ++i) {
      if (i > 0)
        os << " || ";
      os << "index == " << destIndices[i];
    }
    os << ";\n";
  }
  os << "}\n\n";
}

static bool genRTGInstructionMethods(const RecordKeeper &records,
                                     raw_ostream &os) {
  llvm::emitSourceFileHeader("RTG Instruction Method Implementations", os,
                             records);

  for (const Record *opDef : records.getAllDerivedDefinitions("Op")) {
    genRegisterAllocationMethodsForOp(opDef, os);

    if (opDef->isSubClassOf("ISAInstructionFormat"))
      circt::tblgen::genInstructionPrintMethods(opDef, os);
  }

  return false;
}

// Generator registration for RTG instruction-related methods.
static mlir::GenRegistration
    genRTGInstructionMethodsReg("gen-rtg-instruction-methods",
                                "Generate RTG instruction-related methods from "
                                "decorators",
                                genRTGInstructionMethods);
