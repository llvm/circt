//===- LegalNames.cpp - SV/RTL name legalization analysis -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis pass establishes legalized names for SV/RTL operations that are
// safe to use in SV output.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/SV/SVAnalyses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;
using namespace sv;
using namespace rtl;

//===----------------------------------------------------------------------===//
// Name Lookup
//===----------------------------------------------------------------------===//

/// Return the legalized name for an operation or assert if there is none.
StringRef LegalNamesAnalysis::getOperationName(Operation *op) const {
  auto nameIt = operationNames.find(op);
  assert(nameIt != operationNames.end() && "expected valid legalized name");
  return nameIt->second;
}

/// Return the legalized name for an argument to an operation or assert if there
/// is none.
StringRef LegalNamesAnalysis::getArgName(Operation *op, size_t argNum) const {
  auto nameIt = argNames.find(std::make_pair(op, argNum));
  assert(nameIt != argNames.end() && "expected valid legalized name");
  return nameIt->second;
}

/// Return the legalized name for a result from an operation or assert if there
/// is none.
StringRef LegalNamesAnalysis::getResultName(Operation *op,
                                            size_t resultNum) const {
  auto nameIt = resultNames.find(std::make_pair(op, resultNum));
  assert(nameIt != resultNames.end() && "expected valid legalized name");
  return nameIt->second;
}

//===----------------------------------------------------------------------===//
// Name Legalization
//===----------------------------------------------------------------------===//

/// Legalize the name of an operation and register it for later retrieval.
StringRef LegalNamesAnalysis::legalizeOperation(Operation *op,
                                                StringAttr name) {
  auto &entry = operationNames[op];
  if (entry.empty())
    entry = legalizeName(name.getValue(), usedNames, nextGeneratedNameID);

  return entry;
}

/// Legalize the name of an argument to an operation, and register it for later
/// retrieval.
StringRef LegalNamesAnalysis::legalizeArg(Operation *op, size_t argNum,
                                          StringAttr name) {
  auto &entry = argNames[std::make_pair(op, argNum)];
  if (entry.empty())
    entry = legalizeName(name.getValue(), usedNames, nextGeneratedNameID);

  return entry;
}

/// Legalize the name of a result from an operation, and register it for later
/// retrieval.
StringRef LegalNamesAnalysis::legalizeResult(Operation *op, size_t resultNum,
                                             StringAttr name) {
  auto &entry = resultNames[std::make_pair(op, resultNum)];
  if (entry.empty())
    entry = legalizeName(name.getValue(), usedNames, nextGeneratedNameID);
  return entry;
}

//===----------------------------------------------------------------------===//
// Operation Analysis
//===----------------------------------------------------------------------===//

/// Construct a lookup table of legalized and legalized names for an operation.
///
/// You will generally want to use \c getAnalysis<LegalNamesAnalysis>() and
/// \c getChildAnalysis<LegalNamesAnalysis>(op) inside your pass.
LegalNamesAnalysis::LegalNamesAnalysis(Operation *op) {
  if (auto op2 = dyn_cast<mlir::ModuleOp>(op)) {
    analyze(op2);
  } else if (auto op2 = dyn_cast<RTLModuleOp>(op)) {
    analyze(op2);
  } else if (auto op2 = dyn_cast<RTLModuleExternOp>(op)) {
    analyze(op2);
  } else if (auto op2 = dyn_cast<InterfaceOp>(op)) {
    analyze(op2);
  }
}

/// Legalize the name of modules and interfaces in an MLIR module.
void LegalNamesAnalysis::analyze(mlir::ModuleOp op) {
  // Register the names of external modules which we cannot rename. This has to
  // occur in a first pass separate from the modules and interfaces which we are
  // actually allowed to rename, in order to ensure that we don't accidentally
  // rename a module that later collides with an extern module.
  for (auto &op : *op.getBody()) {
    if (auto extMod = dyn_cast<RTLModuleExternOp>(op)) {
      legalizeOperation(&op, extMod.getVerilogModuleNameAttr());
    }
  }

  // Legalize modules and interfaces.
  for (auto &op : *op.getBody()) {
    if (auto module = dyn_cast<RTLModuleOp>(op)) {
      legalizeOperation(&op, module.getNameAttr());
    } else if (auto intf = dyn_cast<InterfaceOp>(op)) {
      legalizeOperation(&op, intf.getNameAttr());
    }
  }
}

void LegalNamesAnalysis::analyzeModulePorts(Operation *module) {
  for (const ModulePortInfo &port : getModulePortInfo(module)) {
    if (port.isOutput()) {
      legalizeResult(module, port.argNum, port.name);
    } else {
      legalizeArg(module, port.argNum, port.name);
    }
  }
}

/// Legalize the ports, instances, regs, and wires of an RTL module.
void LegalNamesAnalysis::analyze(rtl::RTLModuleOp op) {
  // Legalize the ports.
  analyzeModulePorts(op);

  // Legalize instances, regs, and wires.
  op.walk([&](Operation *op) {
    if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      legalizeOperation(op, instanceOp.getNameAttr());
    } else if (isa<RegOp>(op) || isa<WireOp>(op)) {
      legalizeOperation(op, op->getAttrOfType<StringAttr>("name"));
    }
  });
}

/// Register the ports of an RTL extern module.
///
/// Note that we explicitly do not legalize the names, as we do not have control
/// over the corresponding module declaration with it being supplied externally.
void LegalNamesAnalysis::analyze(rtl::RTLModuleExternOp op) {
  // Legalize the ports.
  analyzeModulePorts(op);
}

/// Legalize the signals and modports of an SV interface.
void LegalNamesAnalysis::analyze(sv::InterfaceOp op) {
  // TODO: Once interfaces gain ports we'll want to legalize them here as well,
  // pretty much like the RTLModuleOp.

  // Legalize signals and modports.
  for (auto &op : *op.getBodyBlock()) {
    if (isa<InterfaceSignalOp>(op) || isa<InterfaceModportOp>(op)) {
      legalizeOperation(
          &op, op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()));
    }
  }
}
