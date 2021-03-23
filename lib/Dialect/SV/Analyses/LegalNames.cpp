//===- LegalNames.cpp - SV/RTL name legalization analysis -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This analysis pass establishes sanitized names for SV/RTL operations that are
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

/// Lookup the sanitized name for an operation.
Optional<StringRef>
LegalNamesAnalysis::lookupOperationName(Operation *op) const {
  auto it = operationNames.find(op);
  return it != operationNames.end() ? it->second : Optional<StringRef>();
}

/// Lookup the sanitized name for an argument to an operation.
Optional<StringRef> LegalNamesAnalysis::lookupArgName(Operation *op,
                                                      size_t argNum) const {
  auto it = argNames.find(std::make_pair(op, argNum));
  return it != argNames.end() ? it->second : Optional<StringRef>();
}

/// Lookup the sanitized name for a result from an operation.
Optional<StringRef>
LegalNamesAnalysis::lookupResultName(Operation *op, size_t resultNum) const {
  auto it = resultNames.find(std::make_pair(op, resultNum));
  return it != resultNames.end() ? it->second : Optional<StringRef>();
}

/// Return the sanitized name for an operation or assert if there is none.
StringRef LegalNamesAnalysis::getOperationName(Operation *op) const {
  auto name = lookupOperationName(op);
  assert(name && "expected valid sanitized name");
  return *name;
}

/// Return the sanitized name for an argument to an operation or assert if there
/// is none.
StringRef LegalNamesAnalysis::getArgName(Operation *op, size_t argNum) const {
  auto name = lookupArgName(op, argNum);
  assert(name && "expected valid sanitized name");
  return *name;
}

/// Return the sanitized name for a result from an operation or assert if there
/// is none.
StringRef LegalNamesAnalysis::getResultName(Operation *op,
                                            size_t resultNum) const {
  auto name = lookupResultName(op, resultNum);
  assert(name && "expected valid sanitized name");
  return *name;
}

/// Return the set of used names.
const llvm::StringSet<> &LegalNamesAnalysis::getUsedNames() const {
  return usedNames;
}

//===----------------------------------------------------------------------===//
// Name Registration and Sanitization
//===----------------------------------------------------------------------===//

/// Register a sanitized name for an operation.
///
/// Updates the set of used names.
void LegalNamesAnalysis::registerOperation(Operation *op, StringRef name) {
  usedNames.insert(name);
  operationNames.insert(std::make_pair(op, name));
}

/// Register a sanitized name for an argument to an operation.
///
/// Updates the set of used names.
void LegalNamesAnalysis::registerArg(Operation *op, size_t argNum,
                                     StringRef name) {
  usedNames.insert(name);
  argNames.insert(std::make_pair(std::make_pair(op, argNum), name));
}

/// Register a sanitized name for a result from an operation.
///
/// Updates the set of used names.
void LegalNamesAnalysis::registerResult(Operation *op, size_t resultNum,
                                        StringRef name) {
  usedNames.insert(name);
  resultNames.insert(std::make_pair(std::make_pair(op, resultNum), name));
}

/// Sanitize the name of an operation and register it for later retrieval.
StringRef LegalNamesAnalysis::sanitizeOperation(Operation *op,
                                                StringAttr name) {
  auto it = operationNames.find(op);
  if (it != operationNames.end()) {
    return it->second;
  } else {
    auto updatedName =
        sanitizeName(name.getValue(), usedNames, nextGeneratedNameID);
    registerOperation(op, updatedName);
    return updatedName;
  }
}

/// Sanitize the name of an argument to an operation, and register it for later
/// retrieval.
StringRef LegalNamesAnalysis::sanitizeArg(Operation *op, size_t argNum,
                                          StringAttr name) {
  auto it = argNames.find(std::make_pair(op, argNum));
  if (it != argNames.end()) {
    return it->second;
  } else {
    auto updatedName =
        sanitizeName(name.getValue(), usedNames, nextGeneratedNameID);
    registerArg(op, argNum, updatedName);
    return updatedName;
  }
}

/// Sanitize the name of a result from an operation, and register it for later
/// retrieval.
StringRef LegalNamesAnalysis::sanitizeResult(Operation *op, size_t resultNum,
                                             StringAttr name) {
  auto it = resultNames.find(std::make_pair(op, resultNum));
  if (it != resultNames.end()) {
    return it->second;
  } else {
    auto updatedName =
        sanitizeName(name.getValue(), usedNames, nextGeneratedNameID);
    registerResult(op, resultNum, updatedName);
    return updatedName;
  }
}

//===----------------------------------------------------------------------===//
// Operation Analysis
//===----------------------------------------------------------------------===//

/// Construct a lookup table of sanitized and legalized names for an operation.
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

/// Sanitize the name of modules and interfaces in an MLIR module.
void LegalNamesAnalysis::analyze(mlir::ModuleOp op) {
  // Register the names of external modules which we cannot rename. This has to
  // occur in a first pass separate from the modules and interfaces which we are
  // actually allowed to rename, in order to ensure that we don't accidentally
  // rename a module that later collides with an extern module.
  for (auto &op : *op.getBody()) {
    if (auto extMod = dyn_cast<RTLModuleExternOp>(op)) {
      registerOperation(&op, extMod.getVerilogModuleNameAttr().getValue());
    }
  }

  // Sanitize modules and interfaces.
  for (auto &op : *op.getBody()) {
    if (auto module = dyn_cast<RTLModuleOp>(op)) {
      sanitizeOperation(&op, module.getNameAttr());
    } else if (auto intf = dyn_cast<InterfaceOp>(op)) {
      sanitizeOperation(&op, intf.getNameAttr());
    }
  }
}

/// Sanitize the ports, instances, regs, and wires of an RTL module.
void LegalNamesAnalysis::analyze(rtl::RTLModuleOp op) {
  auto moduleType = rtl::getModuleType(op);

  // Sanitize the inputs.
  for (unsigned i = 0, e = moduleType.getInputs().size(); i < e; ++i) {
    sanitizeArg(op, i, op.getArgAttrOfType<StringAttr>(i, "rtl.name"));
  }

  // Sanitize the results.
  for (unsigned i = 0, e = moduleType.getResults().size(); i < e; ++i) {
    sanitizeResult(op, i, op.getResultAttrOfType<StringAttr>(i, "rtl.name"));
  }

  // Sanitize instances, regs, and wires.
  op.walk([&](Operation *op) {
    if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      sanitizeOperation(op, instanceOp.getNameAttr());
    } else if (isa<RegOp>(op) || isa<WireOp>(op)) {
      sanitizeOperation(op, op->getAttrOfType<StringAttr>("name"));
    }
  });
}

/// Register the ports of an RTL extern module.
///
/// Note that we explicitly do not sanitize the names, as we do not have control
/// over the corresponding module declaration with it being supplied externally.
void LegalNamesAnalysis::analyze(rtl::RTLModuleExternOp op) {
  auto moduleType = rtl::getModuleType(op);

  // Register the inputs.
  for (unsigned i = 0, e = moduleType.getInputs().size(); i < e; ++i) {
    registerArg(op, i,
                op.getArgAttrOfType<StringAttr>(i, "rtl.name").getValue());
  }

  // Register the results.
  for (unsigned i = 0, e = moduleType.getResults().size(); i < e; ++i) {
    registerResult(
        op, i, op.getResultAttrOfType<StringAttr>(i, "rtl.name").getValue());
  }
}

/// Sanitize the signals and modports of an SV interface.
void LegalNamesAnalysis::analyze(sv::InterfaceOp op) {
  // TODO: Once interfaces gain ports we'll want to sanitize them here as well,
  // pretty much like the RTLModuleOp.

  // Sanitize signals and modports.
  for (auto &op : *op.getBodyBlock()) {
    if (isa<InterfaceSignalOp>(op) || isa<InterfaceModportOp>(op)) {
      sanitizeOperation(
          &op, op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()));
    }
  }
}
