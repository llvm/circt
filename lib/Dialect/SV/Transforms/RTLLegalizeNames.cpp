//===- RTLLegalizeNames.cpp - RTL Name Legalization Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass renames modules and variables to avoid conflicts
// with keywords and other declarations.
//
//===----------------------------------------------------------------------===//

#include "SVPassDetail.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Translation/ExportVerilog.h"

using namespace circt;
using namespace sv;
using namespace rtl;

namespace {

/// A lookup table for legalized and legalized output names.
///
/// This analysis establishes a mapping from the name of modules, interfaces,
/// and various other operations to a legalized version that is properly
/// uniquified and does not collide with any keywords.
struct LegalNamesAnalysis {
  LegalNamesAnalysis(mlir::Operation *op);

  /// Return the legalized name for an operation or assert if there is none.
  StringRef getOperationName(Operation *op) const;
  /// Return the legalized name for an argument to an operation or assert if
  /// there is none.
  StringRef getArgName(Operation *op, size_t argNum) const;
  /// Return the legalized name for a result from an operation or assert if
  /// there is none.
  StringRef getResultName(Operation *op, size_t resultNum) const;

private:
  /// Mapping from operations to their legalized name. Used for module, extern
  /// module, and interface operations.
  llvm::DenseMap<Operation *, StringRef> operationNames;

  /// Mapping from operation arguments to their legalized name. Used for module
  /// input ports.
  llvm::DenseMap<std::pair<Operation *, size_t>, StringRef> argNames;

  /// Mapping from operation results to their legalized name. Used for module
  /// output ports.
  llvm::DenseMap<std::pair<Operation *, size_t>, StringRef> resultNames;

  /// Set of used names, to ensure uniqueness.
  llvm::StringSet<> usedNames;

  /// Numeric suffix used as uniquification agent when resolving conflicts.
  size_t nextGeneratedNameID = 0;

  StringRef legalizeOperation(Operation *op, StringAttr name);
  StringRef legalizeArg(Operation *op, size_t argNum, StringAttr name);
  StringRef legalizeResult(Operation *op, size_t resultNum, StringAttr name);

  void analyzeModulePorts(Operation *module);
  void analyze(mlir::ModuleOp op);
  void analyze(rtl::RTLModuleOp op);
  void analyze(rtl::RTLModuleExternOp op);
  void analyze(sv::InterfaceOp op);
};
} // end anonymous namespace

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

//===----------------------------------------------------------------------===//
// RTLLegalizeNamesPass
//===----------------------------------------------------------------------===//

namespace {
struct RTLLegalizeNamesPass
    : public sv::RTLLegalizeNamesBase<RTLLegalizeNamesPass> {
  void runOnOperation() override;

private:
  bool anythingChanged;

  void runOnModule(rtl::RTLModuleOp module);
  void runOnInterface(sv::InterfaceOp intf, mlir::SymbolUserMap &symbolUsers);
};
} // end anonymous namespace

void RTLLegalizeNamesPass::runOnOperation() {
  anythingChanged = false;
  ModuleOp root = getOperation();

  // Analyze the legal names for top-level operations in the MLIR module.
  auto &rootNames = getAnalysis<LegalNamesAnalysis>();

  // Rename modules and interfaces.
  mlir::SymbolTableCollection symbolTable;
  mlir::SymbolUserMap symbolUsers(symbolTable, root);

  for (auto &op : *root.getBody()) {
    if (isa<RTLModuleOp>(op) || isa<InterfaceOp>(op)) {
      auto oldName = SymbolTable::getSymbolName(&op);
      auto newName = rootNames.getOperationName(&op);
      if (oldName != newName) {
        symbolUsers.replaceAllUsesWith(&op, newName);
        SymbolTable::setSymbolName(&op, newName);
        anythingChanged = true;
      }
    }
  }

  // Rename individual operations.
  for (auto &op : *root.getBody()) {
    if (auto module = dyn_cast<RTLModuleOp>(op)) {
      runOnModule(module);
    } else if (auto intf = dyn_cast<InterfaceOp>(op)) {
      runOnInterface(intf, symbolUsers);
    } else if (auto extMod = dyn_cast<RTLModuleExternOp>(op)) {
      auto name = extMod.getVerilogModuleName();
      if (!sv::isNameValid(name)) {
        extMod->emitOpError("with invalid name \"" + name + "\"");
      }
    }
  }

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

void RTLLegalizeNamesPass::runOnModule(rtl::RTLModuleOp module) {
  auto localNames = getChildAnalysis<LegalNamesAnalysis>(module);
  auto moduleType = rtl::getModuleType(module);
  auto inputs = moduleType.getInputs();
  auto results = moduleType.getResults();

  // Rename the inputs.
  bool changedName = false;
  SmallVector<Attribute> names;
  for (size_t i = 0, e = inputs.size(); i != e; ++i) {
    auto oldName = getModuleArgumentNameAttr(module, i);
    auto newName = localNames.getArgName(module, i);
    if (oldName.getValue() == newName)
      names.push_back(oldName);
    else {
      names.push_back(StringAttr::get(module.getContext(), newName));
      changedName = true;
    }
  }
  if (changedName) {
    setModuleArgumentNames(module, names);
    anythingChanged = true;
  }

  changedName = false;
  names.clear();

  // Rename the results if needed.
  for (size_t i = 0, e = results.size(); i != e; ++i) {
    auto oldName = getModuleResultNameAttr(module, i);
    auto newName = localNames.getResultName(module, i);
    if (oldName.getValue() == newName)
      names.push_back(oldName);
    else {
      names.push_back(StringAttr::get(module.getContext(), newName));
      changedName = true;
    }
  }
  if (changedName) {
    setModuleResultNames(module, names);
    anythingChanged = true;
  }

  // Rename the instances, regs, and wires.
  for (auto &op : *module.getBodyBlock()) {
    if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
      auto oldName = instanceOp.getName();
      auto newName = localNames.getOperationName(&op);
      if (oldName != newName) {
        instanceOp.setName(newName);
        anythingChanged = true;
      }
    } else if (isa<RegOp>(op) || isa<WireOp>(op)) {
      auto oldName = op.getAttrOfType<StringAttr>("name").getValue();
      auto newName = localNames.getOperationName(&op);
      if (oldName != newName) {
        op.setAttr("name", StringAttr::get(op.getContext(), newName));
        anythingChanged = true;
      }
    }
  }
}

void RTLLegalizeNamesPass::runOnInterface(sv::InterfaceOp intf,
                                          mlir::SymbolUserMap &symbolUsers) {
  auto localNames = getChildAnalysis<LegalNamesAnalysis>(intf);

  // Rename signals and modports.
  for (auto &op : *intf.getBodyBlock()) {
    if (isa<InterfaceSignalOp>(op) || isa<InterfaceModportOp>(op)) {
      auto oldName = SymbolTable::getSymbolName(&op);
      auto newName = localNames.getOperationName(&op);
      if (oldName != newName) {
        symbolUsers.replaceAllUsesWith(&op, newName);
        SymbolTable::setSymbolName(&op, newName);
        anythingChanged = true;
      }
    }
  }
}

std::unique_ptr<Pass> circt::sv::createRTLLegalizeNamesPass() {
  return std::make_unique<RTLLegalizeNamesPass>();
}
