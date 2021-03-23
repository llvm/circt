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
#include "circt/Dialect/SV/SVAnalyses.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Translation/ExportVerilog.h"

using namespace circt;
using namespace sv;
using namespace rtl;

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
  for (unsigned i = 0; i < inputs.size(); ++i) {
    auto oldName = module.getArgAttrOfType<StringAttr>(i, "rtl.name");
    auto newName = localNames.getArgName(module, i);
    if (oldName.getValue() != newName) {
      module.setArgAttr(i, "rtl.name",
                        StringAttr::get(module.getContext(), newName));
      anythingChanged = true;
    }
  }

  // Rename the results.
  for (unsigned i = 0; i < results.size(); ++i) {
    auto oldName = module.getResultAttrOfType<StringAttr>(i, "rtl.name");
    auto newName = localNames.getResultName(module, i);
    if (oldName.getValue() != newName) {
      module.setResultAttr(i, "rtl.name",
                           StringAttr::get(module.getContext(), newName));
      anythingChanged = true;
    }
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
