//===- InnerSymbolDCE.cpp - Delete Unused Inner Symbols----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass removes inner symbols which have no uses.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-inner-symbol-dce"

using namespace mlir;
using namespace circt;
using namespace firrtl;
using namespace hw;

struct InnerSymbolDCEPass : public InnerSymbolDCEBase<InnerSymbolDCEPass> {
  void runOnOperation() override;

private:
  void findInnerRefs(Attribute attr);
  void insertInnerRef(InnerRefAttr innerRef);
  void removeInnerSyms(FModuleOp op);

  DenseSet<std::pair<StringAttr, StringAttr>> innerRefs;
};

/// Find all InnerRefAttrs inside a given Attribute.
void InnerSymbolDCEPass::findInnerRefs(Attribute attr) {
  // Check if this Attribute is an InnerRefAttr.
  if (auto innerRef = dyn_cast<InnerRefAttr>(attr)) {
    insertInnerRef(innerRef);
    return;
  }

  // Check if any sub-Attributes are InnerRefAttrs.
  if (auto subElementAttr = dyn_cast<SubElementAttrInterface>(attr))
    subElementAttr.walkSubAttrs([&](Attribute subAttr) {
      if (auto innerRef = dyn_cast<InnerRefAttr>(subAttr))
        insertInnerRef(innerRef);
    });
}

/// Add an InnerRefAttr to the set of all InnerRefAttrs.
void InnerSymbolDCEPass::insertInnerRef(InnerRefAttr innerRef) {
  StringAttr moduleName = innerRef.getModule();
  StringAttr symName = innerRef.getName();

  auto [iter, inserted] = innerRefs.insert({moduleName, symName});
  if (!inserted)
    return;

  ++numSymbolsFound;

  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": found " << moduleName
                          << "::" << symName << '\n';);
}

/// Remove all InnerSymAttrs within the FModuleOp that are dead code.
void InnerSymbolDCEPass::removeInnerSyms(FModuleOp moduleOp) {
  // Walk all ops in the FModuleOp.
  moduleOp.walk([&](Operation *op) {
    // Check if the op has an InnerSymAttr.
    auto innerSym = op->getAttrOfType<InnerSymAttr>("inner_sym");
    if (!innerSym)
      return;

    assert(moduleOp == op->getParentOfType<FModuleOp>() &&
           "ops with inner_sym must be inside an FModuleOp");

    // Check if the InnerSymAttr was found as part of any InnerRefAttr.
    auto moduleName = moduleOp.moduleNameAttr();
    auto symName = innerSym.getSymName();
    if (innerRefs.contains({moduleName, symName}))
      return;

    // If not, it's dead code.
    op->removeAttr("inner_sym");

    ++numSymbolsRemoved;

    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << ": removed " << moduleName
                            << "::" << symName << '\n';);
  });
}

void InnerSymbolDCEPass::runOnOperation() {
  // Run on the top-level ModuleOp to include any VerbatimOps that aren't
  // wrapped in a CircuitOp.
  ModuleOp topModule = getOperation();

  // Traverse the entire IR once.
  SmallVector<FModuleOp> modules;
  topModule.walk([&](Operation *op) {
    // Find all InnerRefAttrs.
    for (NamedAttribute namedAttr : op->getAttrs())
      findInnerRefs(namedAttr.getValue());

    // Collect all FModuleOps.
    if (auto moduleOp = dyn_cast<FModuleOp>(op))
      modules.push_back(moduleOp);
  });

  // Traverse all FModuleOps in parallel, removing any InnerSymAttrs that are
  // dead code.
  parallelForEach(&getContext(), modules,
                  [&](FModuleOp moduleOp) { removeInnerSyms(moduleOp); });
}

std::unique_ptr<mlir::Pass> circt::firrtl::createInnerSymbolDCEPass() {
  return std::make_unique<InnerSymbolDCEPass>();
}
