//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PopulateInstanceChoiceSymbols pass, which populates
// globally unique instance macros for all instance choice operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-populate-instance-choice-symbols"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_POPULATEINSTANCECHOICESYMBOLS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
class PopulateInstanceChoiceSymbolsPass
    : public impl::PopulateInstanceChoiceSymbolsBase<
          PopulateInstanceChoiceSymbolsPass> {
public:
  void runOnOperation() override;

private:
  /// Assign a unique instance macro symbol to the given instance choice
  /// operation. Returns the assigned symbol, or nullptr if the operation
  /// already has a symbol.
  FlatSymbolRefAttr assignSymbol(InstanceChoiceOp op);

  /// The namespace associated with the circuit.  This is lazily constructed
  /// using `getNamespace`.
  std::optional<CircuitNamespace> circuitNamespace;
  CircuitNamespace &getNamespace() {
    if (!circuitNamespace)
      circuitNamespace = CircuitNamespace(getOperation());
    return *circuitNamespace;
  }
};
} // namespace

FlatSymbolRefAttr
PopulateInstanceChoiceSymbolsPass::assignSymbol(InstanceChoiceOp op) {
  // Skip if already has an instance macro.
  if (op.getInstanceMacroAttr())
    return nullptr;

  // Get the parent module name.
  auto parentModule = op->getParentOfType<FModuleLike>();

  // Get the option name.
  auto optionName = op.getOptionNameAttr();

  // Generate the instance macro name.
  // This is not public API and can be generated in any way as long as it's
  // unique.
  SmallString<128> instanceMacroName;
  {
    llvm::raw_svector_ostream os(instanceMacroName);
    os << "__target_" << optionName.getValue() << "_"
       << parentModule.getModuleName() << "_" << op.getInstanceName();
  }

  // Ensure global uniqueness using CircuitNamespace.
  auto uniqueName = StringAttr::get(op.getContext(),
                                    getNamespace().newName(instanceMacroName));
  auto instanceMacro = FlatSymbolRefAttr::get(uniqueName);
  op.setInstanceMacroAttr(instanceMacro);

  LLVM_DEBUG(llvm::dbgs() << "Assigned instance macro '" << uniqueName
                          << "' to instance choice '" << op.getInstanceName()
                          << "' in module '" << parentModule.getModuleName()
                          << "'\n");

  return instanceMacro;
}

void PopulateInstanceChoiceSymbolsPass::runOnOperation() {
  auto circuit = getOperation();
  auto &instanceGraph = getAnalysis<InstanceGraph>();

  OpBuilder builder(circuit.getContext());
  builder.setInsertionPointToStart(circuit.getBodyBlock());

  llvm::DenseSet<StringAttr> createdMacros;
  bool changed = false;

  // Iterate through all instance choices.
  instanceGraph.walkPostOrder([&](igraph::InstanceGraphNode &node) {
    auto module = dyn_cast<FModuleLike>(node.getModule().getOperation());
    if (!module)
      return;

    for (auto *record : node) {
      auto op = record->getInstance<InstanceChoiceOp>();
      if (!op)
        continue;

      auto instanceMacro = assignSymbol(op);
      if (!instanceMacro)
        continue;
      changed = true;
      // Create macro declaration only if we haven't created it yet.
      if (createdMacros.insert(instanceMacro.getAttr()).second)
        sv::MacroDeclOp::create(builder, circuit.getLoc(),
                                instanceMacro.getAttr());
    }
  });

  circuitNamespace.reset();
  if (!changed)
    return markAllAnalysesPreserved();

  markAnalysesPreserved<InstanceGraph, InstanceInfo>();
}
