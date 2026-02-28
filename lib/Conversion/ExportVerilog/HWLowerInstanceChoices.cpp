//===- HWLowerInstanceChoices.cpp - IR Prepass for Emitter ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the "prepare" pass that walks the IR before the emitter
// gets involved.  This allows us to do some transformations that would be
// awkward to implement inline in the emitter.
//
// NOTE: This file covers the preparation phase of `ExportVerilog` which mainly
// legalizes the IR and makes adjustments necessary for emission. This is the
// place to mutate the IR if emission needs it. The IR cannot be modified during
// emission itself, which happens in parallel.
//
//===----------------------------------------------------------------------===//

#include "ExportVerilogInternals.h"
#include "circt/Conversion/ExportVerilog.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace circt {
#define GEN_PASS_DEF_HWLOWERINSTANCECHOICES
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace sv;
using namespace ExportVerilog;

namespace {

struct HWLowerInstanceChoicesPass
    : public circt::impl::HWLowerInstanceChoicesBase<
          HWLowerInstanceChoicesPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

LogicalResult ExportVerilog::lowerHWInstanceChoices(mlir::ModuleOp module) {
  // Collect all instance choices & symbols.
  SmallVector<InstanceChoiceOp> instances;
  SymbolCache symCache;
  for (Operation &op : *module.getBody()) {
    if (auto sym = dyn_cast<SymbolOpInterface>(&op))
      symCache.addSymbol(sym);

    if (auto module = dyn_cast<HWModuleOp>(&op))
      module.walk([&](InstanceChoiceOp inst) { instances.push_back(inst); });
  }

  // Build a namespace to generate unique macro names.
  Namespace ns;
  ns.add(symCache);

  auto declBuilder = OpBuilder::atBlockBegin(module.getBody());
  for (InstanceChoiceOp inst : instances) {
    auto parent = inst->getParentOfType<HWModuleOp>();

    auto defaultModuleOp = cast<HWModuleLike>(
        symCache.getDefinition(inst.getDefaultModuleNameAttr()));

    // Generate a macro name to describe the target of this instance.
    SmallString<128> name;
    {
      llvm::raw_svector_ostream os(name);
      os << "__circt_choice_" << parent.getName() << "_"
         << inst.getInstanceName();
    }

    auto symName = ns.newName(name);
    auto symNameAttr = declBuilder.getStringAttr(symName);
    auto symRef = FlatSymbolRefAttr::get(symNameAttr);
    MacroDeclOp::create(declBuilder, inst.getLoc(), symNameAttr,
                        /*args=*/ArrayAttr{},
                        /*verilogName=*/StringAttr{});

    // This pass now generates the macros and attaches them to the instance
    // choice as an attribute. As a better solution, this pass should be moved
    // out of the umbrella of ExportVerilog and it should lower the `hw`
    // instance choices to a better SV-level representation of the operation.
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    sv::IfDefOp::create(
        builder, symName, [&] {},
        [&] {
          sv::MacroDefOp::create(
              builder, symRef, builder.getStringAttr("{{0}}"),
              builder.getArrayAttr(
                  {FlatSymbolRefAttr::get(defaultModuleOp.getNameAttr())}));
        });
    inst->setAttr("hw.choiceTarget", symRef);
  }

  return success();
}

void HWLowerInstanceChoicesPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (failed(lowerHWInstanceChoices(module)))
    signalPassFailure();
}
