//===- HWLegalizeModulesPass.cpp - Lower unsupported IR features away -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers away features in the SV/Comb/HW dialects that are
// unsupported by some tools (e.g. multidimensional arrays) as specified by
// LoweringOptions.  This pass is run relatively late in the pipeline in
// preparation for emission.  Any passes run after this (e.g. PrettifyVerilog)
// must be aware they cannot introduce new invalid constructs.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Support/LoweringOptions.h"
#include "mlir/IR/Matchers.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// HWLegalizeModulesPass
//===----------------------------------------------------------------------===//

namespace {
struct HWLegalizeModulesPass
    : public sv::HWLegalizeModulesBase<HWLegalizeModulesPass> {
  void runOnOperation() override;

private:
  void processPostOrder(Block &block);

  bool anythingChanged;

  /// This tells us what language features we're allowed to use in generated
  /// Verilog.
  LoweringOptions options;

  /// This pass will be run on multiple hw.modules, this keeps track of the
  /// contents of LoweringOptions so we don't have to reparse the
  /// LoweringOptions for every hw.module.
  StringAttr lastParsedOptions;
};
} // end anonymous namespace

void HWLegalizeModulesPass::processPostOrder(Block &body) {

  // Walk the block bottom-up, processing the region tree inside out.
  for (auto &op :
       llvm::make_early_inc_range(llvm::reverse(body.getOperations()))) {
    if (op.getNumRegions()) {
      for (auto &region : op.getRegions())
        for (auto &regionBlock : region.getBlocks())
          processPostOrder(regionBlock);
    }

    // If we aren't allowing multi-dimensional arrays, reject the IR as invalid.
    // TODO: We should eventually implement a "lower types" like feature in this
    // pass.
    if (options.disallowPackedArrays) {
      for (auto value : op.getResults()) {
        if (value.getType().isa<hw::ArrayType>()) {
          op.emitError("unsupported packed array expression");
        }
      }
    }
  }
}

void HWLegalizeModulesPass::runOnOperation() {
  hw::HWModuleOp thisModule = getOperation();

  // Parse the lowering options if necessary.
  auto optionsAttr = LoweringOptions::getAttributeFrom(
      cast<ModuleOp>(thisModule->getParentOp()));
  if (optionsAttr != lastParsedOptions) {
    if (optionsAttr)
      options = LoweringOptions(optionsAttr.getValue(), [&](Twine error) {
        thisModule.emitError(error);
      });
    else
      options = LoweringOptions();
    lastParsedOptions = optionsAttr;
  }

  // Keeps track if anything changed during this pass, used to determine if
  // the analyses were preserved.
  anythingChanged = false;

  // Walk the operations in post-order, transforming any that are interesting.
  processPostOrder(*thisModule.getBodyBlock());

  // If we did not change anything in the IR mark all analysis as preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

std::unique_ptr<Pass> circt::sv::createHWLegalizeModulesPass() {
  return std::make_unique<HWLegalizeModulesPass>();
}
