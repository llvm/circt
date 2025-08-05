//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test for priority cuts implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_TESTPRIORITYCUTS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace circt::synth;

#define DEBUG_TYPE "synth-test-priority-cuts"

//===----------------------------------------------------------------------===//
// Test Priority Cuts Pass
//===----------------------------------------------------------------------===//

namespace {

StringRef getTestVariableName(Value value) {
  if (auto *op = value.getDefiningOp()) {
    if (auto name = op->getAttrOfType<StringAttr>("sv.namehint"))
      return name.getValue();
    return "<unknown>";
  }

  auto blockArg = cast<BlockArgument>(value);
  auto hwOp = dyn_cast<hw::HWModuleOp>(blockArg.getOwner()->getParentOp());
  if (!hwOp)
    return "<unknown>";
  return hwOp.getInputName(blockArg.getArgNumber());
}

struct TestPriorityCutsPass
    : public impl::TestPriorityCutsBase<TestPriorityCutsPass> {
  using TestPriorityCutsBase<TestPriorityCutsPass>::TestPriorityCutsBase;

  void runOnOperation() override {
    auto hwModule = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Running TestPriorityCuts on module: "
                            << hwModule.getName() << "\n");

    // Set up cut rewriter options for testing
    CutRewriterOptions options;
    options.maxCutInputSize = maxCutInputSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    // Currently there is no behavioral difference between timing and area
    // so just test with timing strategy.
    options.strategy = circt::synth::OptimizationStrategyTiming;

    // Create cut enumerator to test priority cuts
    CutEnumerator enumerator(options);
    llvm::outs() << "Enumerating cuts for module: " << hwModule.getModuleName()
                 << "\n";

    // Test cut enumeration - this will generate and prioritize cuts
    auto result = enumerator.enumerateCuts(hwModule);

    if (failed(result)) {
      LLVM_DEBUG(llvm::dbgs() << "Cut enumeration failed\n");
      signalPassFailure();
      return;
    }

    // Get the enumerated cuts and report statistics
    auto cutSets = enumerator.takeVector();
    for (auto &[value, cutSetPtr] : cutSets) {
      auto &cutSet = *cutSetPtr;
      llvm::outs() << getTestVariableName(value) << " "
                   << cutSet.getCuts().size() << " cuts:";
      for (const Cut &cut : cutSet.getCuts()) {
        llvm::outs() << " {";
        llvm::interleaveComma(cut.inputs, llvm::outs(), [&](Value input) {
          llvm::outs() << getTestVariableName(input);
        });
        llvm::outs() << "}"
                     << "@t" << cut.getTruthTable().table.getZExtValue() << ""
                     << "d" << cut.getDepth();
      }
      llvm::outs() << "\n";
    }
    llvm::outs() << "Cut enumeration completed successfully\n";

    // This is a test pass, so we don't modify the IR
    markAllAnalysesPreserved();
  }
};

} // namespace
