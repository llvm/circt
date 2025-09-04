//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a generic LUT mapper pass using the cut-based rewriting
// framework. It performs technology mapping by converting combinational logic
// networks into implementations using K-input lookup tables (LUTs).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/Transforms/CutRewriter.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/NPNClass.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "synth-generic-lut-mapper"

using namespace circt;
using namespace circt::synth;

namespace circt {
namespace synth {
#define GEN_PASS_DEF_GENERICLUTMAPPER
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

//===----------------------------------------------------------------------===//
// Generic LUT Pattern
//===----------------------------------------------------------------------===//

/// A generic K-input LUT pattern that can implement any boolean function
/// with up to K inputs using a lookup table.
struct GenericLUT : public CutRewritePattern {
  unsigned k; // Maximum number of inputs for this LUT

  GenericLUT(mlir::MLIRContext *context, unsigned k)
      : CutRewritePattern(context), k(k) {}

  bool match(const Cut &cut) const override {
    // This pattern can implement any cut with at most k inputs
    return cut.getInputSize() <= k && cut.getOutputSize() == 1;
  }

  llvm::FailureOr<Operation *> rewrite(mlir::OpBuilder &rewriter,
                                       Cut &cut) const override {
    // NOTE: Don't use NPN since it's unnecessary.
    auto truthTable = cut.getTruthTable();
    LLVM_DEBUG({
      llvm::dbgs() << "Rewriting cut with " << cut.getInputSize()
                   << " inputs and " << cut.getInputSize()
                   << " operations to a generic LUT with " << k << " inputs.\n";
      cut.dump(llvm::dbgs());
      llvm::dbgs() << "Truth table details:\n";
      truthTable.dump(llvm::dbgs());
    });

    SmallVector<bool> lutTable;
    // Convert the truth table to a LUT table
    for (uint32_t i = 0; i < truthTable.table.getBitWidth(); ++i)
      lutTable.push_back(truthTable.table[i]);

    auto arrayAttr = rewriter.getBoolArrayAttr(
        lutTable); // Create a boolean array attribute.

    // Reverse the inputs to match the LUT input order
    SmallVector<Value> lutInputs(cut.inputs.rbegin(), cut.inputs.rend());

    // Generate comb.truth table operation.
    auto truthTableOp = rewriter.create<comb::TruthTableOp>(
        cut.getRoot()->getLoc(), lutInputs, arrayAttr);

    // Replace the root operation with the truth table operation
    return truthTableOp.getOperation();
  }

  double getArea() const override {
    // Each LUT has unit area regardless of the function it implements
    return 1.0;
  }

  DelayType getDelay(unsigned inputIndex, unsigned outputIndex) const override {
    // All LUTs have unit delay from any input to any output
    return 1;
  }

  unsigned getNumOutputs() const override { return 1; }

  StringRef getPatternName() const override { return "GenericLUT"; }
};

//===----------------------------------------------------------------------===//
// Generic LUT Mapper Pass
//===----------------------------------------------------------------------===//

struct GenericLUTMapperPass
    : public impl::GenericLutMapperBase<GenericLUTMapperPass> {
  using GenericLutMapperBase<GenericLUTMapperPass>::GenericLutMapperBase;

  void runOnOperation() override {
    auto module = getOperation();
    // Create the cut rewriter options
    CutRewriterOptions options;
    options.strategy = strategy;
    options.maxCutInputSize = maxLutSize;
    options.maxCutSizePerRoot = maxCutsPerRoot;
    options.allowNoMatch = false;
    options.attachDebugTiming = test;

    // Create the pattern for generic K-LUT
    SmallVector<std::unique_ptr<CutRewritePattern>, 4> patterns;
    patterns.push_back(
        std::make_unique<GenericLUT>(module->getContext(), maxLutSize));

    // Create the pattern set
    CutRewritePatternSet patternSet(std::move(patterns));

    // Create the cut rewriter
    CutRewriter rewriter(options, patternSet);

    // Apply the rewriting
    if (failed(rewriter.run(module)))
      return signalPassFailure();
  }
};
