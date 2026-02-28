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
#include "circt/Support/TruthTable.h"
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
  SmallVector<DelayType, 8> cachedDelays; // Pre-computed unit delays

  GenericLUT(mlir::MLIRContext *context, unsigned k)
      : CutRewritePattern(context), k(k), cachedDelays(k, 1) {}

  std::optional<MatchResult> match(CutEnumerator &enumerator,
                                   const Cut &cut) const override {
    const auto &network = enumerator.getLogicNetwork();
    // This pattern can implement any cut with at most k inputs
    if (cut.getInputSize() > k || cut.getOutputSize(network) != 1)
      return std::nullopt;

    // Create match result with a reference to cached delays
    return MatchResult(
        1.0, ArrayRef<DelayType>(cachedDelays).take_front(cut.getInputSize()));
  }

  llvm::FailureOr<Operation *> rewrite(mlir::OpBuilder &rewriter,
                                       CutEnumerator &enumerator,
                                       const Cut &cut) const override {
    const auto &network = enumerator.getLogicNetwork();
    // NOTE: Don't use NPN since it's unnecessary.
    const auto &truthTableOpt = cut.getTruthTable();
    if (!truthTableOpt)
      return failure();
    const auto &truthTable = *truthTableOpt;
    LLVM_DEBUG({
      llvm::dbgs() << "Rewriting cut with " << cut.getInputSize()
                   << " inputs and " << cut.getInputSize()
                   << " operations to a generic LUT with " << k << " inputs.\n";
      cut.dump(llvm::dbgs(), network);
      llvm::dbgs() << "Truth table details:\n";
      truthTable.dump(llvm::dbgs());
    });

    SmallVector<bool> lutTable;
    // Convert the truth table to a LUT table
    for (uint32_t i = 0; i < truthTable.table.getBitWidth(); ++i)
      lutTable.push_back(truthTable.table[i]);

    // Reverse the inputs to match the LUT input order
    SmallVector<Value> lutInputs;
    lutInputs.reserve(cut.inputs.size());
    for (auto i : llvm::reverse(cut.inputs))
      lutInputs.push_back(network.getValue(i));

    auto *rootOp = network.getGate(cut.getRootIndex()).getOperation();
    assert(rootOp && "cut root must be a valid operation");

    // Generate comb.truth table operation.
    auto truthTableOp = comb::TruthTableOp::create(rewriter, rootOp->getLoc(),
                                                   lutInputs, lutTable);

    // Replace the root operation with the truth table operation
    return truthTableOp.getOperation();
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
