//===- GreedyCutDecomp.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs cut decomposition on AIGs based on a naive greedy
// algorithm. We first convert all `aig.and_inv` to `aig.cut` that have a single
// operation and then try to merge cut operations on inputs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "aig-greedy-cut-decomp"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_GREEDYCUTDECOMP
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct AndInverterOpToCutPattern : public OpRewritePattern<aig::AndInverterOp> {
  using OpRewritePattern<aig::AndInverterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(aig::AndInverterOp op,
                                PatternRewriter &rewriter) const override {
    if (op->getParentOfType<aig::CutOp>())
      return failure();

    auto cutOp = rewriter.create<aig::CutOp>(
        op.getLoc(), op.getResult().getType(), op.getInputs(),
        [&](Block::BlockArgListType args) {
          auto result = rewriter.create<aig::AndInverterOp>(
              op.getLoc(), op.getResult().getType(), args,
              op.getInvertedAttr());
          rewriter.create<aig::OutputOp>(op.getLoc(), ValueRange{result});
        });

    rewriter.replaceOp(op, cutOp);
    return success();
  }
};

static aig::CutOp mergeCuts(Location loc, MutableArrayRef<Operation *> cuts,
                            ArrayRef<Value> inputs, Value output,
                            PatternRewriter &rewriter) {
  if (!mlir::computeTopologicalSorting(cuts))
    return {};

  assert(cuts.size() >= 2);

  DenseMap<Value, Value> valueToNewValue, inputsToBlockArg;
  auto cutOp = rewriter.create<aig::CutOp>(
      loc, output.getType(), inputs, [&](Block::BlockArgListType args) {
        for (auto [i, input] : llvm::enumerate(inputs))
          inputsToBlockArg[input] = args[i];

        for (auto [i, cut] : llvm::enumerate(cuts)) {
          auto cutOp = cast<aig::CutOp>(cut);
          assert(cutOp.getNumResults() == 1);
          for (auto [arg, input] : llvm::zip(
                   cutOp.getBodyBlock()->getArguments(), cutOp.getInputs())) {
            auto it = inputsToBlockArg.find(input);
            if (it != inputsToBlockArg.end()) {
              rewriter.replaceAllUsesWith(arg, it->second);
            } else {
              auto cutOp = dyn_cast<aig::CutOp>(input.getDefiningOp());
              assert(cutOp && cutOp.getNumResults() == 1);
              rewriter.replaceAllUsesWith(arg, valueToNewValue.at(input));
            }
          }

          assert(cutOp.getNumResults() == 1);
          valueToNewValue[cutOp.getResult(0)] =
              cutOp.getBodyBlock()->getTerminator()->getOperand(0);
        }
      });

  rewriter.replaceAllUsesWith(output, cutOp.getResult(0));
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(cutOp.getBodyBlock());
    rewriter.create<aig::OutputOp>(loc, ValueRange{valueToNewValue.at(output)});
  }

  for (auto oldCut : llvm::reverse(cuts)) {
    auto *oldCutBlock = cast<aig::CutOp>(oldCut).getBodyBlock();
    auto oldCutOutput = oldCutBlock->getTerminator();
    rewriter.eraseOp(oldCutOutput);
    // Erase arguments before inlining. Arguments are already replaced.
    oldCutBlock->eraseArguments([](BlockArgument block) { return true; });
    rewriter.inlineBlockBefore(oldCutBlock, cutOp.getBodyBlock(),
                               cutOp.getBodyBlock()->begin());
    rewriter.eraseOp(oldCut);
  }

  return cutOp;
}

struct MergeCutPattern : public OpRewritePattern<aig::CutOp> {
  MergeCutPattern(MLIRContext *context, unsigned cutLimit)
      : OpRewritePattern<aig::CutOp>(context), cutLimit(cutLimit) {}
  const unsigned cutLimit;

  LogicalResult matchAndRewrite(aig::CutOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() >= cutLimit)
      return failure();

    SmallVector<aig::CutOp> inputCuts;
    {
      SetVector<aig::CutOp> inputCutsSet;
      inputCutsSet.insert(op);
      for (auto cut : op.getOperands()) {
        if (auto cutOp = cut.getDefiningOp<aig::CutOp>();
            cutOp && cutOp.getNumResults() == 1)
          inputCutsSet.insert(cutOp);
      }

      inputCuts = std::move(inputCutsSet.takeVector());
    }

    if (inputCuts.size() <= 1)
      return failure();

    SmallVector<Value> inputs;
    inputs.reserve(inputCuts.size());

    // This is naive implementation of the cut emuration of the local inputs.
    // FIXME: This is really dumb algorithm, but it is just a proof of concept.

    LLVM_DEBUG(llvm::dbgs() << "Trying to merge " << op << "\n");
    for (unsigned i = (1 << (inputCuts.size() - 1)) - 1; i != 0; --i) {
      auto checkSubsetMerge = [&](unsigned i) -> LogicalResult {
        SetVector<Value> inValues;
        llvm::SmallDenseSet<aig::CutOp, 4> cutSet;
        SmallVector<Value> outValues;

        for (unsigned j = 0; j < inputCuts.size(); ++j) {
          if (i & (1 << j)) {
            cutSet.insert(inputCuts[j]);
            for (auto in : inputCuts[j].getInputs())
              inValues.insert(in);
            outValues.push_back(inputCuts[j].getResult(0));
            LLVM_DEBUG(llvm::dbgs() << "Added " << inputCuts[j] << "\n");
          }
        }

        Value singleOutput;
        for (auto out : outValues) {
          // Users of cuts must be closed under the cut set.
          bool isClosed =
              llvm::all_of(out.getUsers(), [&cutSet](Operation *user) {
                if (auto cutOp = dyn_cast<aig::CutOp>(user))
                  return cutSet.contains(cutOp);
                return false;
              });
          inValues.remove(out);
          if (!isClosed) {
            if (singleOutput) {
              LLVM_DEBUG(llvm::dbgs() << "Not closed\n");
              return failure();
            }

            singleOutput = out;
          }
        }

        if (!singleOutput || inValues.size() > cutLimit) {
          LLVM_DEBUG(llvm::dbgs() << "Limit exceeded\n");
          return failure();
        }

        SmallVector<Operation *> subsetCuts;
        for (unsigned j = 0; j < inputCuts.size(); ++j) {
          if (i & (1 << j))
            subsetCuts.push_back(inputCuts[j]);
        }

        // Ok, let's merge the cuts.
        auto cutOp = mergeCuts(op.getLoc(), subsetCuts, inValues.takeVector(),
                               singleOutput, rewriter);
        if (!cutOp) {
          LLVM_DEBUG(llvm::dbgs() << "Failed to merge\n");
          return failure();
        }
        return success();
      };

      // Always enable a bit for 0 (the original cut).
      auto result = checkSubsetMerge((i << 1) | 1);
      if (succeeded(result))
        return result;
    }
    return failure();
  }
};

struct SinkConstantPattern : public mlir::OpRewritePattern<aig::CutOp> {
  using mlir::OpRewritePattern<aig::CutOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(aig::CutOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> oldInputs, oldArgs;
    auto *block = op.getBodyBlock();
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(block);
      BitVector eraseArgs(block->getNumArguments());
      bool changed = false;
      for (auto [i, in] : llvm::enumerate(op.getInputs())) {
        if (auto constOp = in.getDefiningOp<hw::ConstantOp>()) {
          eraseArgs.set(i);
          auto cloned = rewriter.clone(*constOp);
          rewriter.replaceAllUsesWith(block->getArgument(i),
                                      cloned->getResult(0));
          changed = true;
        } else {
          oldInputs.push_back(in);
          oldArgs.push_back(block->getArgument(i));
        }
      }
      if (!changed)
        return failure();
      block->eraseArguments(eraseArgs);
    }

    rewriter.modifyOpInPlace(
        op, [&]() { op.getInputsMutable().assign(oldInputs); });
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Greedy Cut Decomposition Pass
//===----------------------------------------------------------------------===//

namespace {
struct GreedyCutDecompPass
    : public impl::GreedyCutDecompBase<GreedyCutDecompPass> {
  using GreedyCutDecompBase::GreedyCutDecompBase;
  void runOnOperation() override;
  using GreedyCutDecompBase::cutSizes;
};
} // namespace

void GreedyCutDecompPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<AndInverterOpToCutPattern, SinkConstantPattern>(
      patterns.getContext());
  patterns.add<MergeCutPattern>(patterns.getContext(), cutSizes.getValue());
  mlir::FrozenRewritePatternSet frozen(std::move(patterns));
  mlir::GreedyRewriteConfig config;

  config.useTopDownTraversal = true;

  if (failed(
          mlir::applyPatternsAndFoldGreedily(getOperation(), frozen, config)))
    return signalPassFailure();
}
