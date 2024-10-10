//===- ElaborationPass.cpp - RTG ElaborationPass implementation -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass elaborates the random parts of the RTG dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <random>

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_ELABORATION
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;

#define DEBUG_TYPE "rtg-elaboration"

namespace {
struct ElaborationPass : public rtg::impl::ElaborationBase<ElaborationPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // end namespace

static unsigned interpret(Value value) { return 1; }

static LogicalResult elaborate(rtg::SelectRandomOp op,
                               std::optional<unsigned> seed) {
  std::random_device dev;
  unsigned s = dev();
  if (seed.has_value())
    s = *seed;
  std::mt19937 rng(s);

  std::vector<float> idx;
  std::vector<float> prob;
  idx.push_back(0);
  for (unsigned i = 1; i <= op.getRatios().size(); ++i) {
    idx.push_back(i);
    unsigned p = interpret(op.getRatios()[i - 1]);
    prob.push_back(p);
  }

  std::piecewise_constant_distribution<float> dist(idx.begin(), idx.end(),
                                                   prob.begin());

  unsigned selected = dist(rng);

  auto snippet = op.getSnippets()[selected].getDefiningOp<rtg::SnippetOp>();
  if (!snippet)
    return failure();

  IRRewriter rewriter(op);
  auto *clone = snippet->clone();
  rewriter.inlineBlockBefore(&clone->getRegion(0).front(), op);
  clone->erase();

  op->erase();

  return success();
}

void ElaborationPass::runOnOperation() {
  auto moduleOp = getOperation();
  // TODO: Use a proper visitor
  if (moduleOp
          .walk([&](Operation *op) -> WalkResult {
            return TypeSwitch<Operation *, LogicalResult>(op)
                .Case<rtg::SelectRandomOp>([&](auto op) {
                  std::optional<unsigned> s = std::nullopt;
                  if (seed.hasValue())
                    s = seed.getValue();
                  return elaborate(op, s);
                })
                .Default([](auto op) { return success(); });
          })
          .wasInterrupted())
    signalPassFailure();
}
