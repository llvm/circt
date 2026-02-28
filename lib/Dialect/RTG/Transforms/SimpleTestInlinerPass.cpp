//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/IR/PatternMatch.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_SIMPLETESTINLINERPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Simple Test Inliner Pass
//===----------------------------------------------------------------------===//

namespace {
struct SimpleTestInlinerPass
    : public rtg::impl::SimpleTestInlinerPassBase<SimpleTestInlinerPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void SimpleTestInlinerPass::runOnOperation() {
  const auto &symTbl = getAnalysis<SymbolTable>();
  IRRewriter rewriter(getOperation());

  for (auto fileOp : getOperation().getOps<emit::FileOp>()) {
    rewriter.setInsertionPointToStart(fileOp.getBody());
    auto segOp =
        SegmentOp::create(rewriter, fileOp->getLoc(), SegmentKind::Text);
    segOp.getBodyRegion().emplaceBlock();

    Block *fileBlock = fileOp.getBody();
    Block *segBlock = segOp.getBody();

    auto opsToMove = llvm::make_early_inc_range(
        llvm::make_range(std::next(fileBlock->begin()), fileBlock->end()));
    for (auto &op : opsToMove)
      op.moveBefore(segBlock, segBlock->end());

    for (auto refOp : llvm::make_early_inc_range(segOp.getOps<emit::RefOp>())) {
      auto testOp = symTbl.lookup<TestOp>(refOp.getTargetAttr().getAttr());
      if (!testOp) {
        refOp.emitError("invalid symbol reference: ") << refOp.getTargetAttr();
        return signalPassFailure();
      }

      bool allArgsUnused =
          llvm::all_of(testOp.getBody()->getArguments(),
                       [](auto arg) { return arg.use_empty(); });
      if (!allArgsUnused) {
        testOp->emitError("cannot inline test with used arguments");
        return signalPassFailure();
      }

      testOp.getBody()->eraseArguments(0, testOp.getBody()->getNumArguments());
      rewriter.setInsertionPoint(refOp);
      auto testBeginComment = ConstantOp::create(
          rewriter, refOp->getLoc(),
          StringAttr::get(Twine("Begin of test '") + testOp.getSymName() + "'",
                          StringType::get(rewriter.getContext())));
      CommentOp::create(rewriter, refOp->getLoc(), testBeginComment);
      auto newTestOp = cast<TestOp>(testOp->clone());
      rewriter.inlineBlockBefore(newTestOp.getBody(), refOp, {});
      auto testEndComment = ConstantOp::create(
          rewriter, refOp->getLoc(),
          StringAttr::get(Twine("End of test '") + testOp.getSymName() + "'",
                          StringType::get(rewriter.getContext())));
      CommentOp::create(rewriter, refOp->getLoc(), testEndComment);
      newTestOp.erase();
      refOp.erase();
    }
  }

  for (auto &op : llvm::make_early_inc_range(getOperation().getOps()))
    if (isa<TargetOp, TestOp>(&op))
      op.erase();
}
