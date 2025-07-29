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
    for (auto refOp :
         llvm::make_early_inc_range(fileOp.getOps<emit::RefOp>())) {
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
      CommentOp::create(rewriter, refOp->getLoc(),
                        rewriter.getStringAttr("Begin of test '" +
                                               testOp.getSymName() + "'"));
      auto newTestOp = cast<TestOp>(testOp->clone());
      rewriter.inlineBlockBefore(newTestOp.getBody(), refOp, {});
      CommentOp::create(
          rewriter, refOp->getLoc(),
          rewriter.getStringAttr("End of test '" + testOp.getSymName() + "'"));
      newTestOp.erase();
      refOp.erase();
    }
  }

  for (auto &op : llvm::make_early_inc_range(getOperation().getOps()))
    if (isa<TargetOp, TestOp>(&op))
      op.erase();
}
