//===- Vectorization.cpp -  Vectorize primitive operations ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass performs vectorization for primitive operations, e.g:
// vector_create (or a[0], b[0]), (or a[1], b[1]), (or a[2], b[2])
// => elementwise_or a, b
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-vectorization"

using namespace circt;
using namespace firrtl;

namespace {
//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

template <typename OpTy, typename ResultOpType>
class VectorCreateToLogicElementwise : public mlir::RewritePattern {
public:
  VectorCreateToLogicElementwise(MLIRContext *context)
      : RewritePattern(VectorCreateOp::getOperationName(), 0, context) {}

  LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto vectorCreateOp = cast<VectorCreateOp>(op);
    FVectorType type = vectorCreateOp.getType();
    if (type.hasUninferredWidth() || !type_isa<UIntType>(type.getElementType()))
      return failure();

    SmallVector<Value> lhs, rhs;

    // Vectorize if all operands are `OpTy`. Currently there is no other
    // condition so it could be too aggressive.
    if (llvm::all_of(op->getOperands(), [&](Value operand) {
          auto op = operand.getDefiningOp<OpTy>();
          if (!op)
            return false;
          lhs.push_back(op.getLhs());
          rhs.push_back(op.getRhs());
          return true;
        })) {
      auto lhsVec = rewriter.createOrFold<VectorCreateOp>(
          op->getLoc(), vectorCreateOp.getType(), lhs);
      auto rhsVec = rewriter.createOrFold<VectorCreateOp>(
          op->getLoc(), vectorCreateOp.getType(), rhs);
      rewriter.replaceOpWithNewOp<ResultOpType>(op, lhsVec, rhsVec);
      return success();
    }
    return failure();
  }
};

template <typename OpTy, typename ResultOpType>
class ChainedReducer : public mlir::RewritePattern {
public:
  ChainedReducer(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 0, context) {}

  bool recurse(OpTy root, Value vec, DenseSet<size_t> &indexes) const {
    auto lhsSub =
        dyn_cast_or_null<SubindexOp>(root->getOperand(0).getDefiningOp());
    auto rhsSub =
        dyn_cast_or_null<SubindexOp>(root->getOperand(1).getDefiningOp());
    auto lhsOp = dyn_cast<OpTy>(root->getOperand(0).getDefiningOp());
    auto rhsOp = dyn_cast<OpTy>(root->getOperand(1).getDefiningOp());
    // op(subindex(vec,x), op_chain);
    // op(op_chain, subindex(vec,x))
    // op(subindex(vec,x), subindex(vec,y))
    if (lhsSub && rhsOp && lhsSub.getInput() == vec) {
      indexes.insert(lhsSub.getIndex());
      return recurse(rhsOp, vec, indexes);
    }
    if (rhsSub && lhsOp && rhsSub.getInput() == vec) {
      indexes.insert(rhsSub.getIndex());
      return recurse(lhsOp, vec, indexes);
    }
    if (lhsSub && rhsSub && lhsSub.getInput() == vec &&
        rhsSub.getInput() == vec) {
      indexes.insert(lhsSub.getIndex());
      indexes.insert(rhsSub.getIndex());
      return true;
    }
    return false;
  }

  LogicalResult
  matchAndRewrite(Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto root = cast<OpTy>(op);
    // Try each recursion in turn
    if (auto lhsSub =
            dyn_cast_or_null<SubindexOp>(root->getOperand(0).getDefiningOp())) {
      DenseSet<size_t> indexes;
      if (recurse(root, lhsSub.getInput(), indexes) &&
          indexes.size() ==
              firrtl::type_cast<FVectorType>(lhsSub.getInput().getType())
                  .getNumElements()) {
        rewriter.replaceOpWithNewOp<ResultOpType>(op, lhsSub.getInput());
        return success();
      }
    }
    if (auto rhsSub =
            dyn_cast_or_null<SubindexOp>(root->getOperand(1).getDefiningOp())) {
      DenseSet<size_t> indexes;
      if (recurse(root, rhsSub.getInput(), indexes) &&
          indexes.size() ==
              firrtl::type_cast<FVectorType>(rhsSub.getInput().getType())
                  .getNumElements()) {
        rewriter.replaceOpWithNewOp<ResultOpType>(op, rhsSub.getInput());
        return success();
      }
    }
    return failure();
  }
};

struct VectorizationPass : public VectorizationBase<VectorizationPass> {
  VectorizationPass() = default;
  void runOnOperation() override;
};

} // namespace

void VectorizationPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Running Vectorization "
                             "--------------------------------------===\n"
                          << "Module: '" << getOperation().getName() << "'\n";);

  RewritePatternSet patterns(&getContext());
  patterns.insert<VectorCreateToLogicElementwise<OrPrimOp, OrVecOp>,
                  VectorCreateToLogicElementwise<AndPrimOp, AndVecOp>,
                  VectorCreateToLogicElementwise<XorPrimOp, XorVecOp>,
                  VectorCreateToLogicElementwise<AddPrimOp, AddVecOp>,
                  VectorCreateToLogicElementwise<SubPrimOp, SubVecOp>,
                  VectorCreateToLogicElementwise<LEQPrimOp, LEQVecOp>,
                  VectorCreateToLogicElementwise<LTPrimOp, LTVecOp>,
                  VectorCreateToLogicElementwise<GEQPrimOp, GEQVecOp>,
                  VectorCreateToLogicElementwise<GTPrimOp, GTVecOp>,
                  VectorCreateToLogicElementwise<EQPrimOp, EQVecOp>,
                  VectorCreateToLogicElementwise<NEQPrimOp, NEQVecOp>,
                  ChainedReducer<OrPrimOp, OrRVecOp>,
                  ChainedReducer<AndPrimOp, AndRVecOp>,
                  ChainedReducer<XorPrimOp, XorRVecOp>>(&getContext());
  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  (void)applyPatternsAndFoldGreedily(getOperation(), frozenPatterns);
}

std::unique_ptr<mlir::Pass> circt::firrtl::createVectorizationPass() {
  return std::make_unique<VectorizationPass>();
}
