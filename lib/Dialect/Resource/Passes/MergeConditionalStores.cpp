//===- MergeConditionalStores.cpp -----------------------------------------===//
//
// If-conversion pre-pass for the affine-to-loopschedule pipeline.
//
// The AffineToLoopSchedule pass inherits CIRCT's IfOpHoisting pattern, which
// hoists ALL ops out of scf.if branches to expose them to the scheduler -
// including stores. For an affine.if whose branches each store to the same
// address, that miscompiles: every branch's store executes unconditionally
// and the last-scheduled one wins. It also inflates memory port demand
// (N stores per iteration instead of 1), distorting II and BRAM port
// estimates relative to what Vitis synthesizes (a single predicated/muxed
// write).
//
// This pass rewrites, bottom-up:
//
//   affine.if #set(...) {            %cond = <expand #set as arith ops>
//     <pure ops A>                   <pure ops A>
//     affine.store %a, %m[i, j]  =>  <pure ops B>
//   } else {                         %v = arith.select %cond, %a, %b
//     <pure ops B>                   affine.store %v, %m[i, j]
//     affine.store %b, %m[i, j]
//   }
//
// Nested else-if chains converge under the greedy driver: the innermost if
// is rewritten first, leaving its parent's else block as pure ops + select
// + store, which then matches in turn.
//
// Match requirements (rewrite is skipped otherwise, leaving the if intact):
//   - the affine.if yields no results and has an else region;
//   - each branch contains exactly one affine.store and otherwise only
//     memory-effect-free, region-free ops (safe to speculate);
//   - both stores target the same memref through the same access map with
//     the same map operands.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::affine;

namespace {

struct MergeConditionalStores : public OpRewritePattern<AffineIfOp> {
  using OpRewritePattern<AffineIfOp>::OpRewritePattern;

  /// Returns the block's single affine.store if the block consists of exactly
  /// one store plus ops that are safe to speculate above the if; otherwise
  /// returns nullptr.
  ///
  /// "Safe to speculate" = region-free and either fully memory-effect-free or
  /// read-only (e.g. an affine.load of a buffer with no aliasing store in this
  /// region). cgeist emits the else-branch value via an affine.load *inside*
  /// the branch, so disqualifying all reads would wrongly skip the symmetric
  /// triangular writeback (store %cst / load+store) and leave the if for
  /// IfOpHoisting to miscompile into a double store.
  ///
  /// A nested (not-yet-converted) affine.if has a region and fails this test,
  /// which is what forces bottom-up convergence.
  static AffineStoreOp getSingleStore(Block *block) {
    if (!block)
      return nullptr;
    AffineStoreOp store;
    for (Operation &op : *block) {
      if (auto s = dyn_cast<AffineStoreOp>(op)) {
        if (store)
          return nullptr; // more than one store
        store = s;
        continue;
      }
      if (isa<AffineYieldOp>(op))
        continue;
      // Regions (e.g. a nested, not-yet-merged affine.if) are never hoistable.
      if (op.getNumRegions() != 0)
        return nullptr;
      // Pure ops hoist freely.
      if (isMemoryEffectFree(&op))
        continue;
      // Otherwise only read-only ops may be speculated above the if. Anything
      // that writes / allocates / frees disqualifies the block.
      auto mem = dyn_cast<MemoryEffectOpInterface>(&op);
      if (!mem)
        return nullptr; // unknown effects: be conservative
      SmallVector<MemoryEffects::EffectInstance> effects;
      mem.getEffects(effects);
      bool readOnly = !effects.empty() &&
                      llvm::all_of(effects, [](const MemoryEffects::EffectInstance &e) {
                        return isa<MemoryEffects::Read>(e.getEffect());
                      });
      if (!readOnly)
        return nullptr;
      // The store we merge to executes unconditionally after hoisting, so this
      // read will too. Only safe if its address is computable above the if —
      // require all operands to dominate the if (defined outside this block).
      for (Value operand : op.getOperands())
        if (operand.getParentBlock() == block)
          return nullptr;
    }
    return store;
  }

  LogicalResult matchAndRewrite(AffineIfOp ifOp,
                                PatternRewriter &rewriter) const override {
    if (ifOp.getNumResults() != 0 || !ifOp.hasElse())
      return failure();

    AffineStoreOp thenStore = getSingleStore(ifOp.getThenBlock());
    AffineStoreOp elseStore = getSingleStore(ifOp.getElseBlock());
    if (!thenStore || !elseStore)
      return failure();

    // Both branches must write the same address of the same memref.
    if (thenStore.getMemRef() != elseStore.getMemRef() ||
        thenStore.getAffineMap() != elseStore.getAffineMap() ||
        !llvm::equal(thenStore.getMapOperands(), elseStore.getMapOperands()))
      return failure();

    Location loc = ifOp.getLoc();
    rewriter.setInsertionPoint(ifOp);

    // Materialize the integer-set condition as an i1, mirroring what
    // AffineIfLowering in --lower-affine does: each constraint becomes
    // `expr >= 0` (or `expr == 0`), and constraints are AND-ed together.
    IntegerSet set = ifOp.getIntegerSet();
    ValueRange operands = ifOp.getOperands();
    ValueRange dims = operands.take_front(set.getNumDims());
    ValueRange syms = operands.drop_front(set.getNumDims());

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cond;
    for (unsigned i = 0, e = set.getNumConstraints(); i < e; ++i) {
      Value expr =
          expandAffineExpr(rewriter, loc, set.getConstraint(i), dims, syms);
      if (!expr)
        return failure();
      auto pred =
          set.isEq(i) ? arith::CmpIPredicate::eq : arith::CmpIPredicate::sge;
      Value c = rewriter.create<arith::CmpIOp>(loc, pred, expr, zero);
      cond = cond ? rewriter.create<arith::AndIOp>(loc, cond, c).getResult()
                  : c;
    }
    if (!cond)
      return failure(); // degenerate empty set; leave the if alone

    // Hoist the pure ops of both branches above the if. They are
    // speculatable by the getSingleStore() check, and only reference values
    // defined above the if, so SSA dominance is preserved.
    auto hoist = [&](Block *block) {
      for (Operation &op : llvm::make_early_inc_range(*block)) {
        if (isa<AffineYieldOp, AffineStoreOp>(op))
          continue;
        op.moveBefore(ifOp);
      }
    };
    hoist(ifOp.getThenBlock());
    hoist(ifOp.getElseBlock());

    // Merge the two stores into a single select-fed store.
    Value merged = rewriter.create<arith::SelectOp>(
        loc, cond, thenStore.getValueToStore(), elseStore.getValueToStore());
    rewriter.create<AffineStoreOp>(loc, merged, thenStore.getMemRef(),
                                   thenStore.getAffineMap(),
                                   thenStore.getMapOperands());
    rewriter.eraseOp(ifOp);
    return success();
  }
};

struct MergeConditionalStoresPass
    : public PassWrapper<MergeConditionalStoresPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeConditionalStoresPass)

  StringRef getArgument() const final { return "merge-conditional-stores"; }
  StringRef getDescription() const final {
    return "If-convert affine.if branches that store to the same address "
           "into a single select-fed store";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, AffineDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MergeConditionalStores>(&getContext());
    // NOTE: older trees spell this applyPatternsAndFoldGreedily.
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace circt::hls_analysis {

std::unique_ptr<mlir::Pass> createMergeConditionalStores() {
  return std::make_unique<MergeConditionalStoresPass>();
}

void registerMergeConditionalStoresPass() {
  mlir::PassRegistration<MergeConditionalStoresPass>();
}
} // namespace circt::hls_analysis