//===- AffineParallelUnroll.cpp - Unroll AffineParallelOp ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unroll AffineParallelOp to facilitate lowering to Calyx ParOp.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace calyx {
#define GEN_PASS_DEF_AFFINEPARALLELUNROLL
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"
} // namespace calyx
} // namespace circt

using namespace circt;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::arith;

namespace {
struct CalyxParLICM {
  // This pass tries to prevent potential memory banking contention by hoisting
  // memory accesses after AffineParallelUnroll. It only hoists memory reads
  // that occur *more than once* inside `scf.execute_region`s. Since
  // AffineParallelUnroll converts loop indices to constants, this consecutive
  // pass can safely analyze and remove redundant accesses.
  LogicalResult run(AffineParallelOp affineParallelOp);

  // Computes and collects all memory accesses (read/write) that have constant
  // access indices.
  DenseMap<Operation *, SmallVector<int64_t, 4>>
  computeConstMemAccessIndices(AffineParallelOp affineParallelOp);

  // Performs memory contention analysis on memory write operations, returning
  // `failure` if the pass identifies two write operations write to the same
  // memory reference at the same access indices in different parallel regions.
  LogicalResult
  writeOpAnalysis(DenseMap<Operation *, SmallVector<int64_t, 4>> &);

  // Tries to hoist memory read operations that will cause memory access
  // contention, such as reading from the same memory reference with the same
  // access indices in different parallel regions.
  void readOpHoistAnalysis(AffineParallelOp affineParallelOp,
                           DenseMap<Operation *, SmallVector<int64_t, 4>> &);

  DenseMap<std::pair<Value, SmallVector<int64_t, 4>>, scf::ExecuteRegionOp>
      constantWriteOpIndices;

  DenseMap<std::pair<Value, SmallVector<int64_t, 4>>, int> constantReadOpCounts;
};
} // end anonymous namespace

namespace llvm {
template <>
struct DenseMapInfo<std::pair<Value, SmallVector<int64_t, 4>>> {
  using PairType = std::pair<Value, SmallVector<int64_t, 4>>;

  static inline PairType getEmptyKey() {
    return {DenseMapInfo<Value>::getEmptyKey(), {}};
  }

  static inline PairType getTombstoneKey() {
    return {DenseMapInfo<Value>::getTombstoneKey(), {}};
  }

  static unsigned getHashValue(const PairType &pair) {
    unsigned hash = DenseMapInfo<Value>::getHashValue(pair.first);
    for (const auto &v : pair.second)
      hash = llvm::hash_combine(hash, DenseMapInfo<int64_t>::getHashValue(v));
    return hash;
  }

  static bool isEqual(const PairType &lhs, const PairType &rhs) {
    return lhs.first == rhs.first && lhs.second == rhs.second;
  }
};
} // namespace llvm

DenseMap<Operation *, SmallVector<int64_t, 4>>
CalyxParLICM::computeConstMemAccessIndices(AffineParallelOp affineParallelOp) {
  DenseMap<Operation *, SmallVector<int64_t, 4>> constantMemAccessIndices;

  MLIRContext *ctx = affineParallelOp->getContext();
  auto executeRegionOps =
      affineParallelOp.getBody()->getOps<scf::ExecuteRegionOp>();
  for (auto executeRegionOp : executeRegionOps) {
    executeRegionOp.walk([&](Operation *op) {
      if (!isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
        return WalkResult::advance();

      auto read = dyn_cast<AffineReadOpInterface>(op);
      AffineMap map = read ? read.getAffineMap()
                           : cast<AffineWriteOpInterface>(op).getAffineMap();
      ValueRange mapOperands =
          read ? read.getMapOperands()
               : cast<AffineWriteOpInterface>(op).getMapOperands();

      SmallVector<Attribute> operandConsts;
      for (Value operand : mapOperands) {
        if (auto constOp =
                operand.template getDefiningOp<arith::ConstantIndexOp>()) {
          operandConsts.push_back(
              IntegerAttr::get(IndexType::get(ctx), constOp.value()));
        } else {
          return WalkResult::advance();
        }
      }

      SmallVector<int64_t, 4> evaluatedIndices, foldedResults;
      bool hasPoison = false;
      map.partialConstantFold(operandConsts, &foldedResults, &hasPoison);
      if (!(hasPoison || foldedResults.empty()))
        constantMemAccessIndices[op] = foldedResults;

      return WalkResult::advance();
    });
  }

  return constantMemAccessIndices;
}

LogicalResult CalyxParLICM::writeOpAnalysis(
    DenseMap<Operation *, SmallVector<int64_t, 4>> &constantMemAccessIndices) {
  for (auto &[memOp, constIndices] : constantMemAccessIndices) {
    auto writeOp = dyn_cast<AffineWriteOpInterface>(memOp);
    if (!writeOp) {
      continue;
    }

    auto parentExecuteRegionOp =
        writeOp->getParentOfType<scf::ExecuteRegionOp>();
    auto key = std::pair(writeOp.getMemRef(), constIndices);
    if (constantWriteOpIndices.find(key) != constantWriteOpIndices.end() &&
        constantWriteOpIndices[key] != parentExecuteRegionOp) {
      // It cannot occur more than once in different parallel regions, which
      // will result in write contention.
      return failure();
    }

    constantWriteOpIndices[key] = parentExecuteRegionOp;
  }
  return success();
}

void CalyxParLICM::readOpHoistAnalysis(
    AffineParallelOp affineParallelOp,
    DenseMap<Operation *, SmallVector<int64_t, 4>> &constantMemAccessIndices) {
  for (auto &[memOp, constIndices] : constantMemAccessIndices) {
    auto readOp = dyn_cast<AffineReadOpInterface>(memOp);
    if (!readOp)
      continue;

    auto key = std::pair(readOp.getMemRef(), constIndices);
    if (constantWriteOpIndices.contains(key))
      continue;
    constantReadOpCounts[key]++;
  }
  bool shouldHoist = llvm::any_of(
      constantReadOpCounts, [](const auto &entry) { return entry.second > 1; });
  if (!shouldHoist)
    return;

  OpBuilder builder(affineParallelOp);
  DenseMap<std::pair<Value, SmallVector<int64_t, 4>>, ValueRange> hoistedReads;
  for (auto &[memOp, constIndices] : constantMemAccessIndices) {
    auto readOp = dyn_cast<AffineReadOpInterface>(memOp);
    if (!readOp)
      continue;

    auto key = std::pair(readOp.getMemRef(), constIndices);
    if (constantReadOpCounts[key] > 1) {
      if (hoistedReads.find(key) == hoistedReads.end()) {
        builder.setInsertionPoint(affineParallelOp);
        Operation *clonedRead = builder.clone(*readOp.getOperation());
        hoistedReads[key] = clonedRead->getOpResults();
      }
      readOp->replaceAllUsesWith(hoistedReads[key]);
    }
  }
}

LogicalResult CalyxParLICM::run(AffineParallelOp affineParallelOp) {
  auto constantMemAccessIndices =
      computeConstMemAccessIndices(affineParallelOp);

  if (failed(writeOpAnalysis(constantMemAccessIndices))) {
    return failure();
  }

  readOpHoistAnalysis(affineParallelOp, constantMemAccessIndices);

  return success();
}

namespace {

struct AffineParallelUnroll : public OpRewritePattern<AffineParallelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp affineParallelOp,
                                PatternRewriter &rewriter) const override {
    if (affineParallelOp->hasAttr("calyx.unroll"))
      // We assume that having "calyx.unroll" attribute means that it has
      // already been unrolled.
      return failure();

    if (!affineParallelOp.getResults().empty()) {
      affineParallelOp.emitError(
          "affine.parallel with reductions is not supported yet");
      return failure();
    }

    Location loc = affineParallelOp.getLoc();

    rewriter.setInsertionPointAfter(affineParallelOp);
    // Create a single-iteration parallel loop op and mark its special by
    // setting the "calyx.unroll" attribute.
    AffineMap lbMap = AffineMap::get(0, 0, rewriter.getAffineConstantExpr(0),
                                     rewriter.getContext());
    AffineMap ubMap = AffineMap::get(0, 0, rewriter.getAffineConstantExpr(1),
                                     rewriter.getContext());
    auto newParallelOp = rewriter.create<AffineParallelOp>(
        loc, /*resultTypes=*/TypeRange(),
        /*reductions=*/SmallVector<arith::AtomicRMWKind>(),
        /*lowerBoundsMap=*/lbMap, /*lowerBoundsOperands=*/SmallVector<Value>(),
        /*upperBoundsMap=*/ubMap, /*upperBoundsOperands=*/SmallVector<Value>(),
        /*steps=*/SmallVector<int64_t>({1}));
    newParallelOp->setAttr("calyx.unroll", rewriter.getBoolAttr(true));

    SmallVector<int64_t> pLoopLowerBounds =
        affineParallelOp.getLowerBoundsMap().getConstantResults();
    if (pLoopLowerBounds.empty()) {
      affineParallelOp.emitError(
          "affine.parallel must have constant lower bounds");
      return failure();
    }
    SmallVector<int64_t> pLoopUpperBounds =
        affineParallelOp.getUpperBoundsMap().getConstantResults();
    if (pLoopUpperBounds.empty()) {
      affineParallelOp.emitError(
          "affine.parallel must have constant upper bounds");
      return failure();
    }
    SmallVector<int64_t, 8> pLoopSteps = affineParallelOp.getSteps();

    Block *pLoopBody = affineParallelOp.getBody();
    MutableArrayRef<BlockArgument> pLoopIVs = affineParallelOp.getIVs();

    OpBuilder insideBuilder(newParallelOp);
    SmallVector<int64_t> indices = pLoopLowerBounds;
    while (true) {
      insideBuilder.setInsertionPointToStart(newParallelOp.getBody());
      // Create an `scf.execute_region` to wrap each unrolled block since
      // `affine.parallel` requires only one block in the body region.
      auto executeRegionOp =
          insideBuilder.create<scf::ExecuteRegionOp>(loc, TypeRange{});
      Region &executeRegionRegion = executeRegionOp.getRegion();
      Block *executeRegionBlock = &executeRegionRegion.emplaceBlock();

      OpBuilder regionBuilder(executeRegionOp);
      // Each iteration starts with a fresh mapping, so each new block’s
      // argument of a region-based operation (such as `affine.for`) get
      // re-mapped independently.
      IRMapping operandMap;
      regionBuilder.setInsertionPointToEnd(executeRegionBlock);
      // Map induction variables to constant indices
      for (unsigned i = 0; i < indices.size(); ++i) {
        Value ivConstant =
            regionBuilder.create<arith::ConstantIndexOp>(loc, indices[i]);
        operandMap.map(pLoopIVs[i], ivConstant);
      }

      for (auto it = pLoopBody->begin(); it != std::prev(pLoopBody->end());
           ++it)
        regionBuilder.clone(*it, operandMap);

      // A terminator should always be inserted in `scf.execute_region`'s block.
      regionBuilder.create<scf::YieldOp>(loc);

      // Increment indices using `step`.
      bool done = false;
      for (int dim = indices.size() - 1; dim >= 0; --dim) {
        indices[dim] += pLoopSteps[dim];
        if (indices[dim] < pLoopUpperBounds[dim])
          break;
        indices[dim] = pLoopLowerBounds[dim];
        if (dim == 0)
          // All combinations have been generated
          done = true;
      }
      if (done)
        break;
    }

    rewriter.replaceOp(affineParallelOp, newParallelOp);

    return success();
  }
};

struct AffineParallelUnrollPass
    : public circt::calyx::impl::AffineParallelUnrollBase<
          AffineParallelUnrollPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }
  void runOnOperation() override;
};

} // end anonymous namespace

void AffineParallelUnrollPass::runOnOperation() {
  auto *ctx = &getContext();
  ConversionTarget target(*ctx);

  RewritePatternSet patterns(ctx);
  patterns.add<AffineParallelUnroll>(ctx);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitError("Failed to unroll affine.parallel");
    signalPassFailure();
  }

  getOperation()->walk([&](AffineParallelOp parOp) {
    if (parOp->hasAttr("calyx.unroll")) {
      if (failed(CalyxParLICM().run(parOp))) {
        parOp.emitError("Failed to unroll");
        signalPassFailure();
      }
    }
  });

  RewritePatternSet canonicalizePatterns(ctx);
  scf::IndexSwitchOp::getCanonicalizationPatterns(canonicalizePatterns, ctx);
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(canonicalizePatterns)))) {
    getOperation()->emitError("Failed to apply canonicalization.");
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> circt::calyx::createAffineParallelUnrollPass() {
  return std::make_unique<AffineParallelUnrollPass>();
}
