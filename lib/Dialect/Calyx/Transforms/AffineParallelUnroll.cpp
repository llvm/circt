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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

// A shared interface to store affine memory accesses.
struct AffineAccessExpr {
  Value memref;
  AffineMap map;
  SmallVector<Value, 4> operands;

  bool operator==(const AffineAccessExpr &other) const {
    return memref == other.memref && map == other.map &&
           operands == other.operands;
  }
};

namespace llvm {
template <>
struct DenseMapInfo<AffineAccessExpr> {
  static AffineAccessExpr getEmptyKey() {
    return {DenseMapInfo<Value>::getEmptyKey(), {}, {}};
  }

  static AffineAccessExpr getTombstoneKey() {
    return {DenseMapInfo<Value>::getTombstoneKey(), {}, {}};
  }

  static unsigned getHashValue(const AffineAccessExpr &expr) {
    unsigned h = DenseMapInfo<Value>::getHashValue(expr.memref);
    h = llvm::hash_combine(h, DenseMapInfo<AffineMap>::getHashValue(expr.map));
    for (Value operand : expr.operands)
      h = llvm::hash_combine(h, DenseMapInfo<Value>::getHashValue(operand));
    return h;
  }

  static bool isEqual(const AffineAccessExpr &a, const AffineAccessExpr &b) {
    return a == b;
  }
};
} // namespace llvm

namespace {
// This pass tries to prevent potential memory banking contention by hoisting
// memory reads after AffineParallelUnroll. It only hoists memory reads
// that occur *more than once* inside `scf.execute_region`s. Since
// AffineParallelUnroll converts loop indices to constants, this consecutive
// pass can safely analyze and remove redundant accesses.
struct MemoryBankConflictResolver {
  LogicalResult run(AffineParallelOp affineParallelOp);

  // Performs memory contention analysis on memory write operations, returning
  // `failure` if the pass identifies two write operations write to the same
  // memory reference at the same access indices in different parallel regions.
  LogicalResult writeOpAnalysis(AffineParallelOp affineParallelOp);

  // Tries to hoist memory read operations that will cause memory access
  // contention, such as reading from the same memory reference with the same
  // access indices in different parallel regions.
  LogicalResult readOpAnalysis(AffineParallelOp affineParallelOp);

  // Returns if `readOp`'s access indices are invariant with respect to
  // `affineParallelOp`.
  bool isInvariantToAffineParallel(AffineReadOpInterface readOp,
                                   AffineParallelOp affineParallelOp);

  // Accumulate all memory reads and writes to the fields.
  void accumulateReadWriteOps(AffineParallelOp affineParallelOp);

  // Stores all memory reads in current `affineParallelOp`.
  DenseSet<AffineReadOpInterface> allReadOps;
  // Stores all memory writes in current `affineParallelOp`.
  DenseSet<AffineWriteOpInterface> allWriteOps;
  // Stores the memory references across all `scf.execute_region` ops under
  // current `affineParallelOp`.
  DenseMap<Value, scf::ExecuteRegionOp> writtenMemRefs;
  // Counts the number of reads across all `scf.execute_region` ops under
  // current `affineParallelOp`.
  DenseMap<AffineAccessExpr, int> readOpCounts;
  // Records the memory reads that are already hoisted outside current
  // `affineParallelOp`.
  DenseMap<AffineAccessExpr, Value> hoistedReads;
};
} // end anonymous namespace

void MemoryBankConflictResolver::accumulateReadWriteOps(
    AffineParallelOp affineParallelOp) {
  auto executeRegionOps =
      affineParallelOp.getBody()->getOps<scf::ExecuteRegionOp>();
  for (auto executeRegionOp : executeRegionOps) {
    executeRegionOp.walk([&](Operation *op) {
      if (!isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
        return WalkResult::advance();

      if (auto read = dyn_cast<AffineReadOpInterface>(op)) {
        allReadOps.insert(read);

        AffineAccessExpr key{read.getMemRef(), read.getAffineMap(),
                             read.getMapOperands()};
        ++readOpCounts[key];
      } else {
        allWriteOps.insert(cast<AffineWriteOpInterface>(op));
      }

      return WalkResult::advance();
    });
  }
}

LogicalResult
MemoryBankConflictResolver::writeOpAnalysis(AffineParallelOp affineParallelOp) {
  for (auto writeOp : allWriteOps) {
    scf::ExecuteRegionOp parentExecuteRegion =
        writeOp->getParentOfType<scf::ExecuteRegionOp>();
    auto memref = writeOp.getMemRef();

    auto it = writtenMemRefs.find(memref);
    if (it != writtenMemRefs.end() && it->second != parentExecuteRegion) {
      writeOp.emitError("Multiple writes to the same memory reference");
      return failure();
    }

    writtenMemRefs[memref] = parentExecuteRegion;
  }

  return success();
}

bool MemoryBankConflictResolver::isInvariantToAffineParallel(
    AffineReadOpInterface readOp, AffineParallelOp affineParallelOp) {
  // Check if any operand depends on loop IVs
  for (Value iv : affineParallelOp.getIVs()) {
    for (Value operand : readOp.getMapOperands()) {
      if (operand == iv)
        return false;

      // Walk through defining operation chains, such as `affine.apply`s.
      if (Operation *def = operand.getDefiningOp()) {
        if (affineParallelOp->isAncestor(def) && !isa<arith::ConstantOp>(def)) {
          // Operand is computed inside the loop, not invariant
          return false;
        }
      }
    }
  }

  // Check if memref is written in loop
  Value memref = readOp.getMemRef();
  for (Operation *user : memref.getUsers()) {
    if (user == readOp.getOperation())
      continue;

    if (affineParallelOp->isAncestor(user) &&
        hasEffect<MemoryEffects::Write>(user, memref))
      return false;
  }

  return true;
}

LogicalResult
MemoryBankConflictResolver::readOpAnalysis(AffineParallelOp affineParallelOp) {
  OpBuilder builder(affineParallelOp);
  for (auto readOp : allReadOps) {
    AffineAccessExpr key{readOp.getMemRef(), readOp.getAffineMap(),
                         readOp.getMapOperands()};
    auto it = readOpCounts.find(key);
    if (it == readOpCounts.end() || it->second <= 1 ||
        !isInvariantToAffineParallel(readOp, affineParallelOp))
      continue;

    // Only hoist once per unique access
    auto hoistedReadsIt = hoistedReads.find(key);
    if (hoistedReadsIt == hoistedReads.end()) {
      builder.setInsertionPoint(affineParallelOp);
      Operation *cloned = builder.clone(*readOp.getOperation());
      hoistedReadsIt = hoistedReads.insert({key, cloned->getResult(0)}).first;
    }

    readOp->getResult(0).replaceAllUsesWith(hoistedReadsIt->second);
    readOp.getOperation()->erase();
  }

  return success();
}

LogicalResult
MemoryBankConflictResolver::run(AffineParallelOp affineParallelOp) {
  accumulateReadWriteOps(affineParallelOp);

  if (failed(writeOpAnalysis(affineParallelOp))) {
    return failure();
  }

  if (failed(readOpAnalysis(affineParallelOp))) {
    return failure();
  }

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
    auto newParallelOp = AffineParallelOp::create(
        rewriter, loc, /*resultTypes=*/TypeRange(),
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
          scf::ExecuteRegionOp::create(insideBuilder, loc, TypeRange{});
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
            arith::ConstantIndexOp::create(regionBuilder, loc, indices[i]);
        operandMap.map(pLoopIVs[i], ivConstant);
      }

      for (auto it = pLoopBody->begin(); it != std::prev(pLoopBody->end());
           ++it)
        regionBuilder.clone(*it, operandMap);

      // A terminator should always be inserted in `scf.execute_region`'s block.
      scf::YieldOp::create(regionBuilder, loc);

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
    registry.insert<mlir::memref::MemRefDialect>();
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

  // `AffineParallelUnroll` pattern introduces constant values, so running
  // `canonicalizePatterns` before `MemoryBankConflictResolver` will help ease
  // the analysis in `MemoryBankConflictResolver`.
  PassManager pm(ctx);
  pm.addPass(calyx::createExcludeExecuteRegionCanonicalizePass());

  if (failed(pm.run(getOperation()))) {
    getOperation()->emitError("Nested PassManager failed when running "
                              "ExcludeExecuteRegionCanonicalize pass.");
    signalPassFailure();
  }

  getOperation()->walk([&](AffineParallelOp parOp) {
    if (parOp->hasAttr("calyx.unroll")) {
      if (failed(MemoryBankConflictResolver().run(parOp))) {
        parOp.emitError("Failed to unroll");
        signalPassFailure();
      }
    }
  });
}

std::unique_ptr<mlir::Pass> circt::calyx::createAffineParallelUnrollPass() {
  return std::make_unique<AffineParallelUnrollPass>();
}
