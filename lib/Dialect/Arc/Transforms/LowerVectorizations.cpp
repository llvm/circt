//===- LowerVectorizations.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Arc/ArcPassesEnums.cpp.inc"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERVECTORIZATIONS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

/// Vectorizes the `arc.vectorize` boundary by packing the vector elements into
/// an integer value. Returns the vectorized version of the op. May invalidate
/// the passed operation.
///
/// Example:
/// ```mlir
/// %0:2 = arc.vectorize (%in0, %in1), (%in2, %in3) : (i1, i1, i1, i1) -> (i1,
/// i1) { ^bb0(%arg0: i1, %arg1: i1):
///   %1 = comb.and %arg0, %arg1 : i1
///   arc.vectorize.return %1 : i1
/// }
/// ```
/// becomes
/// ```mlir
/// %0 = comb.concat %in0, %in1 : i1, i1
/// %1 = comb.concat %in2, %in3 : i1, i1
/// %2 = arc.vectorize (%0), (%1) : (i2, i2) -> i2 {
/// ^bb0(%arg0: i1, %arg1: i1):
///   %11 = comb.and %arg0, %arg1 : i1
///   arc.vectorize.return %11 : i1
/// }
/// %3 = comb.extract %2 from 0 : (i2) -> i1
/// %4 = comb.extract %2 from 1 : (i2) -> i1
/// ```
static VectorizeOp lowerBoundaryScalar(VectorizeOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  SmallVector<ValueRange> vectors;

  for (ValueRange range : op.getInputs()) {
    unsigned bw = range.front().getType().getIntOrFloatBitWidth();
    vectors.push_back(
        comb::ConcatOp::create(builder,
                               builder.getIntegerType(bw * range.size()), range)
            ->getResults());
  }

  unsigned width = op->getResult(0).getType().getIntOrFloatBitWidth();
  VectorizeOp newOp = VectorizeOp::create(
      builder, builder.getIntegerType(width * op->getNumResults()), vectors);
  newOp.getBody().takeBody(op.getBody());

  for (OpResult res : op.getResults()) {
    Value newRes = comb::ExtractOp::create(
        builder, newOp.getResult(0), width * res.getResultNumber(), width);
    res.replaceAllUsesWith(newRes);
  }

  op->erase();
  return newOp;
}

/// Vectorizes the `arc.vectorize` boundary by using the `vector` type and
/// dialect for SIMD-based vectorization. Returns the vectorized version of the
/// op. May invalidate the passed operation.
///
/// Example:
/// ```mlir
/// %0:2 = arc.vectorize (%in0, %in1), (%in2, %in2) : (i64, i64, i32, i32) ->
/// (i64, i64) { ^bb0(%arg0: i64, %arg1: i32):
///   %c0_i32 = hw.constant 0 : i32
///   %1 = comb.concat %c0_i32, %arg1 : i32, i32
///   %2 = comb.and %arg0, %1 : i64
///   arc.vectorize.return %2 : i64
/// }
/// ```
/// becomes
/// ```mlir
/// %cst = arith.constant dense<0> : vector<2xi64>
/// %0 = vector.insert %in0, %cst [0] : i64 into vector<2xi64>
/// %1 = vector.insert %in1, %0 [1] : i64 into vector<2xi64>
/// %2 = vector.broadcast %in2 : i32 to vector<2xi32>
/// %3 = arc.vectorize (%1), (%2) : (vector<2xi64>, vector<2xi32>) ->
/// vector<2xi64> { ^bb0(%arg0: i64, %arg1: i32):
///   %c0_i32 = hw.constant 0 : i32
///   %4 = comb.concat %c0_i32, %arg1 : i32, i32
///   %5 = comb.and %arg0, %4 : i64
///   arc.vectorize.return %5 : i64
/// }
/// %4 = vector.extract %3[0] : vector<2xi64>
/// %5 = vector.extract %3[1] : vector<2xi64>
/// ```
static VectorizeOp lowerBoundaryVector(VectorizeOp op) {
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  SmallVector<ValueRange> vectors;

  for (ValueRange range : op.getInputs()) {
    // Insert a broadcast operation if all elements of a vector are the same
    // because it's a significantly cheaper instruction.
    VectorType type = VectorType::get(SmallVector<int64_t>(1, range.size()),
                                      range.front().getType());
    if (llvm::all_equal(range)) {
      vectors.push_back(
          vector::BroadcastOp::create(builder, type, range.front())
              ->getResults());
      continue;
    }

    // Otherwise do a gather.
    ValueRange vector =
        arith::ConstantOp::create(
            builder,
            DenseElementsAttr::get(
                type, SmallVector<Attribute>(
                          range.size(),
                          builder.getIntegerAttr(type.getElementType(), 0))))
            ->getResults();
    for (auto [i, element] : llvm::enumerate(range))
      vector = vector::InsertOp::create(builder, element, vector.front(), i)
                   ->getResults();

    vectors.push_back(vector);
  }

  VectorType resType = VectorType::get(
      SmallVector<int64_t>(1, op->getNumResults()), op.getResult(0).getType());
  VectorizeOp newOp = VectorizeOp::create(builder, resType, vectors);
  newOp.getBody().takeBody(op.getBody());

  for (OpResult res : op.getResults())
    res.replaceAllUsesWith(vector::ExtractOp::create(
        builder, newOp.getResult(0), res.getResultNumber()));

  op->erase();
  return newOp;
}

/// Vectorizes the boundary of the given `arc.vectorize` operation if it is not
/// already vectorized. If the body of the `arc.vectorize` operation is already
/// vectorized the same vectorization technique (SIMD or scalar) is chosen.
/// Otherwise,
///  * packs the vector in a scalar if it fits in a 64-bit integer or
///  * uses the `vector` type and dialect for SIMD vectorization
/// Returns the vectorized version of the op. May invalidate the passed
/// operation.
static VectorizeOp lowerBoundary(VectorizeOp op) {
  // Nothing to do if it is already vectorized.
  if (op.isBoundaryVectorized())
    return op;

  // If the body is already vectorized, we must use the same vectorization
  // technique. Otherwise, we would produce invalid IR.
  if (op.isBodyVectorized()) {
    if (isa<VectorType>(op.getBody().front().getArgumentTypes().front()))
      return lowerBoundaryVector(op);
    return lowerBoundaryScalar(op);
  }

  // If the vector can fit in an i64 value, use scalar vectorization, otherwise
  // use SIMD.
  unsigned numLanes = op.getInputs().size();
  unsigned maxLaneWidth = 0;
  for (OperandRange range : op.getInputs())
    maxLaneWidth =
        std::max(maxLaneWidth, range.front().getType().getIntOrFloatBitWidth());

  if ((numLanes * maxLaneWidth <= 64) &&
      op->getResult(0).getType().getIntOrFloatBitWidth() *
              op->getNumResults() <=
          64)
    return lowerBoundaryScalar(op);
  return lowerBoundaryVector(op);
}

/// Vectorizes the body of the given `arc.vectorize` operation if it is not
/// already vectorized. If the boundary of the `arc.vectorize` operation is
/// already vectorized the same vectorization technique (SIMD or scalar) is
/// chosen. Otherwise,
///  * packs the vector in a scalar if it fits in a 64-bit integer or
///  * uses the `vector` type and dialect for SIMD vectorization
/// Returns the vectorized version of the op or failure. May invalidate the
/// passed operation.
static FailureOr<VectorizeOp> lowerBody(VectorizeOp op) {
  if (op.isBodyVectorized())
    return op;

  return op->emitError("lowering body not yet supported");
}

/// Inlines the `arc.vectorize` operations body once both the boundary and body
/// are vectorized.
///
/// Example:
/// ```mlir
/// %0 = comb.concat %in0, %in1 : i1, i1
/// %1 = comb.concat %in2, %in2 : i1, i1
/// %2 = arc.vectorize (%0), (%1) : (i2, i2) -> i2 {
/// ^bb0(%arg0: i2, %arg1: i2):
///   %12 = arith.andi %arg0, %arg1 : i2
///   arc.vectorize.return %12 : i2
/// }
/// %3 = comb.extract %2 from 0 : (i2) -> i1
/// %4 = comb.extract %2 from 1 : (i2) -> i1
/// ```
/// becomes
/// ```mlir
/// %0 = comb.concat %in0, %in1 : i1, i1
/// %1 = comb.concat %in2, %in2 : i1, i1
/// %2 = arith.andi %0, %1 : i2
/// %3 = comb.extract %2 from 0 : (i2) -> i1
/// %4 = comb.extract %2 from 1 : (i2) -> i1
/// ```
static LogicalResult inlineBody(VectorizeOp op) {
  if (!(op.isBodyVectorized() && op.isBoundaryVectorized()))
    return op->emitError(
        "can only inline body if boundary and body are already vectorized");

  Block &block = op.getBody().front();
  for (auto [operand, arg] : llvm::zip(op.getInputs(), block.getArguments()))
    arg.replaceAllUsesWith(operand.front());

  Operation *terminator = block.getTerminator();
  op->getResult(0).replaceAllUsesWith(terminator->getOperand(0));
  terminator->erase();

  op->getBlock()->getOperations().splice(op->getIterator(),
                                         block.getOperations());
  op->erase();

  return success();
}

namespace {
/// A pass to vectorize (parts of) an `arc.vectorize` operation.
struct LowerVectorizationsPass
    : public arc::impl::LowerVectorizationsBase<LowerVectorizationsPass> {
  LowerVectorizationsPass() = default;
  explicit LowerVectorizationsPass(LowerVectorizationsModeEnum mode)
      : LowerVectorizationsPass() {
    this->mode.setValue(mode);
  }

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](VectorizeOp op) -> WalkResult {
      switch (mode) {
      case LowerVectorizationsModeEnum::Full:
        if (auto newOp = lowerBody(lowerBoundary(op));
            succeeded(newOp) && succeeded(inlineBody(*newOp)))
          return WalkResult::advance();
        return WalkResult::interrupt();
      case LowerVectorizationsModeEnum::Boundary:
        return lowerBoundary(op), WalkResult::advance();
      case LowerVectorizationsModeEnum::Body:
        return static_cast<LogicalResult>(lowerBody(op));
      case LowerVectorizationsModeEnum::InlineBody:
        return inlineBody(op);
      }
      llvm_unreachable("all enum cases must be handled above");
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
arc::createLowerVectorizationsPass(LowerVectorizationsModeEnum mode) {
  return std::make_unique<LowerVectorizationsPass>(mode);
}
