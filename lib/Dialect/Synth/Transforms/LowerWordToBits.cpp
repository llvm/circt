//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements bit-blasting for logic synthesis operations.
// It converts multi-bit operations (AIG, MIG, combinatorial) into equivalent
// single-bit operations, enabling more efficient synthesis and optimization.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Synth/SynthOps.h"
#include "circt/Dialect/Synth/Transforms/SynthPasses.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "synth-lower-word-to-bits"

namespace circt {
namespace synth {
#define GEN_PASS_DEF_LOWERWORDTOBITS
#include "circt/Dialect/Synth/Transforms/SynthPasses.h.inc"
} // namespace synth
} // namespace circt

using namespace circt;
using namespace synth;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// Check if an operation should be lowered to bit-level operations.
static bool shouldLowerOperation(Operation *op) {
  return isa<aig::AndInverterOp, mig::MajorityInverterOp, comb::AndOp,
             comb::OrOp, comb::XorOp>(op);
}

namespace {

//===----------------------------------------------------------------------===//
// BitBlaster - Bit-level lowering implementation
//===----------------------------------------------------------------------===//

/// The BitBlaster class implements the core bit-blasting algorithm.
/// It manages the lowering of multi-bit operations to single-bit operations
/// while maintaining correctness and optimizing for constant propagation.
class BitBlaster {
public:
  explicit BitBlaster(hw::HWModuleOp moduleOp) : moduleOp(moduleOp) {}

  /// Run the bit-blasting algorithm on the module.
  LogicalResult run();

  //===--------------------------------------------------------------------===//
  // Statistics
  //===--------------------------------------------------------------------===//

  /// Number of bits that were lowered from multi-bit to single-bit operations
  size_t numLoweredBits = 0;

  /// Number of constant bits that were identified and optimized
  size_t numLoweredConstants = 0;

  /// Number of operations that were lowered
  size_t numLoweredOps = 0;

private:
  //===--------------------------------------------------------------------===//
  // Core Lowering Methods
  //===--------------------------------------------------------------------===//

  /// Lower a multi-bit value to individual bits.
  /// This is the main entry point for bit-blasting a value.
  ArrayRef<Value> lowerValueToBits(Value value);
  template <typename OpTy>
  ArrayRef<Value> lowerInvertibleOperations(OpTy op);
  template <typename OpTy>
  ArrayRef<Value> lowerCombOperations(OpTy op);
  ArrayRef<Value>
  lowerOp(Operation *op,
          llvm::function_ref<Value(OpBuilder &builder, ValueRange)> createOp);

  /// Extract a specific bit from a value.
  /// Handles various IR constructs that can represent bit extraction.
  Value extractBit(Value value, size_t index);

  /// Compute and cache known bits for a value.
  /// Uses operation-specific logic to determine which bits are constants.
  const llvm::KnownBits &computeKnownBits(Value value);

  /// Get or create a boolean constant (0 or 1).
  /// Constants are cached to avoid duplication.
  Value getBoolConstant(bool value);

  //===--------------------------------------------------------------------===//
  // Helper Methods
  //===--------------------------------------------------------------------===//

  /// Insert lowered bits into the cache.
  ArrayRef<Value> insertBits(Value value, SmallVector<Value> bits) {
    auto it = loweredValues.insert({value, std::move(bits)});
    assert(it.second && "value already inserted");
    return it.first->second;
  }

  /// Insert computed known bits into the cache.
  const llvm::KnownBits &insertKnownBits(Value value, llvm::KnownBits bits) {
    auto it = knownBits.insert({value, std::move(bits)});
    return it.first->second;
  }

  /// Cache for lowered values (multi-bit -> vector of single-bit values)
  llvm::MapVector<Value, SmallVector<Value>> loweredValues;

  /// Cache for computed known bits information
  llvm::MapVector<Value, llvm::KnownBits> knownBits;

  /// Cached boolean constants (false at index 0, true at index 1)
  std::array<Value, 2> constants;

  /// Reference to the module being processed
  hw::HWModuleOp moduleOp;
};

} // namespace

//===----------------------------------------------------------------------===//
// BitBlaster Implementation
//===----------------------------------------------------------------------===//

const llvm::KnownBits &BitBlaster::computeKnownBits(Value value) {
  // Check cache first
  auto *it = knownBits.find(value);
  if (it != knownBits.end())
    return it->second;

  auto width = hw::getBitWidth(value.getType());
  assert(width && "value type must have a known bit width");
  auto *op = value.getDefiningOp();

  // For block arguments, return unknown bits
  if (!op)
    return insertKnownBits(value, llvm::KnownBits(*width));

  llvm::KnownBits result(*width);
  if (auto aig = dyn_cast<aig::AndInverterOp>(op)) {
    // Initialize to all ones for AND operation
    result.One = APInt::getAllOnes(width);
    result.Zero = APInt::getZero(width);

    for (auto [operand, inverted] :
         llvm::zip(aig.getInputs(), aig.getInverted())) {
      auto operandKnownBits = computeKnownBits(operand);
      if (inverted)
        // Complement the known bits by swapping Zero and One
        std::swap(operandKnownBits.Zero, operandKnownBits.One);
      result &= operandKnownBits;
    }
  } else if (auto mig = dyn_cast<mig::MajorityInverterOp>(op)) {
    // Give up if it's not a 3-input majority inverter.
    if (mig.getNumOperands() == 3) {
      std::array<llvm::KnownBits, 3> operandsKnownBits;
      for (auto [i, operand, inverted] :
           llvm::enumerate(mig.getInputs(), mig.getInverted())) {
        operandsKnownBits[i] = computeKnownBits(operand);
        // Complement the known bits by swapping Zero and One
        if (inverted)
          std::swap(operandsKnownBits[i].Zero, operandsKnownBits[i].One);
      }

      result = (operandsKnownBits[0] & operandsKnownBits[1]) |
               (operandsKnownBits[0] & operandsKnownBits[2]) |
               (operandsKnownBits[1] & operandsKnownBits[2]);
    }
  } else {
    // For other operations, use the standard known bits computation
    // TODO: This is not optimal as it has a depth limit and does not check
    // cached results.
    result = comb::computeKnownBits(value);
  }

  return insertKnownBits(value, std::move(result));
}

Value BitBlaster::extractBit(Value value, size_t index) {
  auto width = hw::getBitWidth(value.getType());
  assert(width && "value type must have a known bit width");
  if (*width <= 1)
    return value;

  auto *op = value.getDefiningOp();

  // If the value is a block argument, extract the bit.
  if (!op)
    return lowerValueToBits(value)[index];

  return TypeSwitch<Operation *, Value>(op)
      .Case<comb::ConcatOp>([&](comb::ConcatOp op) {
        for (auto operand : llvm::reverse(op.getOperands())) {
          auto width = hw::getBitWidth(operand.getType());
          assert(width && "operand type must have a known bit width");
          if (index < *width)
            return extractBit(operand, index);
          index -= *width;
        }
        llvm_unreachable("index out of bounds");
      })
      .Case<comb::ExtractOp>([&](comb::ExtractOp ext) {
        return extractBit(ext.getInput(),
                          static_cast<size_t>(ext.getLowBit()) + index);
      })
      .Case<comb::ReplicateOp>([&](comb::ReplicateOp op) {
        auto operandWidth = hw::getBitWidth(op.getOperand().getType());
        assert(operandWidth && "operand type must have a known bit width");
        return extractBit(op.getInput(), index % *operandWidth);
      })
      .Case<hw::ConstantOp>([&](hw::ConstantOp op) {
        auto value = op.getValue();
        return getBoolConstant(value[index]);
      })
      .Default([&](auto op) { return lowerValueToBits(value)[index]; });
}

ArrayRef<Value> BitBlaster::lowerValueToBits(Value value) {
  auto *it = loweredValues.find(value);
  if (it != loweredValues.end())
    return it->second;

  auto width = hw::getBitWidth(value.getType());
  assert(width && "value type must have a known bit width");
  if (*width <= 1)
    return insertBits(value, {value});

  auto *op = value.getDefiningOp();
  if (!op) {
    SmallVector<Value> results;
    OpBuilder builder(value.getContext());
    builder.setInsertionPointAfterValue(value);
    comb::extractBits(builder, value, results);
    return insertBits(value, std::move(results));
  }

  return TypeSwitch<Operation *, ArrayRef<Value>>(op)
      .Case<aig::AndInverterOp, mig::MajorityInverterOp>(
          [&](auto op) { return lowerInvertibleOperations(op); })
      .Case<comb::AndOp, comb::OrOp, comb::XorOp>(
          [&](auto op) { return lowerCombOperations(op); })
      .Default([&](auto op) {
        OpBuilder builder(value.getContext());
        builder.setInsertionPoint(op);
        SmallVector<Value> results;
        comb::extractBits(builder, value, results);

        return insertBits(value, std::move(results));
      });
}

LogicalResult BitBlaster::run() {
  // Topologically sort operations in graph regions so that walk visits them in
  // the topological order.
  if (failed(topologicallySortGraphRegionBlocks(
          moduleOp, [](Value value, Operation *op) -> bool {
            // Otherthan target ops, all other ops are always ready.
            return !(shouldLowerOperation(op) ||
                     isa<comb::ExtractOp, comb::ReplicateOp, comb::ConcatOp,
                         comb::ReplicateOp>(op));
          }))) {
    // If we failed to topologically sort operations we cannot proceed.
    return mlir::emitError(moduleOp.getLoc(), "there is a combinational cycle");
  }

  // Lower target operations
  moduleOp.walk([&](Operation *op) {
    // If the block is in a graph region, topologically sort it first.
    if (shouldLowerOperation(op))
      (void)lowerValueToBits(op->getResult(0));
  });

  // Replace operations with concatenated results if needed
  for (auto &[value, results] :
       llvm::make_early_inc_range(llvm::reverse(loweredValues))) {
    auto width = hw::getBitWidth(value.getType());
    assert(width && "value type must have a known bit width");
    if (*width <= 1)
      continue;

    auto *op = value.getDefiningOp();
    if (!op)
      continue;

    if (value.use_empty()) {
      op->erase();
      continue;
    }

    // If a target operation still has an use (e.g. connected to output or
    // instance), replace the value with the concatenated result.
    if (shouldLowerOperation(op)) {
      OpBuilder builder(op);
      std::reverse(results.begin(), results.end());
      auto concat = comb::ConcatOp::create(builder, value.getLoc(), results);
      value.replaceAllUsesWith(concat);
      op->erase();
    }
  }

  return success();
}

Value BitBlaster::getBoolConstant(bool value) {
  if (!constants[value]) {
    auto builder = OpBuilder::atBlockBegin(moduleOp.getBodyBlock());
    constants[value] = hw::ConstantOp::create(builder, builder.getUnknownLoc(),
                                              builder.getI1Type(), value);
  }
  return constants[value];
}

template <typename OpTy>
ArrayRef<Value> BitBlaster::lowerInvertibleOperations(OpTy op) {
  auto createOp = [&](OpBuilder &builder, ValueRange operands) {
    return builder.createOrFold<OpTy>(op.getLoc(), operands, op.getInverted());
  };
  return lowerOp(op, createOp);
}

template <typename OpTy>
ArrayRef<Value> BitBlaster::lowerCombOperations(OpTy op) {
  auto createOp = [&](OpBuilder &builder, ValueRange operands) {
    return builder.createOrFold<OpTy>(op.getLoc(), operands,
                                      op.getTwoStateAttr());
  };
  return lowerOp(op, createOp);
}

ArrayRef<Value> BitBlaster::lowerOp(
    Operation *op,
    llvm::function_ref<Value(OpBuilder &builder, ValueRange)> createOp) {
  auto value = op->getResult(0);
  OpBuilder builder(op);
  auto width = hw::getBitWidth(value.getType());
  assert(width && "value type must have a known bit width");
  assert(*width > 1 && "expected multi-bit operation");

  auto known = computeKnownBits(value);
  APInt knownMask = known.Zero | known.One;

  // Update statistics
  numLoweredConstants += knownMask.popcount();
  numLoweredBits += width;
  ++numLoweredOps;

  SmallVector<Value> results;
  results.reserve(width);

  for (int64_t i = 0; i < width; ++i) {
    SmallVector<Value> operands;
    operands.reserve(op->getNumOperands());
    if (knownMask[i]) {
      // Use known constant value
      results.push_back(getBoolConstant(known.One[i]));
      continue;
    }

    // Extract the i-th bit from each operand
    for (auto operand : op->getOperands())
      operands.push_back(extractBit(operand, i));

    // Create the single-bit operation
    auto result = createOp(builder, operands);
    results.push_back(result);

    // Add name hint if present
    if (auto name = op->getAttrOfType<StringAttr>("sv.namehint")) {
      auto newName = StringAttr::get(
          op->getContext(), name.getValue() + "[" + std::to_string(i) + "]");
      if (auto *loweredOp = result.getDefiningOp())
        loweredOp->setAttr("sv.namehint", newName);
    }
  }

  assert(results.size() == static_cast<size_t>(width));
  return insertBits(value, std::move(results));
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerWordToBitsPass
    : public impl::LowerWordToBitsBase<LowerWordToBitsPass> {
  void runOnOperation() override;
};
} // namespace

void LowerWordToBitsPass::runOnOperation() {
  BitBlaster driver(getOperation());
  if (failed(driver.run()))
    return signalPassFailure();

  // Update statistics
  numLoweredBits += driver.numLoweredBits;
  numLoweredConstants += driver.numLoweredConstants;
  numLoweredOps += driver.numLoweredOps;
}
