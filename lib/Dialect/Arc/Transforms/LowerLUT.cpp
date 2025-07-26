//===- LowerLUT.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-lut"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERLUT
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//

namespace {

/// Allows to compute the constant lookup-table entries given the LutOp
/// operation and caches the result. Also provides additional utility functions
/// related to lookup-table materialization.
class LutCalculator {
public:
  /// Compute all the lookup-table enties if they haven't already been computed
  /// and cache the results. Note that calling this function is very expensive
  /// in terms of runtime as it calls the constant folders of all operations
  /// inside the LutOp for all possible input values.
  LogicalResult computeTableEntries(LutOp lut);

  /// Get a reference to the cached lookup-table entries. `computeTableEntries`
  /// has to be called before calling this function.
  ArrayRef<IntegerAttr> getRefToTableEntries();
  /// Get a copy of the cached lookup-table entries. `computeTableEntries` has
  /// to be called before calling this function.
  void getCopyOfTableEntries(SmallVector<IntegerAttr> &tableEntries);
  /// Materialize uniqued hw::ConstantOp operations for all cached lookup-table
  /// entries. `computeTableEntries` has to be called before calling this
  /// function.
  void getTableEntriesAsConstValues(OpBuilder &builder,
                                    SmallVector<Value> &tableEntries);
  /// Compute and return the total size of the table in bits.
  uint32_t getTableSize();
  /// Compute and return the summed up bit-width of all input values.
  uint32_t getInputBitWidth();

private:
  LutOp lut;
  SmallVector<IntegerAttr> table;
};

} // namespace

//===----------------------------------------------------------------------===//
// Data structure implementations
//===----------------------------------------------------------------------===//

// Note that this function is very expensive in terms of runtime since it
// computes the LUT entries by calling the operation's folders
// O(2^inputBitWidth) times.
LogicalResult LutCalculator::computeTableEntries(LutOp lut) {
  // If we already have precomputed the entries for this LUT operation, we don't
  // need to re-compute it. This is important, because the dialect conversion
  // framework may try several lowering patterns for the same LutOp after
  // another and recomputing it every time would be very expensive.
  if (this->lut == lut && !table.empty())
    return success();

  // Cache this LUT to be able to apply above shortcut next time and clear the
  // currently cached table entries from a previous LUT.
  this->lut = lut;
  table.clear();

  // Allocate memory
  DenseMap<Value, SmallVector<Attribute>> vals;
  const uint32_t bw = getInputBitWidth();

  for (auto arg : lut.getBodyBlock()->getArguments())
    vals[arg] = SmallVector<Attribute>(1U << bw);

  for (auto &operation : lut.getBodyBlock()->without_terminator()) {
    for (auto operand : operation.getResults()) {
      if (vals.count(operand))
        continue;
      vals[operand] = SmallVector<Attribute>(1U << bw);
    }
  }

  // Initialize inputs
  for (int i = 0; i < (1 << bw); ++i) {
    const APInt input(bw, i);
    size_t offset = bw;
    for (auto arg : lut.getBodyBlock()->getArguments()) {
      const unsigned argBitWidth = arg.getType().getIntOrFloatBitWidth();
      offset -= argBitWidth;
      vals[arg][i] = IntegerAttr::get(arg.getType(),
                                      input.extractBits(argBitWidth, offset));
    }
  }

  for (auto &operation : lut.getBodyBlock()->without_terminator()) {
    // We need to rearange the vectors to use the operation folers. There is
    // probably still some potential for optimization here.
    SmallVector<SmallVector<Attribute>, 8> constants(1U << bw);
    for (size_t j = 0, e = operation.getNumOperands(); j < e; ++j) {
      SmallVector<Attribute> &tmp = vals[operation.getOperand(j)];
      for (int i = (1U << bw) - 1; i >= 0; i--)
        constants[i].push_back(tmp[i]);
    }

    // Call the operation folders
    SmallVector<SmallVector<OpFoldResult>, 8> results(
        1U << bw, SmallVector<OpFoldResult, 8>());
    for (int i = (1U << bw) - 1; i >= 0; i--) {
      if (failed(operation.fold(constants[i], results[i]))) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to fold operation '";
                   operation.print(llvm::dbgs()); llvm::dbgs() << "'\n");
        return failure();
      }
    }

    // Store the folder's results in the value map.
    for (size_t i = 0, e = operation.getNumResults(); i < e; ++i) {
      SmallVector<Attribute> &ref = vals[operation.getResult(i)];
      for (int j = (1U << bw) - 1; j >= 0; j--) {
        Attribute foldAttr;
        if (!(foldAttr = dyn_cast<Attribute>(results[j][i])))
          foldAttr = vals[llvm::cast<Value>(results[j][i])][j];
        ref[j] = foldAttr;
      }
    }
  }

  // Store the LUT's output values in the correct order in the table entry
  // cache.
  auto outValue = lut.getBodyBlock()->getTerminator()->getOperand(0);
  for (int j = (1U << bw) - 1; j >= 0; j--)
    table.push_back(cast<IntegerAttr>(vals[outValue][j]));

  return success();
}

ArrayRef<IntegerAttr> LutCalculator::getRefToTableEntries() { return table; }

void LutCalculator::getCopyOfTableEntries(
    SmallVector<IntegerAttr> &tableEntries) {
  tableEntries.append(table);
}

void LutCalculator::getTableEntriesAsConstValues(
    OpBuilder &builder, SmallVector<Value> &tableEntries) {
  // Since LUT entries tend to have a very small bit-width (mostly 1-3 bits),
  // there are many duplicate constants. Creating a single constant operation
  // for each unique number saves us a lot of CSE afterwards.
  DenseMap<IntegerAttr, Value> map;
  for (auto entry : table) {
    if (!map.count(entry))
      map[entry] = hw::ConstantOp::create(builder, lut.getLoc(), entry);

    tableEntries.push_back(map[entry]);
  }
}

uint32_t LutCalculator::getInputBitWidth() {
  unsigned bw = 0;
  for (auto val : lut.getInputs())
    bw += cast<IntegerType>(val.getType()).getWidth();
  return bw;
}

uint32_t LutCalculator::getTableSize() {
  return (1 << getInputBitWidth()) *
         lut.getOutput().getType().getIntOrFloatBitWidth();
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

/// Lower lookup-tables that have a total size of less than 256 bits to an
/// integer that is shifed and truncated according to the lookup/index value.
/// Encoding the lookup tables as intermediate values in the instruction stream
/// should provide better performnace than loading from some global constant.
struct LutToInteger : OpConversionPattern<LutOp> {
  LutToInteger(LutCalculator &calculator, MLIRContext *context)
      : OpConversionPattern<LutOp>(context), lutCalculator(calculator) {}

  LogicalResult
  matchAndRewrite(LutOp lut, LutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (failed(lutCalculator.computeTableEntries(lut)))
      return failure();

    const uint32_t tableSize = lutCalculator.getTableSize();
    const uint32_t inputBw = lutCalculator.getInputBitWidth();

    if (tableSize > 256)
      return failure();

    // Concatenate the lookup table entries to a single integer.
    auto constants = lutCalculator.getRefToTableEntries();
    APInt result(tableSize, 0);
    unsigned nextInsertion = tableSize;

    for (auto attr : constants) {
      auto chunk = attr.getValue();
      nextInsertion -= chunk.getBitWidth();
      result.insertBits(chunk, nextInsertion);
    }

    Value table = hw::ConstantOp::create(rewriter, lut.getLoc(), result);

    // Zero-extend the lookup/index value to the same bit-width as the table,
    // because the shift operation requires both operands to have the same
    // bit-width.
    Value zextValue =
        hw::ConstantOp::create(rewriter, lut->getLoc(),
                               rewriter.getIntegerType(tableSize - inputBw), 0);
    Value entryOffset = comb::ConcatOp::create(rewriter, lut.getLoc(),
                                               zextValue, lut.getInputs());
    Value resultBitWidth = hw::ConstantOp::create(
        rewriter, lut.getLoc(), entryOffset.getType(),
        lut.getResult().getType().getIntOrFloatBitWidth());
    Value lookupValue = comb::MulOp::create(rewriter, lut.getLoc(), entryOffset,
                                            resultBitWidth);

    // Shift the table and truncate to the bitwidth of the output value.
    Value shiftedTable =
        comb::ShrUOp::create(rewriter, lut->getLoc(), table, lookupValue);
    const Value extracted = comb::ExtractOp::create(
        rewriter, lut.getLoc(), shiftedTable, 0,
        lut.getOutput().getType().getIntOrFloatBitWidth());

    rewriter.replaceOp(lut, extracted);
    return success();
  }

  LutCalculator &lutCalculator;
};

/// Lower lookup-tables with a total size bigger than 256 bits to a constant
/// array that is stored as constant global data and thus a lookup consists of a
/// memory load at the correct offset of that global data frame.
struct LutToArray : OpConversionPattern<LutOp> {
  LutToArray(LutCalculator &calculator, MLIRContext *context)
      : OpConversionPattern<LutOp>(context), lutCalculator(calculator) {}

  LogicalResult
  matchAndRewrite(LutOp lut, LutOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (failed(lutCalculator.computeTableEntries(lut)))
      return failure();

    auto constants = lutCalculator.getRefToTableEntries();
    SmallVector<Attribute> constantAttrs(constants.begin(), constants.end());
    auto tableSize = lutCalculator.getTableSize();
    auto inputBw = lutCalculator.getInputBitWidth();

    if (tableSize <= 256)
      return failure();

    Value table = hw::AggregateConstantOp::create(
        rewriter, lut.getLoc(),
        hw::ArrayType::get(lut.getType(), constantAttrs.size()),
        rewriter.getArrayAttr(constantAttrs));
    Value lookupValue = comb::ConcatOp::create(rewriter, lut.getLoc(),
                                               rewriter.getIntegerType(inputBw),
                                               lut.getInputs());
    const Value extracted =
        hw::ArrayGetOp::create(rewriter, lut.getLoc(), table, lookupValue);

    rewriter.replaceOp(lut, extracted);
    return success();
  }

  LutCalculator &lutCalculator;
};

} // namespace

//===----------------------------------------------------------------------===//
// Lower LUT pass
//===----------------------------------------------------------------------===//

namespace {

/// Lower LutOp operations to comb and hw operations.
struct LowerLUTPass : public arc::impl::LowerLUTBase<LowerLUTPass> {
  void runOnOperation() override;
};

} // namespace

void LowerLUTPass::runOnOperation() {
  MLIRContext &context = getContext();
  ConversionTarget target(context);
  RewritePatternSet patterns(&context);
  target.addLegalDialect<comb::CombDialect, hw::HWDialect, arc::ArcDialect>();
  target.addIllegalOp<arc::LutOp>();

  // TODO: This class could be factored out into an analysis if there is a need
  // to access precomputed lookup-tables in some other pass.
  LutCalculator lutCalculator;
  patterns.add<LutToInteger, LutToArray>(lutCalculator, &context);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> arc::createLowerLUTPass() {
  return std::make_unique<LowerLUTPass>();
}
