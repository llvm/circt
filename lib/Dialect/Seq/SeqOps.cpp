//===- SeqOps.cpp - Implement the Seq operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements sequential ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "circt/Support/FoldUtils.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "circt/Dialect/HW/HWTypes.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace circt;
using namespace seq;

bool circt::seq::isValidIndexValues(Value hlmemHandle, ValueRange addresses) {
  auto memType = cast<seq::HLMemType>(hlmemHandle.getType());
  auto shape = memType.getShape();
  if (shape.size() != addresses.size())
    return false;

  for (auto [dim, addr] : llvm::zip(shape, addresses)) {
    auto addrType = dyn_cast<IntegerType>(addr.getType());
    if (!addrType)
      return false;
    if (addrType.getIntOrFloatBitWidth() != llvm::Log2_64_Ceil(dim))
      return false;
  }
  return true;
}

// If there was no name specified, check to see if there was a useful name
// specified in the asm file.
static void setNameFromResult(OpAsmParser &parser, OperationState &result) {
  if (result.attributes.getNamed("name"))
    return;
  // If there is no explicit name attribute, get it from the SSA result name.
  // If numeric, just use an empty name.
  StringRef resultName = parser.getResultName(0).first;
  if (!resultName.empty() && isdigit(resultName[0]))
    resultName = "";
  result.addAttribute("name", parser.getBuilder().getStringAttr(resultName));
}

static bool canElideName(OpAsmPrinter &p, Operation *op) {
  if (!op->hasAttr("name"))
    return true;

  auto name = op->getAttrOfType<StringAttr>("name").getValue();
  if (name.empty())
    return true;

  SmallString<32> resultNameStr;
  llvm::raw_svector_ostream tmpStream(resultNameStr);
  p.printOperand(op->getResult(0), tmpStream);
  auto actualName = tmpStream.str().drop_front();
  return actualName == name;
}

static ParseResult
parseOptionalTypeMatch(OpAsmParser &parser, Type refType,
                       std::optional<OpAsmParser::UnresolvedOperand> operand,
                       Type &type) {
  if (operand)
    type = refType;
  return success();
}

static void printOptionalTypeMatch(OpAsmPrinter &p, Operation *op, Type refType,
                                   Value operand, Type type) {
  // Nothing to do - this is strictly an implicit parsing helper.
}

static ParseResult parseOptionalImmutableTypeMatch(
    OpAsmParser &parser, Type refType,
    std::optional<OpAsmParser::UnresolvedOperand> operand, Type &type) {
  if (operand)
    type = seq::ImmutableType::get(refType);
  return success();
}

static void printOptionalImmutableTypeMatch(OpAsmPrinter &p, Operation *op,
                                            Type refType, Value operand,
                                            Type type) {
  // Nothing to do - this is strictly an implicit parsing helper.
}

//===----------------------------------------------------------------------===//
// ReadPortOp
//===----------------------------------------------------------------------===//

ParseResult ReadPortOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc loc = parser.getCurrentLocation();

  OpAsmParser::UnresolvedOperand memOperand, rdenOperand;
  bool hasRdEn = false;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 2> addressOperands;
  seq::HLMemType memType;

  if (parser.parseOperand(memOperand) ||
      parser.parseOperandList(addressOperands, OpAsmParser::Delimiter::Square))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("rden"))) {
    if (failed(parser.parseOperand(rdenOperand)))
      return failure();
    hasRdEn = true;
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(memType))
    return failure();

  llvm::SmallVector<Type> operandTypes = memType.getAddressTypes();
  operandTypes.insert(operandTypes.begin(), memType);

  llvm::SmallVector<OpAsmParser::UnresolvedOperand> allOperands = {memOperand};
  llvm::copy(addressOperands, std::back_inserter(allOperands));
  if (hasRdEn) {
    operandTypes.push_back(parser.getBuilder().getI1Type());
    allOperands.push_back(rdenOperand);
  }

  if (parser.resolveOperands(allOperands, operandTypes, loc, result.operands))
    return failure();

  result.addTypes(memType.getElementType());

  llvm::SmallVector<int32_t, 2> operandSizes;
  operandSizes.push_back(1); // memory handle
  operandSizes.push_back(addressOperands.size());
  operandSizes.push_back(hasRdEn ? 1 : 0);
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(operandSizes));
  return success();
}

void ReadPortOp::print(OpAsmPrinter &p) {
  p << " " << getMemory() << "[" << getAddresses() << "]";
  if (getRdEn())
    p << " rden " << getRdEn();
  p.printOptionalAttrDict((*this)->getAttrs(), {"operandSegmentSizes"});
  p << " : " << getMemory().getType();
}

void ReadPortOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  auto memName = getMemory().getDefiningOp<seq::HLMemOp>().getName();
  setNameFn(getReadData(), (memName + "_rdata").str());
}

void ReadPortOp::build(OpBuilder &builder, OperationState &result, Value memory,
                       ValueRange addresses, Value rdEn, unsigned latency) {
  auto memType = cast<seq::HLMemType>(memory.getType());
  ReadPortOp::build(builder, result, memType.getElementType(), memory,
                    addresses, rdEn, latency);
}

//===----------------------------------------------------------------------===//
// WritePortOp
//===----------------------------------------------------------------------===//

ParseResult WritePortOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  OpAsmParser::UnresolvedOperand memOperand, dataOperand, wrenOperand;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 2> addressOperands;
  seq::HLMemType memType;

  if (parser.parseOperand(memOperand) ||
      parser.parseOperandList(addressOperands,
                              OpAsmParser::Delimiter::Square) ||
      parser.parseOperand(dataOperand) || parser.parseKeyword("wren") ||
      parser.parseOperand(wrenOperand) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(memType))
    return failure();

  llvm::SmallVector<Type> operandTypes = memType.getAddressTypes();
  operandTypes.insert(operandTypes.begin(), memType);
  operandTypes.push_back(memType.getElementType());
  operandTypes.push_back(parser.getBuilder().getI1Type());

  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 2> allOperands(
      addressOperands);
  allOperands.insert(allOperands.begin(), memOperand);
  allOperands.push_back(dataOperand);
  allOperands.push_back(wrenOperand);

  if (parser.resolveOperands(allOperands, operandTypes, loc, result.operands))
    return failure();

  return success();
}

void WritePortOp::print(OpAsmPrinter &p) {
  p << " " << getMemory() << "[" << getAddresses() << "] " << getInData()
    << " wren " << getWrEn();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << getMemory().getType();
}

//===----------------------------------------------------------------------===//
// HLMemOp
//===----------------------------------------------------------------------===//

void HLMemOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getHandle(), getName());
}

void HLMemOp::build(OpBuilder &builder, OperationState &result, Value clk,
                    Value rst, StringRef name, llvm::ArrayRef<int64_t> shape,
                    Type elementType) {
  HLMemType t = HLMemType::get(builder.getContext(), shape, elementType);
  HLMemOp::build(builder, result, t, clk, rst, name);
}

//===----------------------------------------------------------------------===//
// FIFOOp
//===----------------------------------------------------------------------===//

// Flag threshold custom directive
static ParseResult parseFIFOFlagThreshold(OpAsmParser &parser,
                                          IntegerAttr &threshold,
                                          Type &outputFlagType,
                                          StringRef directive) {
  // look for an optional "almost_full $threshold" group.
  if (succeeded(parser.parseOptionalKeyword(directive))) {
    int64_t thresholdValue;
    if (succeeded(parser.parseInteger(thresholdValue))) {
      threshold = parser.getBuilder().getI64IntegerAttr(thresholdValue);
      outputFlagType = parser.getBuilder().getI1Type();
      return success();
    }
    return parser.emitError(parser.getNameLoc(),
                            "expected integer value after " + directive +
                                " directive");
  }
  return success();
}

ParseResult parseFIFOAFThreshold(OpAsmParser &parser, IntegerAttr &threshold,
                                 Type &outputFlagType) {
  return parseFIFOFlagThreshold(parser, threshold, outputFlagType,
                                "almost_full");
}

ParseResult parseFIFOAEThreshold(OpAsmParser &parser, IntegerAttr &threshold,
                                 Type &outputFlagType) {
  return parseFIFOFlagThreshold(parser, threshold, outputFlagType,
                                "almost_empty");
}

void printFIFOAFThreshold(OpAsmPrinter &p, Operation *op, IntegerAttr threshold,
                          Type outputFlagType) {
  if (threshold)
    p << "almost_full"
      << " " << threshold.getInt();
}

void printFIFOAEThreshold(OpAsmPrinter &p, Operation *op, IntegerAttr threshold,
                          Type outputFlagType) {
  if (threshold)
    p << "almost_empty"
      << " " << threshold.getInt();
}

void FIFOOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getOutput(), "out");
  setNameFn(getEmpty(), "empty");
  setNameFn(getFull(), "full");
  if (auto ae = getAlmostEmpty())
    setNameFn(ae, "almostEmpty");
  if (auto af = getAlmostFull())
    setNameFn(af, "almostFull");
}

LogicalResult FIFOOp::verify() {
  auto aet = getAlmostEmptyThreshold();
  auto aft = getAlmostFullThreshold();
  size_t depth = getDepth();
  if (aft.has_value() && aft.value() > depth)
    return emitOpError("almost full threshold must be <= FIFO depth");

  if (aet.has_value() && aet.value() > depth)
    return emitOpError("almost empty threshold must be <= FIFO depth");

  return success();
}

//===----------------------------------------------------------------------===//
// CompRegOp
//===----------------------------------------------------------------------===//

/// Suggest a name for each result value based on the saved result names
/// attribute.
void CompRegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (auto name = getName())
    setNameFn(getResult(), *name);
}

template <typename TOp>
LogicalResult verifyResets(TOp op) {
  if ((op.getReset() == nullptr) ^ (op.getResetValue() == nullptr))
    return op->emitOpError(
        "either reset and resetValue or neither must be specified");
  bool hasReset = op.getReset() != nullptr;
  if (hasReset && op.getResetValue().getType() != op.getInput().getType())
    return op->emitOpError("reset value must be the same type as the input");

  return success();
}

std::optional<size_t> CompRegOp::getTargetResultIndex() { return 0; }

LogicalResult CompRegOp::verify() { return verifyResets(*this); }

//===----------------------------------------------------------------------===//
// CompRegClockEnabledOp
//===----------------------------------------------------------------------===//

/// Suggest a name for each result value based on the saved result names
/// attribute.
void CompRegClockEnabledOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (auto name = getName())
    setNameFn(getResult(), *name);
}

std::optional<size_t> CompRegClockEnabledOp::getTargetResultIndex() {
  return 0;
}

LogicalResult CompRegClockEnabledOp::verify() { return verifyResets(*this); }

LogicalResult CompRegClockEnabledOp::canonicalize(CompRegClockEnabledOp op,
                                                  PatternRewriter &rewriter) {
  // reg(comb.mux(en, d, ?), en) -> reg(d, en)
  // reg(arith.select(en, d, ?), en) -> reg(d, en)
  auto *inputOp = op.getInput().getDefiningOp();
  if (isa_and_nonnull<comb::MuxOp, arith::SelectOp>(inputOp) &&
      inputOp->getOperand(0) == op.getClockEnable()) {
    rewriter.modifyOpInPlace(
        op, [&] { op.getInputMutable().assign(inputOp->getOperand(1)); });
    return success();
  }

  // Match constant clock enable values.
  APInt en;
  if (mlir::matchPattern(op.getClockEnable(), mlir::m_ConstantInt(&en))) {
    if (en.isAllOnes()) {
      rewriter.replaceOpWithNewOp<seq::CompRegOp>(
          op, op.getInput(), op.getClk(), op.getNameAttr(), op.getReset(),
          op.getResetValue(), op.getInitialValue(), op.getInnerSymAttr());
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ShiftRegOp
//===----------------------------------------------------------------------===//

void ShiftRegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the wire has an optional 'name' attribute, use it.
  if (auto name = getName())
    setNameFn(getResult(), *name);
}

std::optional<size_t> ShiftRegOp::getTargetResultIndex() { return 0; }

LogicalResult ShiftRegOp::verify() {
  if (failed(verifyResets(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// FirRegOp
//===----------------------------------------------------------------------===//

void FirRegOp::build(OpBuilder &builder, OperationState &result, Value input,
                     Value clk, StringAttr name, hw::InnerSymAttr innerSym,
                     Attribute preset) {

  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(input);
  result.addOperands(clk);

  result.addAttribute(getNameAttrName(result.name), name);

  if (innerSym)
    result.addAttribute(getInnerSymAttrName(result.name), innerSym);

  if (preset)
    result.addAttribute(getPresetAttrName(result.name), preset);

  result.addTypes(input.getType());
}

void FirRegOp::build(OpBuilder &builder, OperationState &result, Value input,
                     Value clk, StringAttr name, Value reset, Value resetValue,
                     hw::InnerSymAttr innerSym, bool isAsync) {

  OpBuilder::InsertionGuard guard(builder);

  result.addOperands(input);
  result.addOperands(clk);
  result.addOperands(reset);
  result.addOperands(resetValue);

  result.addAttribute(getNameAttrName(result.name), name);
  if (isAsync)
    result.addAttribute(getIsAsyncAttrName(result.name), builder.getUnitAttr());

  if (innerSym)
    result.addAttribute(getInnerSymAttrName(result.name), innerSym);

  result.addTypes(input.getType());
}

ParseResult FirRegOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  llvm::SMLoc loc = parser.getCurrentLocation();

  using Op = OpAsmParser::UnresolvedOperand;

  Op next, clk;
  if (parser.parseOperand(next) || parser.parseKeyword("clock") ||
      parser.parseOperand(clk))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("sym"))) {
    hw::InnerSymAttr innerSym;
    if (parser.parseCustomAttributeWithFallback(innerSym, /*type=*/nullptr,
                                                "inner_sym", result.attributes))
      return failure();
  }

  // Parse reset [sync|async] %reset, %value
  std::optional<std::pair<Op, Op>> resetAndValue;
  if (succeeded(parser.parseOptionalKeyword("reset"))) {
    bool isAsync;
    if (succeeded(parser.parseOptionalKeyword("async")))
      isAsync = true;
    else if (succeeded(parser.parseOptionalKeyword("sync")))
      isAsync = false;
    else
      return parser.emitError(loc, "invalid reset, expected 'sync' or 'async'");
    if (isAsync)
      result.attributes.append("isAsync", builder.getUnitAttr());

    resetAndValue = {{}, {}};
    if (parser.parseOperand(resetAndValue->first) || parser.parseComma() ||
        parser.parseOperand(resetAndValue->second))
      return failure();
  }

  std::optional<APInt> presetValue;
  llvm::SMLoc presetValueLoc;
  if (succeeded(parser.parseOptionalKeyword("preset"))) {
    presetValueLoc = parser.getCurrentLocation();
    OptionalParseResult presetIntResult =
        parser.parseOptionalInteger(presetValue.emplace());
    if (!presetIntResult.has_value() || failed(*presetIntResult))
      return parser.emitError(loc, "expected integer value");
  }

  Type ty;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(ty))
    return failure();
  result.addTypes({ty});

  if (presetValue) {
    uint64_t width = 0;
    if (hw::type_isa<seq::ClockType>(ty)) {
      width = 1;
    } else {
      auto maybeWidth = hw::getBitWidth(ty);
      if (!maybeWidth)
        return parser.emitError(presetValueLoc,
                                "cannot preset register of unknown width");
      width = *maybeWidth;
    }

    APInt presetResult = presetValue->sextOrTrunc(width);
    if (presetResult.zextOrTrunc(presetValue->getBitWidth()) != *presetValue)
      return parser.emitError(loc, "preset value too large");

    auto builder = parser.getBuilder();
    auto presetTy = builder.getIntegerType(width);
    auto resultAttr = builder.getIntegerAttr(presetTy, presetResult);
    result.addAttribute("preset", resultAttr);
  }

  setNameFromResult(parser, result);

  if (parser.resolveOperand(next, ty, result.operands))
    return failure();

  Type clkTy = ClockType::get(result.getContext());
  if (parser.resolveOperand(clk, clkTy, result.operands))
    return failure();

  if (resetAndValue) {
    Type i1 = IntegerType::get(result.getContext(), 1);
    if (parser.resolveOperand(resetAndValue->first, i1, result.operands) ||
        parser.resolveOperand(resetAndValue->second, ty, result.operands))
      return failure();
  }

  return success();
}

void FirRegOp::print(::mlir::OpAsmPrinter &p) {
  SmallVector<StringRef> elidedAttrs = {
      getInnerSymAttrName(), getIsAsyncAttrName(), getPresetAttrName()};

  p << ' ' << getNext() << " clock " << getClk();

  if (auto sym = getInnerSymAttr()) {
    p << " sym ";
    sym.print(p);
  }

  if (hasReset()) {
    p << " reset " << (getIsAsync() ? "async" : "sync") << ' ';
    p << getReset() << ", " << getResetValue();
  }

  if (auto preset = getPresetAttr()) {
    p << " preset " << preset.getValue();
  }

  if (canElideName(p, *this))
    elidedAttrs.push_back("name");

  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  p << " : " << getNext().getType();
}

/// Verifier for the FIR register op.
LogicalResult FirRegOp::verify() {
  if (getReset() || getResetValue() || getIsAsync()) {
    if (!getReset() || !getResetValue())
      return emitOpError("must specify reset and reset value");
  } else {
    if (getIsAsync())
      return emitOpError("register with no reset cannot be async");
  }
  if (auto preset = getPresetAttr()) {
    auto presetWidth = hw::getBitWidth(preset.getType());
    auto width = hw::getBitWidth(getType());
    if (preset.getType() != getType() &&
        (!presetWidth || !width || *presetWidth != *width))
      return emitOpError("preset type width must match register type");
  }
  return success();
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void FirRegOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // If the register has an optional 'name' attribute, use it.
  if (!getName().empty())
    setNameFn(getResult(), getName());
}

std::optional<size_t> FirRegOp::getTargetResultIndex() { return 0; }

LogicalResult FirRegOp::canonicalize(FirRegOp op, PatternRewriter &rewriter) {

  // If the register has a constant zero reset, drop the reset and reset value
  // altogether (And preserve the PresetAttr).
  if (auto reset = op.getReset()) {
    if (auto constOp = reset.getDefiningOp<hw::ConstantOp>()) {
      if (constOp.getValue().isZero()) {
        rewriter.replaceOpWithNewOp<FirRegOp>(
            op, op.getNext(), op.getClk(), op.getNameAttr(),
            op.getInnerSymAttr(), op.getPresetAttr());
        return success();
      }
    }
  }

  // If the register has a symbol, we can't optimize it away.
  if (op.getInnerSymAttr())
    return failure();

  // Replace a register with a trivial feedback or constant clock with a
  // constant zero.
  // TODO: Once HW aggregate constant values are supported, move this
  // canonicalization to the folder.
  auto isConstant = [&]() -> bool {
    if (op.getNext() == op.getResult())
      return true;
    if (auto clk = op.getClk().getDefiningOp<seq::ToClockOp>())
      return clk.getInput().getDefiningOp<hw::ConstantOp>();
    return false;
  };

  // Preset can block canonicalization only if it is non-zero.
  bool replaceWithConstZero = true;
  if (auto preset = op.getPresetAttr())
    if (!preset.getValue().isZero())
      replaceWithConstZero = false;

  if (isConstant() && !op.getResetValue() && replaceWithConstZero) {
    if (isa<seq::ClockType>(op.getType())) {
      rewriter.replaceOpWithNewOp<seq::ConstClockOp>(
          op, seq::ClockConstAttr::get(rewriter.getContext(), ClockConst::Low));
    } else {
      auto width = hw::getBitWidth(op.getType());
      assert(width && "register type must have known bitwidth");
      auto constant =
          hw::ConstantOp::create(rewriter, op.getLoc(), APInt::getZero(*width));
      rewriter.replaceOpWithNewOp<hw::BitcastOp>(op, op.getType(), constant);
    }
    return success();
  }

  // Canonicalize registers with mux-based constant drivers.
  // This pattern matches registers where the next value is a mux with one
  // branch being the register itself (creating a self-loop) and the other
  // branch being a constant. In such cases, the register effectively holds a
  // constant value and can be replaced with that constant.
  if (auto nextMux = op.getNext().getDefiningOp<comb::MuxOp>()) {
    // Reject optimization if register has preset attribute (for simplicity)
    if (op.getPresetAttr())
      return failure();

    Attribute value;
    Value replacedValue;

    // Check if true branch is self-loop and false branch is constant
    if (nextMux.getTrueValue() == op.getResult() &&
        matchPattern(nextMux.getFalseValue(), m_Constant(&value))) {
      replacedValue = nextMux.getFalseValue();
    }
    // Check if false branch is self-loop and true branch is constant
    else if (nextMux.getFalseValue() == op.getResult() &&
             matchPattern(nextMux.getTrueValue(), m_Constant(&value))) {
      replacedValue = nextMux.getTrueValue();
    }

    if (!replacedValue)
      return failure();

    // Verify reset value compatibility: if register has reset, it must be
    // a constant that matches the mux constant
    if (op.getResetValue()) {
      Attribute resetConst;
      if (matchPattern(op.getResetValue(), m_Constant(&resetConst))) {
        if (resetConst != value)
          return failure();
      } else {
        // Non-constant reset value prevents optimization
        return failure();
      }
    }

    assert(replacedValue);
    // Apply the optimization if all conditions are met
    rewriter.replaceOp(op, replacedValue);
    return success();
  }

  // For reset-less 1d array registers, replace an uninitialized element with
  // constant zero. For example, let `r` be a 2xi1 register and its next value
  // be `{foo, r[0]}`. `r[0]` is connected to itself so will never be
  // initialized. If we don't enable aggregate preservation, `r_0` is replaced
  // with `0`. Hence this canonicalization replaces 0th element of the next
  // value with zero to match the behaviour.
  if (!op.getReset() && !op.getPresetAttr()) {
    if (auto arrayCreate = op.getNext().getDefiningOp<hw::ArrayCreateOp>()) {
      // For now only support 1d arrays.
      // TODO: Support nested arrays and bundles.
      if (isa<IntegerType>(
              hw::type_cast<hw::ArrayType>(op.getResult().getType())
                  .getElementType())) {
        SmallVector<Value> nextOperands;
        bool changed = false;
        for (const auto &[i, value] :
             llvm::enumerate(arrayCreate.getOperands())) {
          auto index = arrayCreate.getOperands().size() - i - 1;
          APInt elementIndex;
          // Check that the corresponding operand is op's element.
          if (auto arrayGet = value.getDefiningOp<hw::ArrayGetOp>())
            if (arrayGet.getInput() == op.getResult() &&
                matchPattern(arrayGet.getIndex(),
                             m_ConstantInt(&elementIndex)) &&
                elementIndex == index) {
              auto width = hw::getBitWidth(arrayGet.getType());
              assert(width && "array element type must have known bitwidth");
              nextOperands.push_back(hw::ConstantOp::create(
                  rewriter, op.getLoc(), APInt::getZero(*width)));
              changed = true;
              continue;
            }
          nextOperands.push_back(value);
        }
        // If one of the operands is self loop, update the next value.
        if (changed) {
          auto newNextVal = hw::ArrayCreateOp::create(
              rewriter, arrayCreate.getLoc(), nextOperands);
          if (arrayCreate->hasOneUse())
            // If the original next value has a single use, we can replace the
            // value directly.
            rewriter.replaceOp(arrayCreate, newNextVal);
          else {
            // Otherwise, replace the entire firreg with a new one.
            rewriter.replaceOpWithNewOp<FirRegOp>(op, newNextVal, op.getClk(),
                                                  op.getNameAttr(),
                                                  op.getInnerSymAttr());
          }

          return success();
        }
      }
    }
  }

  return failure();
}

OpFoldResult FirRegOp::fold(FoldAdaptor adaptor) {
  // If the register has a symbol or preset value, we can't optimize it away.
  // TODO: Handle a preset value.
  if (getInnerSymAttr())
    return {};

  auto presetAttr = getPresetAttr();

  // If the register is held in permanent reset, replace it with its reset
  // value. This works trivially if the reset is asynchronous and therefore
  // level-sensitive, in which case it will always immediately assume the reset
  // value in silicon. If it is synchronous, the register value is undefined
  // until the first clock edge at which point it becomes the reset value, in
  // which case we simply define the initial value to already be the reset
  // value. Works only if no preset.
  if (!presetAttr)
    if (auto reset = getReset())
      if (auto constOp = reset.getDefiningOp<hw::ConstantOp>())
        if (constOp.getValue().isOne())
          return getResetValue();

  // If the register's next value is trivially it's current value, or the
  // register is never clocked, we can replace the register with a constant
  // value.
  bool isTrivialFeedback = (getNext() == getResult());
  bool isNeverClocked =
      adaptor.getClk() != nullptr; // clock operand is constant
  if (!isTrivialFeedback && !isNeverClocked)
    return {};

  // If the register has a const reset value, and no preset, we can replace it
  // with the const reset. We cannot replace it with a non-constant reset value.
  if (auto resetValue = getResetValue()) {
    if (auto *op = resetValue.getDefiningOp()) {
      if (op->hasTrait<OpTrait::ConstantLike>() && !presetAttr)
        return resetValue;
      if (auto constOp = dyn_cast<hw::ConstantOp>(op))
        if (presetAttr.getValue() == constOp.getValue())
          return resetValue;
    }
    return {};
  }

  // Otherwise we want to replace the register with a constant 0. For now this
  // only works with integer types.
  auto intType = dyn_cast<IntegerType>(getType());
  if (!intType)
    return {};
  // If preset present, then replace with preset.
  if (presetAttr)
    return presetAttr;
  return IntegerAttr::get(intType, 0);
}

//===----------------------------------------------------------------------===//
// ClockGateOp
//===----------------------------------------------------------------------===//

OpFoldResult ClockGateOp::fold(FoldAdaptor adaptor) {
  // Forward the clock if one of the enables is always true.
  if (isConstantOne(adaptor.getEnable()) ||
      isConstantOne(adaptor.getTestEnable()))
    return getInput();

  // Fold to a constant zero clock if the enables are always false.
  if (isConstantZero(adaptor.getEnable()) &&
      (!getTestEnable() || isConstantZero(adaptor.getTestEnable())))
    return ClockConstAttr::get(getContext(), ClockConst::Low);

  // Forward constant zero clocks.
  if (auto clockAttr = dyn_cast_or_null<ClockConstAttr>(adaptor.getInput()))
    if (clockAttr.getValue() == ClockConst::Low)
      return ClockConstAttr::get(getContext(), ClockConst::Low);

  // Transitive clock gating - eliminate clock gates that are driven by an
  // identical enable signal somewhere higher in the clock gate hierarchy.
  auto clockGateInputOp = getInput().getDefiningOp<ClockGateOp>();
  while (clockGateInputOp) {
    if (clockGateInputOp.getEnable() == getEnable() &&
        clockGateInputOp.getTestEnable() == getTestEnable())
      return getInput();
    clockGateInputOp = clockGateInputOp.getInput().getDefiningOp<ClockGateOp>();
  }

  return {};
}

LogicalResult ClockGateOp::canonicalize(ClockGateOp op,
                                        PatternRewriter &rewriter) {
  // Remove constant false test enable.
  if (auto testEnable = op.getTestEnable()) {
    if (auto constOp = testEnable.getDefiningOp<hw::ConstantOp>()) {
      if (constOp.getValue().isZero()) {
        rewriter.modifyOpInPlace(op,
                                 [&] { op.getTestEnableMutable().clear(); });
        return success();
      }
    }
  }

  return failure();
}

std::optional<size_t> ClockGateOp::getTargetResultIndex() {
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// ClockMuxOp
//===----------------------------------------------------------------------===//

OpFoldResult ClockMuxOp::fold(FoldAdaptor adaptor) {
  if (isConstantOne(adaptor.getCond()))
    return getTrueClock();
  if (isConstantZero(adaptor.getCond()))
    return getFalseClock();
  return {};
}

//===----------------------------------------------------------------------===//
// FirMemOp
//===----------------------------------------------------------------------===//

LogicalResult FirMemOp::canonicalize(FirMemOp op, PatternRewriter &rewriter) {
  // Do not change memories if symbols point to them.
  if (op.getInnerSymAttr())
    return failure();

  bool readOnly = true, writeOnly = true;

  // If the memory has no read ports, erase it.
  for (auto *user : op->getUsers()) {
    if (isa<FirMemReadOp, FirMemReadWriteOp>(user)) {
      writeOnly = false;
    }
    if (isa<FirMemWriteOp, FirMemReadWriteOp>(user)) {
      readOnly = false;
    }
    assert((isa<FirMemReadOp, FirMemWriteOp, FirMemReadWriteOp>(user)) &&
           "invalid seq.firmem user");
  }
  if (writeOnly) {
    for (auto *user : llvm::make_early_inc_range(op->getUsers()))
      rewriter.eraseOp(user);

    rewriter.eraseOp(op);
    return success();
  }

  if (readOnly && !op.getInit()) {
    // Replace all read ports with a constant 0.
    for (auto *user : llvm::make_early_inc_range(op->getUsers())) {
      auto readOp = cast<FirMemReadOp>(user);
      auto width = hw::getBitWidth(readOp.getType());
      assert(width && "read type must have known bitwidth");
      Value zero = hw::ConstantOp::create(rewriter, readOp.getLoc(),
                                          APInt::getZero(*width));
      if (readOp.getType() != zero.getType())
        zero = hw::BitcastOp::create(rewriter, readOp.getLoc(),
                                     readOp.getType(), zero);
      rewriter.replaceOp(readOp, zero);
    }
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

void FirMemOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  auto nameAttr = (*this)->getAttrOfType<StringAttr>("name");
  if (!nameAttr.getValue().empty())
    setNameFn(getResult(), nameAttr.getValue());
}

std::optional<size_t> FirMemOp::getTargetResultIndex() { return 0; }

template <class Op>
static LogicalResult verifyFirMemMask(Op op) {
  if (auto mask = op.getMask()) {
    auto memType = op.getMemory().getType();
    if (!memType.getMaskWidth())
      return op.emitOpError("has mask operand but memory type '")
             << memType << "' has no mask";
    auto expected = IntegerType::get(op.getContext(), *memType.getMaskWidth());
    if (mask.getType() != expected)
      return op.emitOpError("has mask operand of type '")
             << mask.getType() << "', but memory type requires '" << expected
             << "'";
  }
  return success();
}

LogicalResult FirMemWriteOp::verify() { return verifyFirMemMask(*this); }
LogicalResult FirMemReadWriteOp::verify() { return verifyFirMemMask(*this); }

static bool isConstClock(Value value) {
  if (!value)
    return false;
  return value.getDefiningOp<seq::ConstClockOp>();
}

static bool isConstZero(Value value) {
  if (value)
    if (auto constOp = value.getDefiningOp<hw::ConstantOp>())
      return constOp.getValue().isZero();
  return false;
}

static bool isConstAllOnes(Value value) {
  if (value)
    if (auto constOp = value.getDefiningOp<hw::ConstantOp>())
      return constOp.getValue().isAllOnes();
  return false;
}

LogicalResult FirMemReadOp::canonicalize(FirMemReadOp op,
                                         PatternRewriter &rewriter) {
  // Remove the enable if it is constant true.
  if (isConstAllOnes(op.getEnable())) {
    rewriter.modifyOpInPlace(op, [&] { op.getEnableMutable().erase(0); });
    return success();
  }
  return failure();
}

LogicalResult FirMemWriteOp::canonicalize(FirMemWriteOp op,
                                          PatternRewriter &rewriter) {
  // Remove the write port if it is trivially dead.
  if (isConstZero(op.getEnable()) || isConstZero(op.getMask()) ||
      isConstClock(op.getClk())) {
    auto memOp = op.getMemory().getDefiningOp<FirMemOp>();
    if (memOp.getInnerSymAttr())
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
  bool anyChanges = false;

  // Remove the enable if it is constant true.
  if (auto enable = op.getEnable(); isConstAllOnes(enable)) {
    rewriter.modifyOpInPlace(op, [&] { op.getEnableMutable().erase(0); });
    anyChanges = true;
  }

  // Remove the mask if it is all ones.
  if (auto mask = op.getMask(); isConstAllOnes(mask)) {
    rewriter.modifyOpInPlace(op, [&] { op.getMaskMutable().erase(0); });
    anyChanges = true;
  }

  return success(anyChanges);
}

LogicalResult FirMemReadWriteOp::canonicalize(FirMemReadWriteOp op,
                                              PatternRewriter &rewriter) {
  // Replace the read-write port with a read port if the write behavior is
  // trivially disabled.
  if (isConstZero(op.getEnable()) || isConstZero(op.getMask()) ||
      isConstClock(op.getClk()) || isConstZero(op.getMode())) {
    auto opAttrs = op->getAttrs();
    auto opAttrNames = op.getAttributeNames();
    auto newOp = rewriter.replaceOpWithNewOp<FirMemReadOp>(
        op, op.getMemory(), op.getAddress(), op.getClk(), op.getEnable());
    for (auto namedAttr : opAttrs)
      if (!llvm::is_contained(opAttrNames, namedAttr.getName()))
        newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    return success();
  }
  bool anyChanges = false;

  // Remove the enable if it is constant true.
  if (auto enable = op.getEnable(); isConstAllOnes(enable)) {
    rewriter.modifyOpInPlace(op, [&] { op.getEnableMutable().erase(0); });
    anyChanges = true;
  }

  // Remove the mask if it is all ones.
  if (auto mask = op.getMask(); isConstAllOnes(mask)) {
    rewriter.modifyOpInPlace(op, [&] { op.getMaskMutable().erase(0); });
    anyChanges = true;
  }

  return success(anyChanges);
}

//===----------------------------------------------------------------------===//
// ConstClockOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstClockOp::fold(FoldAdaptor adaptor) {
  return ClockConstAttr::get(getContext(), getValue());
}

//===----------------------------------------------------------------------===//
// ToClockOp/FromClockOp
//===----------------------------------------------------------------------===//

LogicalResult ToClockOp::canonicalize(ToClockOp op, PatternRewriter &rewriter) {
  if (auto fromClock = op.getInput().getDefiningOp<FromClockOp>()) {
    rewriter.replaceOp(op, fromClock.getInput());
    return success();
  }
  return failure();
}

OpFoldResult ToClockOp::fold(FoldAdaptor adaptor) {
  if (auto fromClock = getInput().getDefiningOp<FromClockOp>())
    return fromClock.getInput();
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getInput())) {
    auto value =
        intAttr.getValue().isZero() ? ClockConst::Low : ClockConst::High;
    return ClockConstAttr::get(getContext(), value);
  }
  return {};
}

LogicalResult FromClockOp::canonicalize(FromClockOp op,
                                        PatternRewriter &rewriter) {
  if (auto toClock = op.getInput().getDefiningOp<ToClockOp>()) {
    rewriter.replaceOp(op, toClock.getInput());
    return success();
  }
  return failure();
}

OpFoldResult FromClockOp::fold(FoldAdaptor adaptor) {
  if (auto toClock = getInput().getDefiningOp<ToClockOp>())
    return toClock.getInput();
  if (auto clockAttr = dyn_cast_or_null<ClockConstAttr>(adaptor.getInput())) {
    auto ty = IntegerType::get(getContext(), 1);
    return IntegerAttr::get(ty, clockAttr.getValue() == ClockConst::High);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ClockInverterOp
//===----------------------------------------------------------------------===//

OpFoldResult ClockInverterOp::fold(FoldAdaptor adaptor) {
  if (auto chainedInv = getInput().getDefiningOp<ClockInverterOp>())
    return chainedInv.getInput();
  if (auto clockAttr = dyn_cast_or_null<ClockConstAttr>(adaptor.getInput())) {
    auto clockIn = clockAttr.getValue() == ClockConst::High;
    return ClockConstAttr::get(getContext(),
                               clockIn ? ClockConst::Low : ClockConst::High);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// FIR memory helper
//===----------------------------------------------------------------------===//

FirMemory::FirMemory(hw::HWModuleGeneratedOp op) {
  depth = op->getAttrOfType<IntegerAttr>("depth").getInt();
  numReadPorts = op->getAttrOfType<IntegerAttr>("numReadPorts").getUInt();
  numWritePorts = op->getAttrOfType<IntegerAttr>("numWritePorts").getUInt();
  numReadWritePorts =
      op->getAttrOfType<IntegerAttr>("numReadWritePorts").getUInt();
  readLatency = op->getAttrOfType<IntegerAttr>("readLatency").getUInt();
  writeLatency = op->getAttrOfType<IntegerAttr>("writeLatency").getUInt();
  dataWidth = op->getAttrOfType<IntegerAttr>("width").getUInt();
  if (op->hasAttrOfType<IntegerAttr>("maskGran"))
    maskGran = op->getAttrOfType<IntegerAttr>("maskGran").getUInt();
  else
    maskGran = dataWidth;
  readUnderWrite = op->getAttrOfType<seq::RUWAttr>("readUnderWrite").getValue();
  writeUnderWrite =
      op->getAttrOfType<seq::WUWAttr>("writeUnderWrite").getValue();
  if (auto clockIDsAttr = op->getAttrOfType<ArrayAttr>("writeClockIDs"))
    for (auto clockID : clockIDsAttr)
      writeClockIDs.push_back(
          cast<IntegerAttr>(clockID).getValue().getZExtValue());
  initFilename = op->getAttrOfType<StringAttr>("initFilename").getValue();
  initIsBinary = op->getAttrOfType<BoolAttr>("initIsBinary").getValue();
  initIsInline = op->getAttrOfType<BoolAttr>("initIsInline").getValue();
}

LogicalResult InitialOp::verify() {
  // Check outputs.
  auto *terminator = this->getBody().front().getTerminator();
  if (terminator->getOperands().size() != getNumResults())
    return emitError() << "result type doesn't match with the terminator";
  for (auto [lhs, rhs] :
       llvm::zip(terminator->getOperands().getTypes(), getResultTypes())) {
    if (cast<seq::ImmutableType>(rhs).getInnerType() != lhs)
      return emitError() << cast<seq::ImmutableType>(rhs).getInnerType()
                         << " is expected but got " << lhs;
  }

  auto blockArgs = this->getBody().front().getArguments();

  if (blockArgs.size() != getNumOperands())
    return emitError() << "operand type doesn't match with the block arg";

  for (auto [blockArg, operand] : llvm::zip(blockArgs, getOperands())) {
    if (blockArg.getType() !=
        cast<ImmutableType>(operand.getType()).getInnerType())
      return emitError()
             << blockArg.getType() << " is expected but got "
             << cast<ImmutableType>(operand.getType()).getInnerType();
  }
  return success();
}
void InitialOp::build(OpBuilder &builder, OperationState &result,
                      TypeRange resultTypes, std::function<void()> ctor) {
  OpBuilder::InsertionGuard guard(builder);

  builder.createBlock(result.addRegion());
  SmallVector<Type> types;
  for (auto t : resultTypes)
    types.push_back(seq::ImmutableType::get(t));

  result.addTypes(types);

  if (ctor)
    ctor();
}

TypedValue<seq::ImmutableType>
circt::seq::createConstantInitialValue(OpBuilder builder, Location loc,
                                       mlir::IntegerAttr attr) {
  auto initial = seq::InitialOp::create(builder, loc, attr.getType(), [&]() {
    auto constant = hw::ConstantOp::create(builder, loc, attr);
    seq::YieldOp::create(builder, loc, ArrayRef<Value>{constant});
  });
  return cast<TypedValue<seq::ImmutableType>>(initial->getResult(0));
}

mlir::TypedValue<seq::ImmutableType>
circt::seq::createConstantInitialValue(OpBuilder builder, Operation *op) {
  assert(op->getNumResults() == 1 &&
         op->hasTrait<mlir::OpTrait::ConstantLike>());
  auto initial = seq::InitialOp::create(
      builder, op->getLoc(), op->getResultTypes(), [&]() {
        auto clonedOp = builder.clone(*op);
        seq::YieldOp::create(builder, op->getLoc(), clonedOp->getResults());
      });
  return cast<mlir::TypedValue<seq::ImmutableType>>(initial.getResult(0));
}

Value circt::seq::unwrapImmutableValue(TypedValue<seq::ImmutableType> value) {
  auto resultNum = cast<OpResult>(value).getResultNumber();
  auto initialOp = value.getDefiningOp<seq::InitialOp>();
  assert(initialOp);
  return initialOp.getBodyBlock()->getTerminator()->getOperand(resultNum);
}

FailureOr<seq::InitialOp> circt::seq::mergeInitialOps(Block *block) {
  SmallVector<Operation *> initialOps;
  for (auto &op : *block)
    if (isa<seq::InitialOp>(op))
      initialOps.push_back(&op);

  if (!mlir::computeTopologicalSorting(initialOps, {}))
    return block->getParentOp()->emitError()
           << "initial ops cannot be topologically sorted";

  // No need to merge if there is only one initial op.
  if (initialOps.size() <= 1)
    return initialOps.empty() ? seq::InitialOp()
                              : cast<seq::InitialOp>(initialOps[0]);

  auto initialOp = cast<seq::InitialOp>(initialOps.front());
  auto yieldOp = cast<seq::YieldOp>(initialOp.getBodyBlock()->getTerminator());

  llvm::MapVector<Value, Value>
      resultToYieldOperand; // seq.immutable value to operand.

  for (auto [result, operand] :
       llvm::zip(initialOp.getResults(), yieldOp->getOperands()))
    resultToYieldOperand.insert({result, operand});

  for (size_t i = 1; i < initialOps.size(); ++i) {
    auto currentInitialOp = cast<seq::InitialOp>(initialOps[i]);
    auto operands = currentInitialOp->getOperands();
    for (auto [blockArg, operand] :
         llvm::zip(currentInitialOp.getBodyBlock()->getArguments(), operands)) {
      if (auto initOp = operand.getDefiningOp<seq::InitialOp>()) {
        assert(resultToYieldOperand.count(operand) &&
               "it must be visited already");
        blockArg.replaceAllUsesWith(resultToYieldOperand.lookup(operand));
      } else {
        // Otherwise add the operand to the current block.
        initialOp.getBodyBlock()->addArgument(
            cast<seq::ImmutableType>(operand.getType()).getInnerType(),
            operand.getLoc());
        initialOp.getInputsMutable().append(operand);
      }
    }

    auto currentYieldOp =
        cast<seq::YieldOp>(currentInitialOp.getBodyBlock()->getTerminator());

    for (auto [result, operand] : llvm::zip(currentInitialOp.getResults(),
                                            currentYieldOp->getOperands()))
      resultToYieldOperand.insert({result, operand});

    // Append the operands of the current yield op to the original yield op.
    yieldOp.getOperandsMutable().append(currentYieldOp.getOperands());
    currentYieldOp->erase();

    // Append the operations of the current initial op to the original initial
    // op.
    initialOp.getBodyBlock()->getOperations().splice(
        initialOp.end(), currentInitialOp.getBodyBlock()->getOperations());
  }

  // Move the terminator to the end of the block.
  yieldOp->moveBefore(initialOp.getBodyBlock(),
                      initialOp.getBodyBlock()->end());

  auto builder = OpBuilder::atBlockBegin(block);
  SmallVector<Type> types;
  for (auto [result, operand] : resultToYieldOperand)
    types.push_back(operand.getType());

  // Create a new initial op which accumulates the results of the merged initial
  // ops.
  auto newInitial = seq::InitialOp::create(builder, initialOp.getLoc(), types);
  newInitial.getInputsMutable().append(initialOp.getInputs());

  for (auto [resultAndOperand, newResult] :
       llvm::zip(resultToYieldOperand, newInitial.getResults()))
    resultAndOperand.first.replaceAllUsesWith(newResult);

  // Update the block arguments of the new initial op.
  for (auto oldBlockArg : initialOp.getBodyBlock()->getArguments()) {
    auto blockArg = newInitial.getBodyBlock()->addArgument(
        oldBlockArg.getType(), oldBlockArg.getLoc());
    oldBlockArg.replaceAllUsesWith(blockArg);
  }

  newInitial.getBodyBlock()->getOperations().splice(
      newInitial.end(), initialOp.getBodyBlock()->getOperations());

  // Clean up.
  while (!initialOps.empty())
    initialOps.pop_back_val()->erase();

  return newInitial;
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Seq/Seq.cpp.inc"
