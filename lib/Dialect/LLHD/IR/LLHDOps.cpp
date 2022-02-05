//===- LLHDOps.cpp - Implement the LLHD operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the LLHD ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

using namespace circt;
using namespace mlir;

template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = function_ref<ElementValueT(ElementValueT)>>
static Attribute constFoldUnaryOp(ArrayRef<Attribute> operands,
                                  const CalculationT &calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (!operands[0])
    return {};

  if (auto val = operands[0].dyn_cast<AttrElementT>()) {
    return AttrElementT::get(val.getType(), calculate(val.getValue()));
  } else if (auto val = operands[0].dyn_cast<SplatElementsAttr>()) {
    // Operand is a splat so we can avoid expanding the value out and
    // just fold based on the splat value.
    auto elementResult = calculate(val.getSplatValue<ElementValueT>());
    return DenseElementsAttr::get(val.getType(), elementResult);
  }
  if (auto val = operands[0].dyn_cast<ElementsAttr>()) {
    // Operand is ElementsAttr-derived; perform an element-wise fold by
    // expanding the values.
    auto valIt = val.getValues<ElementValueT>().begin();
    SmallVector<ElementValueT, 4> elementResults;
    elementResults.reserve(val.getNumElements());
    for (size_t i = 0, e = val.getNumElements(); i < e; ++i, ++valIt)
      elementResults.push_back(calculate(*valIt));
    return DenseElementsAttr::get(val.getType(), elementResults);
  }
  return {};
}

template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT = function_ref<
              ElementValueT(ElementValueT, ElementValueT, ElementValueT)>>
static Attribute constFoldTernaryOp(ArrayRef<Attribute> operands,
                                    const CalculationT &calculate) {
  assert(operands.size() == 3 && "ternary op takes three operands");
  if (!operands[0] || !operands[1] || !operands[2])
    return {};
  if (operands[0].getType() != operands[1].getType())
    return {};
  if (operands[0].getType() != operands[2].getType())
    return {};

  if (operands[0].isa<AttrElementT>() && operands[1].isa<AttrElementT>() &&
      operands[2].isa<AttrElementT>()) {
    auto fst = operands[0].cast<AttrElementT>();
    auto snd = operands[1].cast<AttrElementT>();
    auto trd = operands[2].cast<AttrElementT>();

    return AttrElementT::get(
        fst.getType(),
        calculate(fst.getValue(), snd.getValue(), trd.getValue()));
  }
  if (operands[0].isa<SplatElementsAttr>() &&
      operands[1].isa<SplatElementsAttr>() &&
      operands[2].isa<SplatElementsAttr>()) {
    // Operands are splats so we can avoid expanding the values out and
    // just fold based on the splat value.
    auto fst = operands[0].cast<SplatElementsAttr>();
    auto snd = operands[1].cast<SplatElementsAttr>();
    auto trd = operands[2].cast<SplatElementsAttr>();

    auto elementResult = calculate(fst.getSplatValue<ElementValueT>(),
                                   snd.getSplatValue<ElementValueT>(),
                                   trd.getSplatValue<ElementValueT>());
    return DenseElementsAttr::get(fst.getType(), elementResult);
  }
  if (operands[0].isa<ElementsAttr>() && operands[1].isa<ElementsAttr>() &&
      operands[2].isa<ElementsAttr>()) {
    // Operands are ElementsAttr-derived; perform an element-wise fold by
    // expanding the values.
    auto fst = operands[0].cast<ElementsAttr>();
    auto snd = operands[1].cast<ElementsAttr>();
    auto trd = operands[2].cast<ElementsAttr>();

    auto fstIt = fst.getValues<ElementValueT>().begin();
    auto sndIt = snd.getValues<ElementValueT>().begin();
    auto trdIt = trd.getValues<ElementValueT>().begin();
    SmallVector<ElementValueT, 4> elementResults;
    elementResults.reserve(fst.getNumElements());
    for (size_t i = 0, e = fst.getNumElements(); i < e;
         ++i, ++fstIt, ++sndIt, ++trdIt)
      elementResults.push_back(calculate(*fstIt, *sndIt, *trdIt));
    return DenseElementsAttr::get(fst.getType(), elementResults);
  }
  return {};
}

namespace {

struct constant_int_all_ones_matcher {
  bool match(Operation *op) {
    APInt value;
    return mlir::detail::constant_int_op_binder(&value).match(op) &&
           value.isAllOnesValue();
  }
};

} // anonymous namespace

unsigned circt::llhd::getLLHDTypeWidth(Type type) {
  if (auto sig = type.dyn_cast<llhd::SigType>())
    type = sig.getUnderlyingType();
  if (auto array = type.dyn_cast<hw::ArrayType>())
    return array.getSize();
  if (auto tup = type.dyn_cast<hw::StructType>())
    return tup.getElements().size();
  return type.getIntOrFloatBitWidth();
}

Type circt::llhd::getLLHDElementType(Type type) {
  if (auto sig = type.dyn_cast<llhd::SigType>())
    type = sig.getUnderlyingType();
  if (auto array = type.dyn_cast<hw::ArrayType>())
    return array.getElementType();
  return type;
}

//===---------------------------------------------------------------------===//
// LLHD Trait Helper Functions
//===---------------------------------------------------------------------===//

static bool sameKindArbitraryWidth(Type lhsType, Type rhsType) {
  if (lhsType.getTypeID() != rhsType.getTypeID())
    return false;

  if (auto sig = lhsType.dyn_cast<llhd::SigType>())
    return sameKindArbitraryWidth(
        sig.getUnderlyingType(),
        rhsType.cast<llhd::SigType>().getUnderlyingType());

  if (auto array = lhsType.dyn_cast<hw::ArrayType>())
    return array.getElementType() ==
           rhsType.cast<hw::ArrayType>().getElementType();

  return (!lhsType.isa<ShapedType>() ||
          (lhsType.cast<ShapedType>().getElementType() ==
           rhsType.cast<ShapedType>().getElementType()));
}

//===---------------------------------------------------------------------===//
// LLHD Operations
//===---------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantTimeOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::ConstantTimeOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "const has no operands");
  return valueAttr();
}

void llhd::ConstantTimeOp::build(OpBuilder &builder, OperationState &result,
                                 unsigned time, const StringRef &timeUnit,
                                 unsigned delta, unsigned epsilon) {
  auto *ctx = builder.getContext();
  auto attr = TimeAttr::get(ctx, time, timeUnit, delta, epsilon);
  return build(builder, result, TimeType::get(ctx), attr);
}

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::ShlOp::fold(ArrayRef<Attribute> operands) {
  /// llhd.shl(base, hidden, 0) -> base
  if (matchPattern(amount(), m_Zero()))
    return base();

  return constFoldTernaryOp<IntegerAttr>(
      operands, [](APInt base, APInt hidden, APInt amt) {
        base <<= amt;
        base += hidden.getHiBits(amt.getZExtValue());
        return base;
      });
}

//===----------------------------------------------------------------------===//
// ShrOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::ShrOp::fold(ArrayRef<Attribute> operands) {
  /// llhd.shl(base, hidden, 0) -> base
  if (matchPattern(amount(), m_Zero()))
    return base();

  return constFoldTernaryOp<IntegerAttr>(
      operands, [](APInt base, APInt hidden, APInt amt) {
        base = base.getHiBits(base.getBitWidth() - amt.getZExtValue());
        hidden = hidden.getLoBits(amt.getZExtValue());
        hidden <<= base.getBitWidth() - amt.getZExtValue();
        return base + hidden;
      });
}

//===----------------------------------------------------------------------===//
// SigExtractOp and PtrExtractOp
//===----------------------------------------------------------------------===//

template <class Op>
static OpFoldResult foldSigPtrExtractOp(Op op, ArrayRef<Attribute> operands) {

  if (!operands[1])
    return nullptr;

  // llhd.sig.extract(input, 0) with inputWidth == resultWidth => input
  if (op.resultWidth() == op.inputWidth() &&
      operands[1].cast<IntegerAttr>().getValue().isZero())
    return op.input();

  return nullptr;
}

OpFoldResult llhd::SigExtractOp::fold(ArrayRef<Attribute> operands) {
  return foldSigPtrExtractOp(*this, operands);
}

OpFoldResult llhd::PtrExtractOp::fold(ArrayRef<Attribute> operands) {
  return foldSigPtrExtractOp(*this, operands);
}

//===----------------------------------------------------------------------===//
// SigArraySliceOp and PtrArraySliceOp
//===----------------------------------------------------------------------===//

template <class Op>
static OpFoldResult foldSigPtrArraySliceOp(Op op,
                                           ArrayRef<Attribute> operands) {
  if (!operands[1])
    return nullptr;

  // llhd.sig.array_slice(input, 0) with inputWidth == resultWidth => input
  if (op.resultWidth() == op.inputWidth() &&
      operands[1].cast<IntegerAttr>().getValue().isZero())
    return op.input();

  return nullptr;
}

OpFoldResult llhd::SigArraySliceOp::fold(ArrayRef<Attribute> operands) {
  return foldSigPtrArraySliceOp(*this, operands);
}

OpFoldResult llhd::PtrArraySliceOp::fold(ArrayRef<Attribute> operands) {
  return foldSigPtrArraySliceOp(*this, operands);
}

template <class Op>
static LogicalResult canonicalizeSigPtrArraySliceOp(Op op,
                                                    PatternRewriter &rewriter) {
  IntegerAttr amountAttr, indexAttr;

  if (!matchPattern(op.lowIndex(), m_Constant(&indexAttr)))
    return failure();

  // llhd.sig.array_slice(llhd.shr(hidden, base, constant amt), constant index)
  //   with amt + index + sliceWidth <= baseWidth
  //   => llhd.sig.array_slice(base, amt + index)
  if (matchPattern(op.input(),
                   m_Op<llhd::ShrOp>(matchers::m_Any(), matchers::m_Any(),
                                     m_Constant(&amountAttr)))) {
    uint64_t amount = amountAttr.getValue().getZExtValue();
    uint64_t index = indexAttr.getValue().getZExtValue();
    auto shrOp = op.input().template getDefiningOp<llhd::ShrOp>();
    unsigned baseWidth = shrOp.getBaseWidth();

    if (amount + index + op.resultWidth() <= baseWidth) {
      op.inputMutable().assign(shrOp.base());
      Value newIndex = rewriter.create<hw::ConstantOp>(
          op->getLoc(), amountAttr.getValue() + indexAttr.getValue());
      op.lowIndexMutable().assign(newIndex);

      return success();
    }
  }

  // llhd.sig.array_slice(llhd.sig.array_slice(target, a), b)
  //   => llhd.sig.array_slice(target, a+b)
  IntegerAttr a;
  if (matchPattern(op.input(), m_Op<Op>(matchers::m_Any(), m_Constant(&a)))) {
    auto sliceOp = op.input().template getDefiningOp<Op>();
    op.inputMutable().assign(sliceOp.input());
    Value newIndex = rewriter.create<hw::ConstantOp>(
        op->getLoc(), a.getValue() + indexAttr.getValue());
    op.lowIndexMutable().assign(newIndex);

    return success();
  }

  return failure();
}

LogicalResult llhd::SigArraySliceOp::canonicalize(llhd::SigArraySliceOp op,
                                                  PatternRewriter &rewriter) {
  return canonicalizeSigPtrArraySliceOp(op, rewriter);
}

LogicalResult llhd::PtrArraySliceOp::canonicalize(llhd::PtrArraySliceOp op,
                                                  PatternRewriter &rewriter) {
  return canonicalizeSigPtrArraySliceOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// SigArrayGetOp and PtrArrayGetOp
//===----------------------------------------------------------------------===//

template <class Op>
static LogicalResult canonicalizeSigPtrArrayGetOp(Op op,
                                                  PatternRewriter &rewriter) {
  // llhd.sig.array_get(llhd.shr(hidden, base, constant amt), constant index)
  IntegerAttr indexAttr, amountAttr;
  if (matchPattern(op.index(), m_Constant(&indexAttr)) &&
      matchPattern(op.input(),
                   m_Op<llhd::ShrOp>(matchers::m_Any(), matchers::m_Any(),
                                     m_Constant(&amountAttr)))) {
    APInt index = indexAttr.getValue();
    APInt amount = amountAttr.getValue();
    auto shrOp = op.input().template getDefiningOp<llhd::ShrOp>();
    unsigned baseWidth = shrOp.getBaseWidth();
    unsigned hiddenWidth = shrOp.getHiddenWidth();

    // with amt + index < baseWidth
    //   => llhd.sig.array_get(base, amt + index)
    if (amount.getZExtValue() + index.getZExtValue() < baseWidth) {
      op.inputMutable().assign(shrOp.base());
      Value newIndex =
          rewriter.create<hw::ConstantOp>(op->getLoc(), amount + index);
      op.indexMutable().assign(newIndex);

      return success();
    }

    // with amt + index >= baseWidth && amt + index < baseWidth + hiddenWidth
    //   => llhd.sig.array_get(hidden, amt + index - baseWidth)
    if (amount.getZExtValue() + index.getZExtValue() <
        baseWidth + hiddenWidth) {
      op.inputMutable().assign(shrOp.hidden());
      Value newIndex = rewriter.create<hw::ConstantOp>(
          op->getLoc(), amount + index - baseWidth);
      op.indexMutable().assign(newIndex);

      return success();
    }
  }

  return failure();
}

LogicalResult llhd::SigArrayGetOp::canonicalize(llhd::SigArrayGetOp op,
                                                PatternRewriter &rewriter) {
  return canonicalizeSigPtrArrayGetOp(op, rewriter);
}

LogicalResult llhd::PtrArrayGetOp::canonicalize(llhd::PtrArrayGetOp op,
                                                PatternRewriter &rewriter) {
  return canonicalizeSigPtrArrayGetOp(op, rewriter);
}

//===----------------------------------------------------------------------===//
// SigStructExtractOp and PtrStructExtractOp
//===----------------------------------------------------------------------===//

template <class SigPtrType>
static LogicalResult
inferReturnTypesOfStructExtractOp(MLIRContext *context, Optional<Location> loc,
                                  ValueRange operands, DictionaryAttr attrs,
                                  mlir::RegionRange regions,
                                  SmallVectorImpl<Type> &results) {
  Type type = operands[0]
                  .getType()
                  .cast<SigPtrType>()
                  .getUnderlyingType()
                  .template cast<hw::StructType>()
                  .getFieldType(attrs.getNamed("field")
                                    ->getValue()
                                    .cast<StringAttr>()
                                    .getValue());
  if (!type) {
    context->getDiagEngine().emit(loc.getValueOr(UnknownLoc()),
                                  DiagnosticSeverity::Error)
        << "invalid field name specified";
    return failure();
  }
  results.push_back(SigPtrType::get(type));
  return success();
}

LogicalResult llhd::SigStructExtractOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::RegionRange regions,
    SmallVectorImpl<Type> &results) {
  return inferReturnTypesOfStructExtractOp<llhd::SigType>(
      context, loc, operands, attrs, regions, results);
}

LogicalResult llhd::PtrStructExtractOp::inferReturnTypes(
    MLIRContext *context, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::RegionRange regions,
    SmallVectorImpl<Type> &results) {
  return inferReturnTypesOfStructExtractOp<llhd::PtrType>(
      context, loc, operands, attrs, regions, results);
}

//===----------------------------------------------------------------------===//
// DrvOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::DrvOp::fold(ArrayRef<Attribute> operands,
                                SmallVectorImpl<OpFoldResult> &result) {
  if (!enable())
    return failure();

  if (matchPattern(enable(), m_One())) {
    enableMutable().clear();
    return success();
  }

  return failure();
}

LogicalResult llhd::DrvOp::canonicalize(llhd::DrvOp op,
                                        PatternRewriter &rewriter) {
  if (!op.enable())
    return failure();

  if (matchPattern(op.enable(), m_Zero())) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

// Implement this operation for the BranchOpInterface
Optional<MutableOperandRange>
llhd::WaitOp::getMutableSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return destOpsMutable();
}

//===----------------------------------------------------------------------===//
// EntityOp
//===----------------------------------------------------------------------===//

/// Parse an argument list of an entity operation.
/// The argument list and argument types are returned in args and argTypes
/// respectively.
static ParseResult
parseArgumentList(OpAsmParser &parser,
                  SmallVectorImpl<OpAsmParser::OperandType> &args,
                  SmallVectorImpl<Type> &argTypes) {
  auto parseElt = [&]() -> ParseResult {
    OpAsmParser::OperandType argument;
    Type argType;
    if (succeeded(parser.parseOptionalRegionArgument(argument))) {
      if (!argument.name.empty() && succeeded(parser.parseColonType(argType))) {
        args.push_back(argument);
        argTypes.push_back(argType);
      }
    }
    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseElt);
}

/// parse an entity signature with syntax:
/// (%arg0 : T0, %arg1 : T1, <...>) -> (%out0 : T0, %out1 : T1, <...>)
static ParseResult
parseEntitySignature(OpAsmParser &parser, OperationState &result,
                     SmallVectorImpl<OpAsmParser::OperandType> &args,
                     SmallVectorImpl<Type> &argTypes) {
  if (parseArgumentList(parser, args, argTypes))
    return failure();
  // create the integer attribute with the number of inputs.
  IntegerAttr insAttr = parser.getBuilder().getI64IntegerAttr(args.size());
  result.addAttribute("ins", insAttr);
  if (parser.parseArrow() || parseArgumentList(parser, args, argTypes))
    return failure();

  return success();
}

static ParseResult parseEntityOp(OpAsmParser &parser, OperationState &result) {
  StringAttr entityName;
  SmallVector<OpAsmParser::OperandType, 4> args;
  SmallVector<Type, 4> argTypes;

  if (parser.parseSymbolName(entityName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  parseEntitySignature(parser, result, args, argTypes);

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto type = parser.getBuilder().getFunctionType(argTypes, llvm::None);
  result.addAttribute(circt::llhd::EntityOp::getTypeAttrName(),
                      TypeAttr::get(type));

  auto &body = *result.addRegion();
  if (parser.parseRegion(body, args, argTypes))
    return failure();
  if (body.empty())
    body.push_back(std::make_unique<Block>().release());

  return success();
}

static void printArgumentList(OpAsmPrinter &printer,
                              std::vector<BlockArgument> args) {
  printer << "(";
  llvm::interleaveComma(args, printer, [&](BlockArgument arg) {
    printer << arg << " : " << arg.getType();
  });
  printer << ")";
}

static void printEntityOp(OpAsmPrinter &printer, llhd::EntityOp op) {
  std::vector<BlockArgument> ins, outs;
  uint64_t n_ins = op.insAttr().getInt();
  for (uint64_t i = 0; i < op.body().front().getArguments().size(); ++i) {
    // no furter verification for the attribute type is required, already
    // handled by verify.
    if (i < n_ins) {
      ins.push_back(op.body().front().getArguments()[i]);
    } else {
      outs.push_back(op.body().front().getArguments()[i]);
    }
  }
  auto entityName =
      op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
          .getValue();
  printer << " ";
  printer.printSymbolName(entityName);
  printer << " ";
  printArgumentList(printer, ins);
  printer << " -> ";
  printArgumentList(printer, outs);
  printer.printOptionalAttrDictWithKeyword(
      op->getAttrs(),
      /*elidedAttrs =*/{SymbolTable::getSymbolAttrName(),
                        llhd::EntityOp::getTypeAttrName(), "ins"});
  printer << " ";
  printer.printRegion(op.body(), false, false);
}

static LogicalResult verify(llhd::EntityOp op) {
  uint64_t numArgs = op.getNumArguments();
  uint64_t nIns = op.insAttr().getInt();
  // check that there is at most one flag for each argument
  if (numArgs < nIns) {
    return op.emitError(
               "Cannot have more inputs than arguments, expected at most ")
           << numArgs << " but got: " << nIns;
  }

  // Check that all block arguments are of signal type
  for (size_t i = 0; i < numArgs; ++i)
    if (!op.getArgument(i).getType().isa<llhd::SigType>())
      return op.emitError("usage of invalid argument type. Got ")
             << op.getArgument(i).getType() << ", expected LLHD signal type";

  return success();
}

LogicalResult circt::llhd::EntityOp::verifyType() {
  FunctionType type = getType();

  // Fail if function returns any values. An entity's outputs are specially
  // marked arguments.
  if (type.getNumResults() > 0)
    return emitOpError("an entity cannot have return types.");

  // Check that all operands are of signal type
  for (Type inputType : type.getInputs())
    if (!inputType.isa<llhd::SigType>())
      return emitOpError("usage of invalid argument type. Got ")
             << inputType << ", expected LLHD signal type";

  return success();
}

LogicalResult circt::llhd::EntityOp::verifyBody() {
  // check signal names are unique
  llvm::StringMap<bool> sigMap;
  llvm::StringMap<bool> instMap;
  auto walkResult = walk([&sigMap, &instMap](Operation *op) -> WalkResult {
    if (auto sigOp = dyn_cast<SigOp>(op)) {
      if (sigMap[sigOp.name()]) {
        return sigOp.emitError("Redefinition of signal named '")
               << sigOp.name() << "'!";
      }
      sigMap.insert_or_assign(sigOp.name(), true);
    } else if (auto instOp = dyn_cast<InstOp>(op)) {
      if (instMap[instOp.name()]) {
        return instOp.emitError("Redefinition of instance named '")
               << instOp.name() << "'!";
      }
      instMap.insert_or_assign(instOp.name(), true);
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

Region *llhd::EntityOp::getCallableRegion() {
  return isExternal() ? nullptr : &getBody();
}

ArrayRef<Type> llhd::EntityOp::getCallableResults() {
  return getType().getResults();
}

//===----------------------------------------------------------------------===//
// ProcOp
//===----------------------------------------------------------------------===//

LogicalResult circt::llhd::ProcOp::verifyType() {
  // Fail if function returns more than zero values. This is because the
  // outputs of a process are specially marked arguments.
  if (getNumResults() > 0) {
    return emitOpError(
        "process has more than zero return types, this is not allowed");
  }

  // Check that all operands are of signal type
  for (int i = 0, e = getNumFuncArguments(); i < e; ++i) {
    if (!getArgument(i).getType().isa<llhd::SigType>()) {
      return emitOpError("usage of invalid argument type, was ")
             << getArgument(i).getType() << ", expected LLHD signal type";
    }
  }
  return success();
}

LogicalResult circt::llhd::ProcOp::verifyBody() { return success(); }

static LogicalResult verify(llhd::ProcOp op) {
  // Check that the ins attribute is smaller or equal the number of
  // arguments
  uint64_t numArgs = op.getNumArguments();
  uint64_t numIns = op.insAttr().getInt();
  if (numArgs < numIns) {
    return op.emitOpError(
               "Cannot have more inputs than arguments, expected at most ")
           << numArgs << ", got " << numIns;
  }
  return success();
}

static ParseResult
parseProcArgumentList(OpAsmParser &parser, SmallVectorImpl<Type> &argTypes,
                      SmallVectorImpl<OpAsmParser::OperandType> &argNames) {
  if (parser.parseLParen())
    return failure();

  // The argument list either has to consistently have ssa-id's followed by
  // types, or just be a type list.  It isn't ok to sometimes have SSA ID's
  // and sometimes not.
  auto parseArgument = [&]() -> ParseResult {
    llvm::SMLoc loc = parser.getCurrentLocation();

    // Parse argument name if present.
    OpAsmParser::OperandType argument;
    Type argumentType;
    if (succeeded(parser.parseOptionalRegionArgument(argument)) &&
        !argument.name.empty()) {
      // Reject this if the preceding argument was missing a name.
      if (argNames.empty() && !argTypes.empty())
        return parser.emitError(loc, "expected type instead of SSA identifier");
      argNames.push_back(argument);

      if (parser.parseColonType(argumentType))
        return failure();
    } else if (!argNames.empty()) {
      // Reject this if the preceding argument had a name.
      return parser.emitError(loc, "expected SSA identifier");
    } else if (parser.parseType(argumentType)) {
      return failure();
    }

    // Add the argument type.
    argTypes.push_back(argumentType);

    return success();
  };

  // Parse the function arguments.
  if (failed(parser.parseOptionalRParen())) {
    do {
      unsigned numTypedArguments = argTypes.size();
      if (parseArgument())
        return failure();

      llvm::SMLoc loc = parser.getCurrentLocation();
      if (argTypes.size() == numTypedArguments &&
          succeeded(parser.parseOptionalComma()))
        return parser.emitError(loc, "variadic arguments are not allowed");
    } while (succeeded(parser.parseOptionalComma()));
    parser.parseRParen();
  }

  return success();
}

static ParseResult parseProcOp(OpAsmParser &parser, OperationState &result) {
  StringAttr procName;
  SmallVector<OpAsmParser::OperandType, 8> argNames;
  SmallVector<Type, 8> argTypes;
  Builder &builder = parser.getBuilder();

  if (parser.parseSymbolName(procName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (parseProcArgumentList(parser, argTypes, argNames))
    return failure();

  result.addAttribute("ins", builder.getI64IntegerAttr(argTypes.size()));
  if (parser.parseArrow())
    return failure();

  if (parseProcArgumentList(parser, argTypes, argNames))
    return failure();

  auto type = builder.getFunctionType(argTypes, llvm::None);
  result.addAttribute(circt::llhd::ProcOp::getTypeAttrName(),
                      TypeAttr::get(type));

  auto *body = result.addRegion();
  parser.parseRegion(*body, argNames,
                     argNames.empty() ? ArrayRef<Type>() : argTypes);

  return success();
}

/// Print the signature of the `proc` unit. Assumes that it passed the
/// verification.
static void printProcArguments(OpAsmPrinter &p, Operation *op,
                               ArrayRef<Type> types, uint64_t numIns) {
  Region &body = op->getRegion(0);
  auto printList = [&](unsigned i, unsigned max) -> void {
    for (; i < max; ++i) {
      p << body.front().getArgument(i) << " : " << types[i];
      p.printOptionalAttrDict(::mlir::function_like_impl::getArgAttrs(op, i));

      if (i < max - 1)
        p << ", ";
    }
  };

  p << '(';
  printList(0, numIns);
  p << ") -> (";
  printList(numIns, types.size());
  p << ')';
}

static void printProcOp(OpAsmPrinter &printer, llhd::ProcOp op) {
  FunctionType type = op.getType();
  printer << ' ';
  printer.printSymbolName(op.getName());
  printProcArguments(printer, op.getOperation(), type.getInputs(),
                     op.insAttr().getInt());
  printer << " ";
  printer.printRegion(op.body(), false, true);
}

Region *llhd::ProcOp::getCallableRegion() {
  return isExternal() ? nullptr : &getBody();
}

ArrayRef<Type> llhd::ProcOp::getCallableResults() {
  return getType().getResults();
}

//===----------------------------------------------------------------------===//
// InstOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(llhd::InstOp op) {
  // Check that the callee attribute was specified.
  auto calleeAttr = op->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!calleeAttr)
    return op.emitOpError("requires a 'callee' symbol reference attribute");

  auto proc = op->getParentOfType<ModuleOp>().lookupSymbol<llhd::ProcOp>(
      calleeAttr.getValue());
  auto entity = op->getParentOfType<ModuleOp>().lookupSymbol<llhd::EntityOp>(
      calleeAttr.getValue());

  // Verify that the input and output types match the callee.
  if (proc) {
    auto type = proc.getType();

    if (proc.ins() != op.inputs().size())
      return op.emitOpError(
          "incorrect number of inputs for proc instantiation");

    if (type.getNumInputs() != op.getNumOperands())
      return op.emitOpError(
          "incorrect number of outputs for proc instantiation");

    for (size_t i = 0, e = type.getNumInputs(); i != e; ++i)
      if (op.getOperand(i).getType() != type.getInput(i))
        return op.emitOpError("operand type mismatch");

    return success();
  }
  if (entity) {
    auto type = entity.getType();

    if (entity.ins() != op.inputs().size())
      return op.emitOpError(
          "incorrect number of inputs for entity instantiation");

    if (type.getNumInputs() != op.getNumOperands())
      return op.emitOpError(
          "incorrect number of outputs for entity instantiation");

    for (size_t i = 0, e = type.getNumInputs(); i != e; ++i)
      if (op.getOperand(i).getType() != type.getInput(i))
        return op.emitOpError("operand type mismatch");

    return success();
  }
  return op.emitOpError() << "'" << calleeAttr.getValue()
                          << "' does not reference a valid proc or entity";
}

FunctionType llhd::InstOp::getCalleeType() {
  SmallVector<Type, 8> argTypes(getOperandTypes());
  return FunctionType::get(getContext(), argTypes, ArrayRef<Type>());
}

//===----------------------------------------------------------------------===//
// ConnectOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::ConnectOp::canonicalize(llhd::ConnectOp op,
                                            PatternRewriter &rewriter) {
  if (op.lhs() == op.rhs())
    rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// RegOp
//===----------------------------------------------------------------------===//

static ParseResult parseRegOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType signal;
  Type signalType;
  SmallVector<OpAsmParser::OperandType, 8> valueOperands;
  SmallVector<OpAsmParser::OperandType, 8> triggerOperands;
  SmallVector<OpAsmParser::OperandType, 8> delayOperands;
  SmallVector<OpAsmParser::OperandType, 8> gateOperands;
  SmallVector<Type, 8> valueTypes;
  llvm::SmallVector<int64_t, 8> modesArray;
  llvm::SmallVector<int64_t, 8> gateMask;
  int64_t gateCount = 0;

  if (parser.parseOperand(signal))
    return failure();
  while (succeeded(parser.parseOptionalComma())) {
    OpAsmParser::OperandType value;
    OpAsmParser::OperandType trigger;
    OpAsmParser::OperandType delay;
    OpAsmParser::OperandType gate;
    Type valueType;
    StringAttr modeAttr;
    NamedAttrList attrStorage;

    if (parser.parseLParen())
      return failure();
    if (parser.parseOperand(value) || parser.parseComma())
      return failure();
    if (parser.parseAttribute(modeAttr, parser.getBuilder().getNoneType(),
                              "modes", attrStorage))
      return failure();
    auto attrOptional = llhd::symbolizeRegMode(modeAttr.getValue());
    if (!attrOptional)
      return parser.emitError(parser.getCurrentLocation(),
                              "invalid string attribute");
    modesArray.push_back(static_cast<int64_t>(attrOptional.getValue()));
    if (parser.parseOperand(trigger))
      return failure();
    if (parser.parseKeyword("after") || parser.parseOperand(delay))
      return failure();
    if (succeeded(parser.parseOptionalKeyword("if"))) {
      gateMask.push_back(++gateCount);
      if (parser.parseOperand(gate))
        return failure();
      gateOperands.push_back(gate);
    } else {
      gateMask.push_back(0);
    }
    if (parser.parseColon() || parser.parseType(valueType) ||
        parser.parseRParen())
      return failure();
    valueOperands.push_back(value);
    triggerOperands.push_back(trigger);
    delayOperands.push_back(delay);
    valueTypes.push_back(valueType);
  }
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(signalType))
    return failure();
  if (parser.resolveOperand(signal, signalType, result.operands))
    return failure();
  if (parser.resolveOperands(valueOperands, valueTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();
  for (auto operand : triggerOperands)
    if (parser.resolveOperand(operand, parser.getBuilder().getI1Type(),
                              result.operands))
      return failure();
  for (auto operand : delayOperands)
    if (parser.resolveOperand(
            operand, llhd::TimeType::get(parser.getBuilder().getContext()),
            result.operands))
      return failure();
  for (auto operand : gateOperands)
    if (parser.resolveOperand(operand, parser.getBuilder().getI1Type(),
                              result.operands))
      return failure();
  result.addAttribute("gateMask",
                      parser.getBuilder().getI64ArrayAttr(gateMask));
  result.addAttribute("modes", parser.getBuilder().getI64ArrayAttr(modesArray));
  llvm::SmallVector<int32_t, 5> operandSizes;
  operandSizes.push_back(1);
  operandSizes.push_back(valueOperands.size());
  operandSizes.push_back(triggerOperands.size());
  operandSizes.push_back(delayOperands.size());
  operandSizes.push_back(gateOperands.size());
  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getI32VectorAttr(operandSizes));

  return success();
}

static void printRegOp(OpAsmPrinter &printer, llhd::RegOp op) {
  printer << " " << op.signal();
  for (size_t i = 0, e = op.values().size(); i < e; ++i) {
    Optional<llhd::RegMode> mode = llhd::symbolizeRegMode(
        op.modes().getValue()[i].cast<IntegerAttr>().getInt());
    if (!mode) {
      op.emitError("invalid RegMode");
      return;
    }
    printer << ", (" << op.values()[i] << ", \""
            << llhd::stringifyRegMode(mode.getValue()) << "\" "
            << op.triggers()[i] << " after " << op.delays()[i];
    if (op.hasGate(i))
      printer << " if " << op.getGateAt(i);
    printer << " : " << op.values()[i].getType() << ")";
  }
  printer.printOptionalAttrDict(op->getAttrs(),
                                {"modes", "gateMask", "operand_segment_sizes"});
  printer << " : " << op.signal().getType();
}

static LogicalResult verify(llhd::RegOp op) {
  // At least one trigger has to be present
  if (op.triggers().size() < 1)
    return op.emitError("At least one trigger quadruple has to be present.");

  // Values variadic operand must have the same size as the triggers variadic
  if (op.values().size() != op.triggers().size())
    return op.emitOpError("Number of 'values' is not equal to the number of "
                          "'triggers', got ")
           << op.values().size() << " modes, but " << op.triggers().size()
           << " triggers!";

  // Delay variadic operand must have the same size as the triggers variadic
  if (op.delays().size() != op.triggers().size())
    return op.emitOpError("Number of 'delays' is not equal to the number of "
                          "'triggers', got ")
           << op.delays().size() << " modes, but " << op.triggers().size()
           << " triggers!";

  // Array Attribute of RegModes must have the same number of elements as the
  // variadics
  if (op.modes().size() != op.triggers().size())
    return op.emitOpError("Number of 'modes' is not equal to the number of "
                          "'triggers', got ")
           << op.modes().size() << " modes, but " << op.triggers().size()
           << " triggers!";

  // Array Attribute 'gateMask' must have the same number of elements as the
  // triggers and values variadics
  if (op.gateMask().size() != op.triggers().size())
    return op.emitOpError("Size of 'gateMask' is not equal to the size of "
                          "'triggers', got ")
           << op.gateMask().size() << " modes, but " << op.triggers().size()
           << " triggers!";

  // Number of non-zero elements in 'gateMask' has to be the same as the size
  // of the gates variadic, also each number from 1 to size-1 has to occur
  // only once and in increasing order
  unsigned counter = 0;
  unsigned prevElement = 0;
  for (Attribute maskElem : op.gateMask().getValue()) {
    int64_t val = maskElem.cast<IntegerAttr>().getInt();
    if (val < 0)
      return op.emitError("Element in 'gateMask' must not be negative!");
    if (val == 0)
      continue;
    if (val != ++prevElement)
      return op.emitError(
          "'gateMask' has to contain every number from 1 to the "
          "number of gates minus one exactly once in increasing order "
          "(may have zeros in-between).");
    counter++;
  }
  if (op.gates().size() != counter)
    return op.emitError("The number of non-zero elements in 'gateMask' and the "
                        "size of the 'gates' variadic have to match.");

  // Each value must be either the same type as the 'signal' or the underlying
  // type of the 'signal'
  for (auto val : op.values()) {
    if (val.getType() != op.signal().getType() &&
        val.getType() !=
            op.signal().getType().cast<llhd::SigType>().getUnderlyingType()) {
      return op.emitOpError(
          "type of each 'value' has to be either the same as the "
          "type of 'signal' or the underlying type of 'signal'");
    }
  }
  return success();
}

#include "circt/Dialect/LLHD/IR/LLHDEnums.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/LLHD/IR/LLHD.cpp.inc"
