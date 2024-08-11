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
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

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

  if (auto val = dyn_cast<AttrElementT>(operands[0])) {
    return AttrElementT::get(val.getType(), calculate(val.getValue()));
  } else if (auto val = dyn_cast<SplatElementsAttr>(operands[0])) {
    // Operand is a splat so we can avoid expanding the value out and
    // just fold based on the splat value.
    auto elementResult = calculate(val.getSplatValue<ElementValueT>());
    return DenseElementsAttr::get(val.getType(), elementResult);
  }
  if (auto val = dyn_cast<ElementsAttr>(operands[0])) {
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

  if (isa<AttrElementT>(operands[0]) && isa<AttrElementT>(operands[1]) &&
      isa<AttrElementT>(operands[2])) {
    auto fst = cast<AttrElementT>(operands[0]);
    auto snd = cast<AttrElementT>(operands[1]);
    auto trd = cast<AttrElementT>(operands[2]);

    return AttrElementT::get(
        fst.getType(),
        calculate(fst.getValue(), snd.getValue(), trd.getValue()));
  }
  if (isa<SplatElementsAttr>(operands[0]) &&
      isa<SplatElementsAttr>(operands[1]) &&
      isa<SplatElementsAttr>(operands[2])) {
    // Operands are splats so we can avoid expanding the values out and
    // just fold based on the splat value.
    auto fst = cast<SplatElementsAttr>(operands[0]);
    auto snd = cast<SplatElementsAttr>(operands[1]);
    auto trd = cast<SplatElementsAttr>(operands[2]);

    auto elementResult = calculate(fst.getSplatValue<ElementValueT>(),
                                   snd.getSplatValue<ElementValueT>(),
                                   trd.getSplatValue<ElementValueT>());
    return DenseElementsAttr::get(fst.getType(), elementResult);
  }
  if (isa<ElementsAttr>(operands[0]) && isa<ElementsAttr>(operands[1]) &&
      isa<ElementsAttr>(operands[2])) {
    // Operands are ElementsAttr-derived; perform an element-wise fold by
    // expanding the values.
    auto fst = cast<ElementsAttr>(operands[0]);
    auto snd = cast<ElementsAttr>(operands[1]);
    auto trd = cast<ElementsAttr>(operands[2]);

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
    return mlir::detail::constant_int_value_binder(&value).match(op) &&
           value.isAllOnes();
  }
};

} // anonymous namespace

unsigned circt::llhd::getLLHDTypeWidth(Type type) {
  if (auto sig = dyn_cast<hw::InOutType>(type))
    type = sig.getElementType();
  else if (auto ptr = dyn_cast<llhd::PtrType>(type))
    type = ptr.getElementType();
  if (auto array = dyn_cast<hw::ArrayType>(type))
    return array.getNumElements();
  if (auto tup = dyn_cast<hw::StructType>(type))
    return tup.getElements().size();
  return type.getIntOrFloatBitWidth();
}

Type circt::llhd::getLLHDElementType(Type type) {
  if (auto sig = dyn_cast<hw::InOutType>(type))
    type = sig.getElementType();
  else if (auto ptr = dyn_cast<llhd::PtrType>(type))
    type = ptr.getElementType();
  if (auto array = dyn_cast<hw::ArrayType>(type))
    return array.getElementType();
  return type;
}

//===---------------------------------------------------------------------===//
// LLHD Operations
//===---------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantTimeOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::ConstantTimeOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "const has no operands");
  return getValueAttr();
}

void llhd::ConstantTimeOp::build(OpBuilder &builder, OperationState &result,
                                 unsigned time, const StringRef &timeUnit,
                                 unsigned delta, unsigned epsilon) {
  auto *ctx = builder.getContext();
  auto attr = TimeAttr::get(ctx, time, timeUnit, delta, epsilon);
  return build(builder, result, TimeType::get(ctx), attr);
}

//===----------------------------------------------------------------------===//
// SigExtractOp and PtrExtractOp
//===----------------------------------------------------------------------===//

template <class Op>
static OpFoldResult foldSigPtrExtractOp(Op op, ArrayRef<Attribute> operands) {

  if (!operands[1])
    return nullptr;

  // llhd.sig.extract(input, 0) with inputWidth == resultWidth => input
  if (op.getResultWidth() == op.getInputWidth() &&
      cast<IntegerAttr>(operands[1]).getValue().isZero())
    return op.getInput();

  return nullptr;
}

OpFoldResult llhd::SigExtractOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrExtractOp(*this, adaptor.getOperands());
}

OpFoldResult llhd::PtrExtractOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrExtractOp(*this, adaptor.getOperands());
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
  if (op.getResultWidth() == op.getInputWidth() &&
      cast<IntegerAttr>(operands[1]).getValue().isZero())
    return op.getInput();

  return nullptr;
}

OpFoldResult llhd::SigArraySliceOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrArraySliceOp(*this, adaptor.getOperands());
}

OpFoldResult llhd::PtrArraySliceOp::fold(FoldAdaptor adaptor) {
  return foldSigPtrArraySliceOp(*this, adaptor.getOperands());
}

template <class Op>
static LogicalResult canonicalizeSigPtrArraySliceOp(Op op,
                                                    PatternRewriter &rewriter) {
  IntegerAttr indexAttr;
  if (!matchPattern(op.getLowIndex(), m_Constant(&indexAttr)))
    return failure();

  // llhd.sig.array_slice(llhd.sig.array_slice(target, a), b)
  //   => llhd.sig.array_slice(target, a+b)
  IntegerAttr a;
  if (matchPattern(op.getInput(),
                   m_Op<Op>(matchers::m_Any(), m_Constant(&a)))) {
    auto sliceOp = op.getInput().template getDefiningOp<Op>();
    rewriter.modifyOpInPlace(op, [&]() {
      op.getInputMutable().assign(sliceOp.getInput());
      Value newIndex = rewriter.create<hw::ConstantOp>(
          op->getLoc(), a.getValue() + indexAttr.getValue());
      op.getLowIndexMutable().assign(newIndex);
    });

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
// SigStructExtractOp and PtrStructExtractOp
//===----------------------------------------------------------------------===//

template <class SigPtrType>
static LogicalResult inferReturnTypesOfStructExtractOp(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  Type type =
      cast<hw::StructType>(
          cast<SigPtrType>(operands[0].getType()).getElementType())
          .getFieldType(
              cast<StringAttr>(attrs.getNamed("field")->getValue()).getValue());
  if (!type) {
    context->getDiagEngine().emit(loc.value_or(UnknownLoc()),
                                  DiagnosticSeverity::Error)
        << "invalid field name specified";
    return failure();
  }
  results.push_back(SigPtrType::get(type));
  return success();
}

LogicalResult llhd::SigStructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferReturnTypesOfStructExtractOp<hw::InOutType>(
      context, loc, operands, attrs, properties, regions, results);
}

LogicalResult llhd::PtrStructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferReturnTypesOfStructExtractOp<llhd::PtrType>(
      context, loc, operands, attrs, properties, regions, results);
}

//===----------------------------------------------------------------------===//
// DrvOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::DrvOp::fold(FoldAdaptor adaptor,
                                SmallVectorImpl<OpFoldResult> &result) {
  if (!getEnable())
    return failure();

  if (matchPattern(getEnable(), m_One())) {
    getEnableMutable().clear();
    return success();
  }

  return failure();
}

LogicalResult llhd::DrvOp::canonicalize(llhd::DrvOp op,
                                        PatternRewriter &rewriter) {
  if (!op.getEnable())
    return failure();

  if (matchPattern(op.getEnable(), m_Zero())) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

// Implement this operation for the BranchOpInterface
SuccessorOperands llhd::WaitOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOpsMutable());
}

//===----------------------------------------------------------------------===//
// ConnectOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::ConnectOp::canonicalize(llhd::ConnectOp op,
                                            PatternRewriter &rewriter) {
  if (op.getLhs() == op.getRhs())
    rewriter.eraseOp(op);
  return success();
}

#include "circt/Dialect/LLHD/IR/LLHDEnums.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/LLHD/IR/LLHD.cpp.inc"
