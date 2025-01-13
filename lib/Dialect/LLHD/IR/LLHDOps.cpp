//===- LLHDOps.cpp - Implement the LLHD operations ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLHD ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace circt;
using namespace mlir;
using namespace llhd;

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
// SignalOp
//===----------------------------------------------------------------------===//

static Value getValueAtIndex(OpBuilder &builder, Location loc, Value val,
                             unsigned index) {
  return TypeSwitch<Type, Value>(val.getType())
      .Case<hw::StructType>([&](hw::StructType ty) -> Value {
        return builder.create<hw::StructExtractOp>(
            loc, val, ty.getElements()[index].name);
      })
      .Case<hw::ArrayType>([&](hw::ArrayType ty) -> Value {
        Value idx = builder.create<hw::ConstantOp>(
            loc,
            builder.getIntegerType(llvm::Log2_64_Ceil(ty.getNumElements())),
            index);
        return builder.create<hw::ArrayGetOp>(loc, val, idx);
      });
}

void SignalOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (getName() && !getName()->empty())
    setNameFn(getResult(), *getName());
}

SmallVector<DestructurableMemorySlot> SignalOp::getDestructurableSlots() {
  auto type = getType().getElementType();

  auto destructurable = llvm::dyn_cast<DestructurableTypeInterface>(type);
  if (!destructurable)
    return {};

  auto destructuredType = destructurable.getSubelementIndexMap();
  if (!destructuredType)
    return {};

  return {DestructurableMemorySlot{{getResult(), type}, *destructuredType}};
}

DenseMap<Attribute, MemorySlot> SignalOp::destructure(
    const DestructurableMemorySlot &slot,
    const SmallPtrSetImpl<Attribute> &usedIndices, OpBuilder &builder,
    SmallVectorImpl<DestructurableAllocationOpInterface> &newAllocators) {
  assert(slot.ptr == getResult());
  builder.setInsertionPointAfter(*this);

  auto destructurableType =
      cast<DestructurableTypeInterface>(getType().getElementType());
  DenseMap<Attribute, MemorySlot> slotMap;
  SmallVector<std::pair<unsigned, Type>> indices;
  for (auto attr : usedIndices) {
    assert(isa<IntegerAttr>(attr));
    auto elemType = destructurableType.getTypeAtIndex(attr);
    assert(elemType && "used index must exist");
    indices.push_back({cast<IntegerAttr>(attr).getInt(), elemType});
  }

  llvm::sort(indices, [](auto a, auto b) { return a.first < b.first; });

  for (auto [index, type] : indices) {
    Value init = getValueAtIndex(builder, getLoc(), getInit(), index);
    auto sigOp = builder.create<SignalOp>(getLoc(), getNameAttr(), init);
    newAllocators.push_back(sigOp);
    slotMap.try_emplace<MemorySlot>(
        IntegerAttr::get(IndexType::get(getContext()), index),
        {sigOp.getResult(), type});
  }

  return slotMap;
}

std::optional<DestructurableAllocationOpInterface>
SignalOp::handleDestructuringComplete(const DestructurableMemorySlot &slot,
                                      OpBuilder &builder) {
  assert(slot.ptr == getResult());
  this->erase();
  return std::nullopt;
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
// SigArrayGetOp
//===----------------------------------------------------------------------===//

bool SigArrayGetOp::canRewire(const DestructurableMemorySlot &slot,
                              SmallPtrSetImpl<Attribute> &usedIndices,
                              SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                              const DataLayout &dataLayout) {
  if (slot.ptr != getInput())
    return false;
  APInt idx;
  if (!matchPattern(getIndex(), m_ConstantInt(&idx)))
    return false;
  auto index =
      IntegerAttr::get(IndexType::get(getContext()), idx.getZExtValue());
  if (!slot.subelementTypes.contains(index))
    return false;
  usedIndices.insert(index);
  mustBeSafelyUsed.emplace_back<MemorySlot>(
      {getResult(),
       cast<hw::InOutType>(getResult().getType()).getElementType()});
  return true;
}

DeletionKind SigArrayGetOp::rewire(const DestructurableMemorySlot &slot,
                                   DenseMap<Attribute, MemorySlot> &subslots,
                                   OpBuilder &builder,
                                   const DataLayout &dataLayout) {
  APInt idx;
  bool result = matchPattern(getIndex(), m_ConstantInt(&idx));
  (void)result;
  assert(result);
  auto index =
      IntegerAttr::get(IndexType::get(getContext()), idx.getZExtValue());
  auto it = subslots.find(index);
  assert(it != subslots.end());
  replaceAllUsesWith(it->getSecond().ptr);
  return DeletionKind::Delete;
}

LogicalResult SigArrayGetOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success();
}

//===----------------------------------------------------------------------===//
// SigStructExtractOp and PtrStructExtractOp
//===----------------------------------------------------------------------===//

template <class OpType, class SigPtrType>
static LogicalResult inferReturnTypesOfStructExtractOp(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  typename OpType::Adaptor adaptor(operands, attrs, properties, regions);
  Type type =
      cast<hw::StructType>(
          cast<SigPtrType>(adaptor.getInput().getType()).getElementType())
          .getFieldType(adaptor.getField());
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
  return inferReturnTypesOfStructExtractOp<llhd::SigStructExtractOp,
                                           hw::InOutType>(
      context, loc, operands, attrs, properties, regions, results);
}

LogicalResult llhd::PtrStructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  return inferReturnTypesOfStructExtractOp<llhd::PtrStructExtractOp,
                                           llhd::PtrType>(
      context, loc, operands, attrs, properties, regions, results);
}

bool SigStructExtractOp::canRewire(
    const DestructurableMemorySlot &slot,
    SmallPtrSetImpl<Attribute> &usedIndices,
    SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  if (slot.ptr != getInput())
    return false;
  auto index = cast<hw::StructType>(
                   cast<hw::InOutType>(getInput().getType()).getElementType())
                   .getFieldIndex(getFieldAttr());
  if (!index)
    return false;
  auto indexAttr = IntegerAttr::get(IndexType::get(getContext()), *index);
  if (!slot.subelementTypes.contains(indexAttr))
    return false;
  usedIndices.insert(indexAttr);
  mustBeSafelyUsed.emplace_back<MemorySlot>(
      {getResult(),
       cast<hw::InOutType>(getResult().getType()).getElementType()});
  return true;
}

DeletionKind
SigStructExtractOp::rewire(const DestructurableMemorySlot &slot,
                           DenseMap<Attribute, MemorySlot> &subslots,
                           OpBuilder &builder, const DataLayout &dataLayout) {
  auto index = cast<hw::StructType>(
                   cast<hw::InOutType>(getInput().getType()).getElementType())
                   .getFieldIndex(getFieldAttr());
  assert(index.has_value());
  auto indexAttr = IntegerAttr::get(IndexType::get(getContext()), *index);
  auto it = subslots.find(indexAttr);
  assert(it != subslots.end());
  replaceAllUsesWith(it->getSecond().ptr);
  return DeletionKind::Delete;
}

LogicalResult SigStructExtractOp::ensureOnlySafeAccesses(
    const MemorySlot &slot, SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  return success();
}

//===----------------------------------------------------------------------===//
// PrbOp
//===----------------------------------------------------------------------===//

static void getSortedPtrs(DenseMap<Attribute, MemorySlot> &subslots,
                          SmallVectorImpl<std::pair<unsigned, Value>> &sorted) {
  for (auto [attr, mem] : subslots) {
    assert(isa<IntegerAttr>(attr));
    sorted.push_back({cast<IntegerAttr>(attr).getInt(), mem.ptr});
  }

  llvm::sort(sorted, [](auto a, auto b) { return a.first < b.first; });
}

bool PrbOp::canRewire(const DestructurableMemorySlot &slot,
                      SmallPtrSetImpl<Attribute> &usedIndices,
                      SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                      const DataLayout &dataLayout) {
  for (auto [key, _] : slot.subelementTypes)
    usedIndices.insert(key);

  return isa<hw::StructType, hw::ArrayType>(slot.elemType);
}

DeletionKind PrbOp::rewire(const DestructurableMemorySlot &slot,
                           DenseMap<Attribute, MemorySlot> &subslots,
                           OpBuilder &builder, const DataLayout &dataLayout) {
  SmallVector<std::pair<unsigned, Value>> elements;
  SmallVector<Value> probed;
  getSortedPtrs(subslots, elements);
  for (auto [_, val] : elements)
    probed.push_back(builder.create<PrbOp>(getLoc(), val));

  Value repl = TypeSwitch<Type, Value>(getType())
                   .Case<hw::StructType>([&](auto ty) {
                     return builder.create<hw::StructCreateOp>(
                         getLoc(), getType(), probed);
                   })
                   .Case<hw::ArrayType>([&](auto ty) {
                     return builder.create<hw::ArrayCreateOp>(getLoc(), probed);
                   });

  replaceAllUsesWith(repl);
  return DeletionKind::Delete;
}

LogicalResult
PrbOp::ensureOnlySafeAccesses(const MemorySlot &slot,
                              SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                              const DataLayout &dataLayout) {
  return success();
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

bool DrvOp::canRewire(const DestructurableMemorySlot &slot,
                      SmallPtrSetImpl<Attribute> &usedIndices,
                      SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                      const DataLayout &dataLayout) {
  for (auto [key, _] : slot.subelementTypes)
    usedIndices.insert(key);

  return isa<hw::StructType, hw::ArrayType>(slot.elemType);
}

DeletionKind DrvOp::rewire(const DestructurableMemorySlot &slot,
                           DenseMap<Attribute, MemorySlot> &subslots,
                           OpBuilder &builder, const DataLayout &dataLayout) {
  SmallVector<std::pair<unsigned, Value>> driven;
  getSortedPtrs(subslots, driven);

  for (auto [idx, sig] : driven)
    builder.create<DrvOp>(getLoc(), sig,
                          getValueAtIndex(builder, getLoc(), getValue(), idx),
                          getTime(), getEnable());

  return DeletionKind::Delete;
}

LogicalResult
DrvOp::ensureOnlySafeAccesses(const MemorySlot &slot,
                              SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                              const DataLayout &dataLayout) {
  return success();
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
