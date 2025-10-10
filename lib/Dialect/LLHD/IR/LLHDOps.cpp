//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  if (auto sig = dyn_cast<RefType>(type))
    type = sig.getNestedType();
  if (auto array = dyn_cast<hw::ArrayType>(type))
    return array.getNumElements();
  if (auto tup = dyn_cast<hw::StructType>(type))
    return tup.getElements().size();
  return type.getIntOrFloatBitWidth();
}

Type circt::llhd::getLLHDElementType(Type type) {
  if (auto sig = dyn_cast<RefType>(type))
    type = sig.getNestedType();
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
                                 uint64_t time, const StringRef &timeUnit,
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
        return hw::StructExtractOp::create(builder, loc, val,
                                           ty.getElements()[index].name);
      })
      .Case<hw::ArrayType>([&](hw::ArrayType ty) -> Value {
        Value idx = hw::ConstantOp::create(
            builder, loc,
            builder.getIntegerType(llvm::Log2_64_Ceil(ty.getNumElements())),
            index);
        return hw::ArrayGetOp::create(builder, loc, val, idx);
      });
}

void SignalOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  if (getName() && !getName()->empty())
    setNameFn(getResult(), *getName());
}

SmallVector<DestructurableMemorySlot> SignalOp::getDestructurableSlots() {
  auto type = getType().getNestedType();

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
      cast<DestructurableTypeInterface>(getType().getNestedType());
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
    auto sigOp = SignalOp::create(builder, getLoc(), getNameAttr(), init);
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
// SigExtractOp
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

//===----------------------------------------------------------------------===//
// SigArraySliceOp
//===----------------------------------------------------------------------===//

OpFoldResult llhd::SigArraySliceOp::fold(FoldAdaptor adaptor) {
  auto lowIndex = dyn_cast_or_null<IntegerAttr>(adaptor.getLowIndex());
  if (!lowIndex)
    return {};

  // llhd.sig.array_slice(input, 0) with inputWidth == resultWidth => input
  if (getResultWidth() == getInputWidth() && lowIndex.getValue().isZero())
    return getInput();

  return {};
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
      Value newIndex = hw::ConstantOp::create(
          rewriter, op->getLoc(), a.getValue() + indexAttr.getValue());
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
      {getResult(), cast<RefType>(getResult().getType()).getNestedType()});
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
// SigStructExtractOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::SigStructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  typename SigStructExtractOp::Adaptor adaptor(operands, attrs, properties,
                                               regions);
  Type type = cast<hw::StructType>(
                  cast<RefType>(adaptor.getInput().getType()).getNestedType())
                  .getFieldType(adaptor.getField());
  if (!type) {
    context->getDiagEngine().emit(loc.value_or(UnknownLoc()),
                                  DiagnosticSeverity::Error)
        << "invalid field name specified";
    return failure();
  }
  results.push_back(RefType::get(type));
  return success();
}

bool SigStructExtractOp::canRewire(
    const DestructurableMemorySlot &slot,
    SmallPtrSetImpl<Attribute> &usedIndices,
    SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
    const DataLayout &dataLayout) {
  if (slot.ptr != getInput())
    return false;
  auto index =
      cast<hw::StructType>(cast<RefType>(getInput().getType()).getNestedType())
          .getFieldIndex(getFieldAttr());
  if (!index)
    return false;
  auto indexAttr = IntegerAttr::get(IndexType::get(getContext()), *index);
  if (!slot.subelementTypes.contains(indexAttr))
    return false;
  usedIndices.insert(indexAttr);
  mustBeSafelyUsed.emplace_back<MemorySlot>(
      {getResult(), cast<RefType>(getResult().getType()).getNestedType()});
  return true;
}

DeletionKind
SigStructExtractOp::rewire(const DestructurableMemorySlot &slot,
                           DenseMap<Attribute, MemorySlot> &subslots,
                           OpBuilder &builder, const DataLayout &dataLayout) {
  auto index =
      cast<hw::StructType>(cast<RefType>(getInput().getType()).getNestedType())
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
// ProbeOp
//===----------------------------------------------------------------------===//

static void getSortedPtrs(DenseMap<Attribute, MemorySlot> &subslots,
                          SmallVectorImpl<std::pair<unsigned, Value>> &sorted) {
  for (auto [attr, mem] : subslots) {
    assert(isa<IntegerAttr>(attr));
    sorted.push_back({cast<IntegerAttr>(attr).getInt(), mem.ptr});
  }

  llvm::sort(sorted, [](auto a, auto b) { return a.first < b.first; });
}

bool ProbeOp::canRewire(const DestructurableMemorySlot &slot,
                        SmallPtrSetImpl<Attribute> &usedIndices,
                        SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                        const DataLayout &dataLayout) {
  for (auto [key, _] : slot.subelementTypes)
    usedIndices.insert(key);

  return isa<hw::StructType, hw::ArrayType>(slot.elemType);
}

DeletionKind ProbeOp::rewire(const DestructurableMemorySlot &slot,
                             DenseMap<Attribute, MemorySlot> &subslots,
                             OpBuilder &builder, const DataLayout &dataLayout) {
  SmallVector<std::pair<unsigned, Value>> elements;
  SmallVector<Value> probed;
  getSortedPtrs(subslots, elements);
  for (auto [_, val] : elements)
    probed.push_back(ProbeOp::create(builder, getLoc(), val));

  Value repl =
      TypeSwitch<Type, Value>(getType())
          .Case<hw::StructType>([&](auto ty) {
            return hw::StructCreateOp::create(builder, getLoc(), getType(),
                                              probed);
          })
          .Case<hw::ArrayType>([&](auto ty) {
            return hw::ArrayCreateOp::create(builder, getLoc(), probed);
          });

  replaceAllUsesWith(repl);
  return DeletionKind::Delete;
}

LogicalResult
ProbeOp::ensureOnlySafeAccesses(const MemorySlot &slot,
                                SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                                const DataLayout &dataLayout) {
  return success();
}

void ProbeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (mayHaveSSADominance(*getOperation()->getParentRegion()))
    effects.emplace_back(MemoryEffects::Read::get(), &getSignalMutable());
}

//===----------------------------------------------------------------------===//
// DriveOp
//===----------------------------------------------------------------------===//

LogicalResult llhd::DriveOp::fold(FoldAdaptor adaptor,
                                  SmallVectorImpl<OpFoldResult> &result) {
  if (!getEnable())
    return failure();

  if (matchPattern(getEnable(), m_One())) {
    getEnableMutable().clear();
    return success();
  }

  return failure();
}

LogicalResult llhd::DriveOp::canonicalize(llhd::DriveOp op,
                                          PatternRewriter &rewriter) {
  if (!op.getEnable())
    return failure();

  if (matchPattern(op.getEnable(), m_Zero())) {
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

bool DriveOp::canRewire(const DestructurableMemorySlot &slot,
                        SmallPtrSetImpl<Attribute> &usedIndices,
                        SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                        const DataLayout &dataLayout) {
  for (auto [key, _] : slot.subelementTypes)
    usedIndices.insert(key);

  return isa<hw::StructType, hw::ArrayType>(slot.elemType);
}

DeletionKind DriveOp::rewire(const DestructurableMemorySlot &slot,
                             DenseMap<Attribute, MemorySlot> &subslots,
                             OpBuilder &builder, const DataLayout &dataLayout) {
  SmallVector<std::pair<unsigned, Value>> driven;
  getSortedPtrs(subslots, driven);

  for (auto [idx, sig] : driven)
    DriveOp::create(builder, getLoc(), sig,
                    getValueAtIndex(builder, getLoc(), getValue(), idx),
                    getTime(), getEnable());

  return DeletionKind::Delete;
}

LogicalResult
DriveOp::ensureOnlySafeAccesses(const MemorySlot &slot,
                                SmallVectorImpl<MemorySlot> &mustBeSafelyUsed,
                                const DataLayout &dataLayout) {
  return success();
}

//===----------------------------------------------------------------------===//
// ProcessOp
//===----------------------------------------------------------------------===//

LogicalResult ProcessOp::canonicalize(ProcessOp op, PatternRewriter &rewriter) {
  if (!op.getBody().hasOneBlock())
    return failure();

  auto &block = op.getBody().front();
  auto haltOp = dyn_cast<HaltOp>(block.getTerminator());
  if (!haltOp)
    return failure();

  if (op.getNumResults() == 0 && block.getOperations().size() == 1) {
    rewriter.eraseOp(op);
    return success();
  }

  // Only constants and halt terminator are expected in a single block.
  if (!llvm::all_of(block.without_terminator(), [](auto &bodyOp) {
        return bodyOp.template hasTrait<OpTrait::ConstantLike>();
      }))
    return failure();

  auto yieldOperands = haltOp.getYieldOperands();
  llvm::SmallDenseMap<Value, unsigned> uniqueOperands;
  llvm::SmallDenseMap<unsigned, unsigned> origToNewPos;
  llvm::BitVector operandsToErase(yieldOperands.size());

  for (auto [operandNo, operand] : llvm::enumerate(yieldOperands)) {
    auto *defOp = operand.getDefiningOp();
    if (defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
      // If the constant is available outside the process, use it directly;
      // otherwise move it outside.
      if (!defOp->getParentRegion()->isProperAncestor(&op.getBody())) {
        defOp->moveBefore(op);
      }
      rewriter.replaceAllUsesWith(op.getResult(operandNo), operand);
      operandsToErase.set(operandNo);
      continue;
    }

    // Identify duplicate operands to merge and compute updated result
    // positions for the process operation.
    if (!uniqueOperands.contains(operand)) {
      const auto newPos = uniqueOperands.size();
      uniqueOperands.insert(std::make_pair(operand, newPos));
      origToNewPos.insert(std::make_pair(operandNo, newPos));
    } else {
      auto firstOccurrencePos = uniqueOperands.lookup(operand);
      origToNewPos.insert(std::make_pair(operandNo, firstOccurrencePos));
      operandsToErase.set(operandNo);
    }
  }

  const auto countOperandsToErase = operandsToErase.count();
  if (countOperandsToErase == 0)
    return failure();

  // Remove the process operation if all its results have been replaced with
  // constants.
  if (countOperandsToErase == op.getNumResults()) {
    rewriter.eraseOp(op);
    return success();
  }

  haltOp->eraseOperands(operandsToErase);

  SmallVector<Type> resultTypes = llvm::to_vector(haltOp->getOperandTypes());
  auto newProcessOp = ProcessOp::create(rewriter, op.getLoc(), resultTypes,
                                        op->getOperands(), op->getAttrs());
  newProcessOp.getBody().takeBody(op.getBody());

  // Update old results with new values, accounting for pruned halt operands.
  for (auto oldResult : op.getResults()) {
    auto newResultPos = origToNewPos.find(oldResult.getResultNumber());
    if (newResultPos == origToNewPos.end())
      continue;
    auto newResult = newProcessOp.getResult(newResultPos->getSecond());
    rewriter.replaceAllUsesWith(oldResult, newResult);
  }

  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// CombinationalOp
//===----------------------------------------------------------------------===//

LogicalResult CombinationalOp::canonicalize(CombinationalOp op,
                                            PatternRewriter &rewriter) {
  // Inline the combinational region if it consists of a single block and
  // contains no side-effecting operations.
  if (op.getBody().hasOneBlock() && isMemoryEffectFree(op)) {
    auto &block = op.getBody().front();
    auto *terminator = block.getTerminator();
    rewriter.inlineBlockBefore(&block, op, ValueRange{});
    rewriter.replaceOp(op, terminator->getOperands());
    rewriter.eraseOp(terminator);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyYieldResults(Operation *op,
                                        ValueRange yieldOperands) {
  // Determine the result values of the parent.
  auto *parentOp = op->getParentOp();
  TypeRange resultTypes = TypeSwitch<Operation *, TypeRange>(parentOp)
                              .Case<ProcessOp, CombinationalOp>(
                                  [](auto op) { return op.getResultTypes(); })
                              .Case<FinalOp>([](auto) { return TypeRange{}; });

  // Check that the number of yield operands matches the process.
  if (yieldOperands.size() != resultTypes.size())
    return op->emitOpError()
           << "has " << yieldOperands.size()
           << " yield operands, but enclosing '" << parentOp->getName()
           << "' returns " << resultTypes.size();

  // Check that the types match.
  for (unsigned i = 0; i < yieldOperands.size(); ++i)
    if (yieldOperands[i].getType() != resultTypes[i])
      return op->emitError()
             << "type of yield operand " << i << " ("
             << yieldOperands[i].getType() << ") does not match enclosing '"
             << parentOp->getName() << "' result type (" << resultTypes[i]
             << ")";

  return success();
}

LogicalResult WaitOp::verify() {
  return verifyYieldResults(*this, getYieldOperands());
}

//===----------------------------------------------------------------------===//
// HaltOp
//===----------------------------------------------------------------------===//

LogicalResult HaltOp::verify() {
  return verifyYieldResults(*this, getYieldOperands());
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  return verifyYieldResults(*this, getYieldOperands());
}

//===----------------------------------------------------------------------===//
// Auto-Generated Implementations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/LLHD/IR/LLHD.cpp.inc"
#include "circt/Dialect/LLHD/IR/LLHDEnums.cpp.inc"
