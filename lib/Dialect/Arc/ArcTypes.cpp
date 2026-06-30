//===- ArcTypes.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcTypes.h"
#include "circt/Dialect/Arc/ArcDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace arc;
using namespace mlir;

static ParseResult parseXInDimList(AsmParser &p, Type &elementType);
static void printXInDimList(AsmPrinter &p, Type elementType);

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Arc/ArcTypes.cpp.inc"

/// Compute the bit width a type will have when allocated as part of the
/// simulator's storage. This includes any padding and alignment that may be
/// necessary once the type has been mapped to LLVM. The idea is for this
/// function to be conservative, such that we provide sufficient storage bytes
/// for any type.
std::optional<uint64_t> circt::arc::computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;

  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();

  auto computeForArrayType = [&](auto arrayType) -> std::optional<uint64_t> {
    // Compute element width.
    auto maybeWidth = computeLLVMBitWidth(arrayType.getElementType());
    if (!maybeWidth)
      return {};
    // Each element must be at least one byte.
    auto width = std::max<uint64_t>(*maybeWidth, 8);
    // Align elements to their own size, up to 16 bytes.
    auto alignment = llvm::bit_ceil(std::min<uint64_t>(width, 16 * 8));
    auto alignedWidth = llvm::alignToPowerOf2(width, alignment);
    // Multiply by the number of elements in the array.
    return arrayType.getNumElements() * alignedWidth;
  };

  if (auto arrayType = dyn_cast<hw::ArrayType>(type))
    return computeForArrayType(arrayType);

  if (auto arrayRefType = dyn_cast<ArrayRefType>(type))
    return computeForArrayType(arrayRefType);

  if (auto structType = dyn_cast<hw::StructType>(type)) {
    uint64_t structWidth = 0;
    uint64_t structAlignment = 8;
    for (auto element : structType.getElements()) {
      // Compute element width.
      auto maybeWidth = computeLLVMBitWidth(element.type);
      if (!maybeWidth)
        return {};
      // Each element must be at least one byte.
      auto width = std::max<uint64_t>(*maybeWidth, 8);
      // Align elements to their own size, up to 16 bytes.
      auto alignment = llvm::bit_ceil(std::min<uint64_t>(width, 16 * 8));
      auto alignedWidth = llvm::alignToPowerOf2(width, alignment);
      // Pad the struct size to align the element to its alignment need, and add
      // the element.
      structWidth = llvm::alignToPowerOf2(structWidth, alignment);
      structWidth += alignedWidth;
      // Keep track of the struct's alignment need.
      structAlignment = std::max<uint64_t>(alignment, structAlignment);
    }
    // Pad the struct size to align its end with the alignment.
    return llvm::alignToPowerOf2(structWidth, structAlignment);
  }

  // A union's variants all overlap at the same base address, so its size is
  // determined by the largest variant rather than their sum. Honor any explicit
  // per-variant bit offset, and pad the whole thing to the union's alignment.
  if (auto unionType = dyn_cast<hw::UnionType>(type)) {
    uint64_t unionWidth = 0;
    uint64_t unionAlignment = 8;
    for (auto element : unionType.getElements()) {
      // Compute variant width.
      auto maybeWidth = computeLLVMBitWidth(element.type);
      if (!maybeWidth)
        return {};
      // Each variant must be at least one byte.
      auto width = std::max<uint64_t>(*maybeWidth, 8);
      // Align variants to their own size, up to 16 bytes.
      auto alignment = llvm::bit_ceil(std::min<uint64_t>(width, 16 * 8));
      auto alignedWidth = llvm::alignToPowerOf2(width, alignment);
      // The union must be large enough to hold this variant at its offset.
      unionWidth =
          std::max<uint64_t>(unionWidth, element.offset + alignedWidth);
      // Keep track of the union's alignment need.
      unionAlignment = std::max<uint64_t>(alignment, unionAlignment);
    }
    // Pad the union size to align its end with the alignment.
    return llvm::alignToPowerOf2(unionWidth, unionAlignment);
  }

  // We don't know anything about any other types.
  return {};
}

unsigned StateType::getBitWidth() { return *computeLLVMBitWidth(getType()); }

LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  // A coroutine's program counter and persistent state have a layout that is
  // only fixed once coroutines are lowered. Permit them as state types with a
  // deferred bit width; lowering later replaces them with concrete types.
  if (isa<CoroutinePCType, CoroutineStateType>(innerType))
    return success();
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}

unsigned MemoryType::getStride() {
  unsigned stride = (getWordType().getWidth() + 7) / 8;
  return llvm::alignToPowerOf2(stride, llvm::bit_ceil(std::min(stride, 16U)));
}

size_t ArrayRefType::getNumElements() const { return getSize(); }

std::optional<int64_t> ArrayRefType::getBitWidth() const {
  auto elementBitWidth = hw::getBitWidth(getElementType());
  if (elementBitWidth < 0)
    return std::nullopt;
  int64_t numElements = getNumElements();
  if (numElements < 0)
    return std::nullopt;
  return numElements * elementBitWidth;
}

ShapedType ArrayRefType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                   Type elementType) const {
  llvm_unreachable("ArrayRefType::cloneWith not implemented!");
}

bool ArrayRefType::hasRank() const { return true; }

ArrayRef<int64_t> ArrayRefType::getShape() const {
  const uint64_t &size = getImpl()->size;
  return ArrayRef<int64_t>(reinterpret_cast<const int64_t *>(&size), 1);
}

static ParseResult parseXInDimList(AsmParser &p, Type &elementType) {
  return failure(p.parseXInDimensionList() || p.parseType(elementType));
}

static void printXInDimList(AsmPrinter &p, Type elementType) {
  p << "x" << elementType;
}

LogicalResult
ArrayRefType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                     Type elementType, uint64_t size) {
  if (size == 0) {
    return emitError() << "must have nonzero element count";
  }
  return success();
}

void ArcDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Arc/ArcTypes.cpp.inc"
      >();
}
