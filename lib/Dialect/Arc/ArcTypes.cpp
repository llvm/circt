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

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/Arc/ArcTypes.cpp.inc"

/// Compute the bit width a type will have when allocated as part of the
/// simulator's storage. This includes any padding and alignment that may be
/// necessary once the type has been mapped to LLVM. The idea is for this
/// function to be conservative, such that we provide sufficient storage bytes
/// for any type.
static std::optional<uint64_t> computeLLVMBitWidth(Type type) {
  if (isa<seq::ClockType>(type))
    return 1;

  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();

  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
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
  }

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

  // We don't know anything about any other types.
  return {};
}

unsigned StateType::getBitWidth() { return *computeLLVMBitWidth(getType()); }

LogicalResult
StateType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  Type innerType) {
  if (!computeLLVMBitWidth(innerType))
    return emitError() << "state type must have a known bit width; got "
                       << innerType;
  return success();
}

unsigned MemoryType::getStride() {
  unsigned stride = (getWordType().getWidth() + 7) / 8;
  return llvm::alignToPowerOf2(stride, llvm::bit_ceil(std::min(stride, 16U)));
}

void ArcDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/Arc/ArcTypes.cpp.inc"
      >();
}
