//===- FIRRTLUtils.cpp - FIRRTL IR Utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines various utilties to help generate and process FIRRTL IR.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace firrtl;

void circt::firrtl::emitConnect(OpBuilder &builder, Location loc, Value dst,
                                Value src) {
  ImplicitLocOpBuilder locBuilder(loc, builder.getInsertionBlock(),
                                  builder.getInsertionPoint());
  emitConnect(locBuilder, dst, src);
  builder.restoreInsertionPoint(locBuilder.saveInsertionPoint());
}

/// Emit a connect between two values.
void circt::firrtl::emitConnect(ImplicitLocOpBuilder &builder, Value dst,
                                Value src) {
  auto dstType = dst.getType().cast<FIRRTLType>();
  auto srcType = src.getType().cast<FIRRTLType>();

  // If the types are the exact same we can just connect them.
  if (dstType == srcType) {
    // Strict connect does not allow uninferred widths.
    // TODO: enable strict connect.
    if (true || dstType.hasUninferredWidth())
      builder.create<ConnectOp>(dst, src);
    else
      builder.create<StrictConnectOp>(dst, src);
    return;
  }

  if (auto dstBundle = dstType.dyn_cast<BundleType>()) {
    // Connect all the bundle elements pairwise.
    auto numElements = dstBundle.getNumElements();
    // Check if we are trying to create an illegal connect - just create the
    // connect and let the verifier catch it.
    auto srcBundle = srcType.dyn_cast<BundleType>();
    if (!srcBundle || numElements != srcBundle.getNumElements()) {
      builder.create<ConnectOp>(dst, src);
      return;
    }
    for (size_t i = 0; i < numElements; ++i) {
      auto dstField = builder.create<SubfieldOp>(dst, i);
      auto srcField = builder.create<SubfieldOp>(src, i);
      if (dstBundle.getElement(i).isFlip)
        std::swap(dstField, srcField);
      emitConnect(builder, dstField, srcField);
    }
    return;
  }

  if (auto dstVector = dstType.dyn_cast<FVectorType>()) {
    // Connect all the vector elements pairwise.
    auto numElements = dstVector.getNumElements();
    // Check if we are trying to create an illegal connect - just create the
    // connect and let the verifier catch it.
    auto srcVector = srcType.dyn_cast<FVectorType>();
    if (!srcVector || numElements != srcVector.getNumElements()) {
      builder.create<ConnectOp>(dst, src);
      return;
    }
    for (size_t i = 0; i < numElements; ++i) {
      auto dstField = builder.create<SubindexOp>(dst, i);
      auto srcField = builder.create<SubindexOp>(src, i);
      emitConnect(builder, dstField, srcField);
    }
    return;
  }

  // Handle ground types with possibly uninferred widths.
  auto dstWidth = dstType.getBitWidthOrSentinel();
  auto srcWidth = srcType.getBitWidthOrSentinel();
  if (dstWidth < 0 || srcWidth < 0) {
    // If one of these types has an uninferred width, we connect them with a
    // regular connect operation.
    builder.create<ConnectOp>(dst, src);
    return;
  }

  // The source must be extended or truncated.
  if (dstWidth < srcWidth) {
    // firrtl.tail always returns uint even for sint operands.
    IntType tmpType = dstType.cast<IntType>();
    if (tmpType.isSigned())
      tmpType = UIntType::get(dstType.getContext(), dstWidth);
    src = builder.create<TailPrimOp>(tmpType, src, srcWidth - dstWidth);
    // Insert the cast back to signed if needed.
    if (tmpType != dstType)
      src = builder.create<AsSIntPrimOp>(dstType, src);
  } else if (srcWidth < dstWidth) {
    // Need to extend arg.
    src = builder.create<PadPrimOp>(src, dstWidth);
  }

  // Strict connect requires the types to be completely equal, including
  // connecting uint<1> to abstract reset types.
  if (dstType == srcType)
    builder.create<StrictConnectOp>(dst, src);
  else
    builder.create<ConnectOp>(dst, src);
}

IntegerAttr circt::firrtl::getIntAttr(Type type, const APInt &value) {
  auto intType = type.cast<IntType>();
  assert((!intType.hasWidth() ||
          (unsigned)intType.getWidthOrSentinel() == value.getBitWidth()) &&
         "value / type width mismatch");
  auto intSign =
      intType.isSigned() ? IntegerType::Signed : IntegerType::Unsigned;
  auto attrType =
      IntegerType::get(type.getContext(), value.getBitWidth(), intSign);
  return IntegerAttr::get(attrType, value);
}

/// Return an IntegerAttr filled with zeros for the specified FIRRTL integer
/// type. This handles both the known width and unknown width case.
IntegerAttr circt::firrtl::getIntZerosAttr(Type type) {
  int32_t width = abs(type.cast<IntType>().getWidthOrSentinel());
  return getIntAttr(type, APInt(width, 0));
}
