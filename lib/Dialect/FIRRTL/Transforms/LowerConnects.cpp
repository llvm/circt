//===- LowerConnects.cpp - Lower Aggregate Duplex Connects ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After this pass, all data flows right to left in a connect.  There are no 
// connects of an aggregates which still have a flip.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Parallel.h"

using namespace circt;
using namespace firrtl;

/// Emit a connect between two values.
static void emitStrictConnect(ImplicitLocOpBuilder &builder, Value dst,
                                Value src) {
  auto dstFType = dst.getType().cast<FIRRTLType>();
  auto srcFType = src.getType().cast<FIRRTLType>();
  auto dstType = dstFType.dyn_cast<FIRRTLBaseType>();
  auto srcType = srcFType.dyn_cast<FIRRTLBaseType>();

  // Non-base types don't need special handling.
  if (!srcType || !dstType) {
    builder.create<RefConnectOp>(dst, src);
    return;
  }

  if (dstType == srcType && dstType.isPassive() && srcType.isPassive() && !dstType.hasUninferredWidth() && !srcType.hasUninferredWidth()) {
    builder.create<StrictConnectOp>(dst, src);
    return;
  }

  if (auto dstBundle = dstType.dyn_cast<BundleType>()) {
    // Connect all the bundle elements pairwise.
    auto numElements = dstBundle.getNumElements();
    // Connect verifier ensures both sides have the same shape
    for (size_t i = 0; i < numElements; ++i) {
      auto dstField = builder.create<SubfieldOp>(dst, i);
      auto srcField = builder.create<SubfieldOp>(src, i);
      if (dstBundle.getElement(i).isFlip)
        std::swap(dstField, srcField);
      emitStrictConnect(builder, dstField, srcField);
    }
    return;
  }

  if (auto dstVector = dstType.dyn_cast<FVectorType>()) {
    // Connect all the vector elements pairwise.
    auto numElements = dstVector.getNumElements();
    // Connect verifier ensures both sides have the same shape
    for (size_t i = 0; i < numElements; ++i) {
      auto dstField = builder.create<SubindexOp>(dst, i);
      auto srcField = builder.create<SubindexOp>(src, i);
      emitStrictConnect(builder, dstField, srcField);
    }
    return;
  }

  // Handle ground types with possibly uninferred widths.
  auto dstWidth = dstType.getBitWidthOrSentinel();
  auto srcWidth = srcType.getBitWidthOrSentinel();
  if (dstWidth < 0 || srcWidth < 0) {
    // If one of these types has an uninferred width, we connect them with a
    // regular connect operation.
    llvm::errs() << "Uninferred connect\n";
    dst.dump();
    src.dump();
    auto a = builder.create<ConnectOp>(dst, src);
    a.dump();
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
    srcType = src.getType().cast<FIRRTLBaseType>();
  } else if (srcWidth < dstWidth) {
    // Need to extend arg.
    src = builder.create<PadPrimOp>(src, dstWidth);
    srcType = src.getType().cast<FIRRTLBaseType>();
  }

  // Deal with reset converting connects
  if (srcType.isResetType() && dstType.isResetType() && srcType != dstType) {
    src = builder.create<UninferredResetCastOp>(dstType, src);
    srcType = src.getType().cast<FIRRTLBaseType>();
  }

  if (dstType != srcType) {
    llvm::errs() << "Weird connect\n";
    dst.getType().dump();
    src.getType().dump();
    assert(0);
  }
  builder.create<StrictConnectOp>(dst, src);
}


static void lowerConnect(ConnectOp con) {
  ImplicitLocOpBuilder locBuilder(con.getLoc(), con);
    emitStrictConnect(locBuilder, con.getDest(), con.getSrc());
    con.erase();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerConnectsPass : public LowerFIRRTLConnectsBase<LowerConnectsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void LowerConnectsPass::runOnOperation() {
  FModuleOp mod = getOperation();

  // Record all operations in the circuit.
  for (auto con : llvm::make_early_inc_range(mod.getOps<ConnectOp>()))
      lowerConnect(con);

//  if (failed(result))
//    signalPassFailure();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass>
circt::firrtl::createLowerFIRRTLConnectsPass() {
  return std::make_unique<LowerConnectsPass>();
}
