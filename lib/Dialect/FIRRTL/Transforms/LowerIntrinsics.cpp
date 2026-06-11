//===- LowerIntrinsics.cpp - Lower Intrinsics -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerIntrinsics pass.  This pass processes FIRRTL
// generic intrinsic operations and rewrites to their implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLIntrinsics.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERINTRINSICS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntrinsicsPass
    : public circt::firrtl::impl::LowerIntrinsicsBase<LowerIntrinsicsPass> {
  LogicalResult initialize(MLIRContext *context) override;
  void runOnOperation() override;

  std::shared_ptr<IntrinsicLowerings> lowering;
};
} // namespace

/// Build the immutable converter set once, shared across module invocations.
LogicalResult LowerIntrinsicsPass::initialize(MLIRContext *context) {
  // `IntrinsicLowerings` holds only the converter registry -- no mutable
  // per-invocation state. Per-module staging data is passed through
  // `lower(mod, allowUnknown, ctx)` on each `runOnOperation` invocation,
  // making this safe to share across parallel module passes.
  this->lowering = std::make_shared<IntrinsicLowerings>(context);
  IntrinsicLoweringInterfaceCollection collection(context);
  collection.populateIntrinsicLowerings(*this->lowering);
  return success();
}

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  auto mod = getOperation();

  // Phase 1: stage debug-intrinsic data. `dbg.enumdef` ops are written into
  // the IR (they have semantics downstream); `circt_debug_subfield` leaves
  // are gathered into a stack-local side-channel list that is passed to
  // `lower()` via `IntrinsicConvertContext`. Keeping the staging data on
  // this function's stack -- never on the shared `lowering` object -- is what
  // makes the pass safe to run concurrently across modules.
  // (firrtl.module is a Graph region; block-start insertion is cosmetic.)
  OpBuilder builder = OpBuilder::atBlockBegin(mod.getBodyBlock());
  firrtl::DebugLeafList debugLeaves;
  llvm::StringMap<mlir::Value> enumDefByFqn;
  if (mlir::failed(
          firrtl::liftDebugIntrinsics(mod, builder, debugLeaves, enumDefByFqn)))
    return signalPassFailure();

  // Phase 2: run intrinsic lowerings with a stack-local context.
  // `seenVarNames` accumulates names of dbg.variable ops as they are emitted;
  // CirctDebugVarConverter uses it for O(1) duplicate detection instead of
  // walking the growing IR on every invocation.
  llvm::StringSet<> seenVarNames;
  firrtl::IntrinsicConvertContext convCtx{&debugLeaves, &enumDefByFqn,
                                          &seenVarNames};
  auto result = lowering->lower(mod, /*allowUnknownIntrinsics=*/false, convCtx);
  if (failed(result))
    return signalPassFailure();

  numConverted += *result;

  if (*result == 0)
    markAllAnalysesPreserved();
}
