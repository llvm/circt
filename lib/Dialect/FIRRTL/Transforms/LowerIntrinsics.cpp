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
  // The converter set is immutable, so it is safe to share across the parallel
  // per-module invocations; per-module state rides in `lower`'s `ctx`.
  this->lowering = std::make_shared<IntrinsicLowerings>(context);
  IntrinsicLoweringInterfaceCollection collection(context);
  collection.populateIntrinsicLowerings(*this->lowering);
  return success();
}

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  auto mod = getOperation();

  // Phase 1: stage enumdef data, subfield leaves, and the named-declaration
  // index into stack-local side channels (kept off the shared `lowering` for
  // concurrency; see `lower`).
  firrtl::DebugLeafMap debugLeaves;
  llvm::StringMap<firrtl::EnumDefData> enumDefByFqn;
  firrtl::NamedDeclIndex namedDecls;
  if (mlir::failed(firrtl::liftDebugIntrinsics(mod, debugLeaves, enumDefByFqn,
                                               namedDecls)))
    return signalPassFailure();

  // Phase 2: lower with a stack-local context. `seenVarNames` collects
  // dbg.variable names as they are emitted; CirctDebugVarConverter uses it for
  // O(1) duplicate detection instead of walking the growing IR each time.
  llvm::StringSet<> seenVarNames;
  firrtl::IntrinsicConvertContext convCtx{&debugLeaves, &enumDefByFqn,
                                          &namedDecls, &seenVarNames};
  auto result = lowering->lower(mod, /*allowUnknownIntrinsics=*/false, convCtx);
  if (failed(result))
    return signalPassFailure();

  numConverted += *result;

  if (*result == 0)
    markAllAnalysesPreserved();
}
