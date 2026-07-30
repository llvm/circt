//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FullReset pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_FULLRESET
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct FullResetPass
    : public circt::firrtl::impl::FullResetBase<FullResetPass> {
  using FullResetBase::FullResetBase;

  void runOnOperation() override {
    auto &ig = getAnalysis<InstanceGraph>();
    auto &instanceInfo = getAnalysis<InstanceInfo>();
    if (failed(runFullReset(getOperation(), ig, instanceInfo,
                            /*convertAsyncDomainMems=*/true)))
      return signalPassFailure();
    markAnalysesPreserved<InstanceGraph, InstanceInfo>();
  }
};
} // namespace
