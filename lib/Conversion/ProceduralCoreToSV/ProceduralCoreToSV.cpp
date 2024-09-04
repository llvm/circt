//===- ProceduralCoreToSV.cpp - Procedural Core To SV Conversion Pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower procedural core dialect (HW/Sim) operations to the SV dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ProceduralCoreToSV.h"

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Sim/SimDialect.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

namespace circt {
#define GEN_PASS_DEF_PROCEDURALCORETOSV
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace mlir;

static sv::EventControl hwToSvEventControl(hw::EventControl ec) {
  switch (ec) {
  case hw::EventControl::AtPosEdge:
    return sv::EventControl::AtPosEdge;
  case hw::EventControl::AtNegEdge:
    return sv::EventControl::AtNegEdge;
  case hw::EventControl::AtEdge:
    return sv::EventControl::AtEdge;
  }
  llvm_unreachable("Unknown event control kind");
}

namespace {

struct ProceduralOpRewriter : public RewriterBase {
  ProceduralOpRewriter(MLIRContext *ctxt) : RewriterBase::RewriterBase(ctxt) {}
};

struct ProceduralCoreToSVPass
    : public circt::impl::ProceduralCoreToSVBase<ProceduralCoreToSVPass> {
  ProceduralCoreToSVPass() = default;
  void runOnOperation() override;
};
} // namespace

void ProceduralCoreToSVPass::runOnOperation() {
  hw::HWModuleOp theModule = getOperation();

  ProceduralOpRewriter rewriter(theModule.getContext());

  theModule.walk<mlir::WalkOrder::PreOrder>([&](hw::TriggeredOp triggeredOp)
                                                -> WalkResult {
    // Create an AlwaysOp, move the body over and remove the TriggeredOp
    rewriter.setInsertionPoint(triggeredOp);
    auto alwaysOp = rewriter.create<sv::AlwaysOp>(
        triggeredOp.getLoc(),
        ArrayRef<sv::EventControl>{hwToSvEventControl(triggeredOp.getEvent())},
        ArrayRef<Value>{triggeredOp.getTrigger()});
    rewriter.mergeBlocks(triggeredOp.getBodyBlock(), alwaysOp.getBodyBlock(),
                         triggeredOp.getInputs());
    rewriter.eraseOp(triggeredOp);
    // Don't recurse into the body.
    return WalkResult::skip();
  });
}
