//===- LowerFormalToHW.cpp - Formal Preparations --*- C++ -*---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Lower verif.formal to hw.module.
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;

namespace circt {
namespace verif {
#define GEN_PASS_DEF_LOWERFORMALTOHWPASS
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;
using namespace verif;

namespace {
struct LowerFormalToHWPass
    : verif::impl::LowerFormalToHWPassBase<LowerFormalToHWPass> {
  void runOnOperation() override;
};

static LogicalResult lowerFormalToHW(FormalOp op) {
  IRRewriter rewriter(op);

  // Create the ports for all the symbolic values
  SmallVector<hw::PortInfo> ports;
  for (auto symOp : op.getBody().front().getOps<verif::SymbolicValueOp>()) {
    ports.push_back(
        hw::PortInfo({{rewriter.getStringAttr("symbolic_value_" +
                                              std::to_string(ports.size())),
                       symOp.getType(), hw::ModulePort::Input}}));
  }

  auto moduleOp =
      rewriter.create<hw::HWModuleOp>(op.getLoc(), op.getNameAttr(), ports);

  rewriter.inlineBlockBefore(&op.getBody().front(),
                             &moduleOp.getBodyBlock()->front(),
                             op.getBody().getArguments());

  // Replace symbolic values with module arguments
  size_t i = 0;
  for (auto symOp : make_early_inc_range(
           moduleOp.getBodyBlock()->getOps<SymbolicValueOp>())) {
    rewriter.replaceAllUsesWith(symOp.getResult(),
                                moduleOp.getArgumentForInput(i));
    i++;
    rewriter.eraseOp(symOp);
  }

  rewriter.eraseOp(op);
  return success();
}
} // namespace

void LowerFormalToHWPass::runOnOperation() {
  bool changed = false;
  for (auto op :
       llvm::make_early_inc_range(getOperation().getOps<FormalOp>())) {
    if (failed(lowerFormalToHW(op)))
      return signalPassFailure();
    changed = true;
  }
  if (!changed)
    return markAllAnalysesPreserved();
}
