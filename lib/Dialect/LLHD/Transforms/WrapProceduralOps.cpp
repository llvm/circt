//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_WRAPPROCEDURALOPSPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::llhd;

namespace {
struct WrapProceduralOpsPass
    : public llhd::impl::WrapProceduralOpsPassBase<WrapProceduralOpsPass> {
  void runOnOperation() override;
};
} // namespace

void WrapProceduralOpsPass::runOnOperation() {
  for (auto &op : llvm::make_early_inc_range(getOperation().getOps())) {
    if (!isa<scf::SCFDialect>(op.getDialect()) && !isa<func::CallOp>(op))
      continue;
    auto builder = OpBuilder(&op);
    auto wrapperOp =
        builder.create<llhd::CombinationalOp>(op.getLoc(), op.getResultTypes());
    op.replaceAllUsesWith(wrapperOp);
    builder.createBlock(&wrapperOp.getBody());
    auto yieldOp = builder.create<llhd::YieldOp>(op.getLoc(), op.getResults());
    op.moveBefore(yieldOp);
    ++numOpsWrapped;
  }
}
