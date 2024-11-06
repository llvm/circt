//===- SimpleRegAlloc.cpp - RTG Register Allocation implementation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass allocates registers.  This throws out all the standard algorithms
// and essentially does a first-fit.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_SIMPLEREGALLOCPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;

#define DEBUG_TYPE "rtg-simpleregalloc"

namespace {
class SimpleRegAllocPass : public circt::rtg::impl::SimpleRegAllocPassBase<SimpleRegAllocPass> {
public:
  void runOnOperation() override;
};
} // end namespace


void SimpleRegAllocPass::runOnOperation() {
  auto moduleOp = getOperation();
  DenseMap<Dialect*, BitVector> usedRegs;

  if (moduleOp
          .walk([&](Operation *op) -> WalkResult {
            return TypeSwitch<Operation *, LogicalResult>(op)
                .Case<rtg::RegisterOpInterface>([&](auto op) {
                    if (auto r = op.getFixedReg(); r != ~0U) {
                        if (usedRegs[op->getDialect()][r]) {
                            op->emitError("register already in use [") << r << "]";
                            return failure();
                        }
                        usedRegs[op->getDialect()].set(r);
                        return success();
                    }
                    auto usableRegs = op.getAllowedRegs();
                    usableRegs.reset(usedRegs[op->getDialect()]);
                    auto reg = usableRegs.find_first();
                    if (reg == -1) {
                        op->emitError("no available registers");
                        return failure();
                    }
                    op.setFixedReg(reg);
                    return success();
                })
                .Default([](auto op) { return success(); });
          })
          .wasInterrupted())
    signalPassFailure();
}
