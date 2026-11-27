//===- UndrivenToSymbolic.cpp - Replace undriven signals with symbols -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-undriven-to-symbolic"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_UNDRIVENTOSYMBOLICPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;

/// Check if a signal is only ever probed or projected into, but not driven.
static bool
isOnlyProbedOrProjected(SignalOp sigOp,
                        SmallVectorImpl<ProbeOp> &collectedProbes,
                        SmallVectorImpl<Operation *> &collectedProjections) {
  SmallVector<Value> worklist = {sigOp.getResult()};
  while (!worklist.empty()) {
    auto value = worklist.pop_back_val();
    for (auto *user : value.getUsers()) {
      if (auto probeOp = dyn_cast<ProbeOp>(user)) {
        collectedProbes.push_back(probeOp);
      } else if (isa<SigExtractOp, SigArrayGetOp, SigArraySliceOp,
                     SigStructExtractOp>(user)) {
        collectedProjections.push_back(user);
        worklist.push_back(user->getResult(0));
      } else {
        return false;
      }
    }
  }

  return true;
}

/// Create the equivalent `hw` op for a projection, with `newOperand` as its
/// input.
static Value buildHWProjection(OpBuilder &builder, Operation *op,
                               Value newOperand) {
  return TypeSwitch<Operation *, Value>(op)
      .Case([&](SigArrayGetOp op) {
        return builder.createOrFold<hw::ArrayGetOp>(op.getLoc(), newOperand,
                                                    op.getIndex());
      })
      .Case([&](SigArraySliceOp op) {
        return builder.createOrFold<hw::ArraySliceOp>(
            op.getLoc(), op.getType().getNestedType(), newOperand,
            op.getLowIndex());
      })
      .Case([&](SigStructExtractOp op) {
        // SigStructExtractOp is used for both struct and union member access.
        if (isa<hw::UnionType>(newOperand.getType()))
          return builder.createOrFold<hw::UnionExtractOp>(
              op.getLoc(), newOperand, op.getFieldAttr());
        return builder.createOrFold<hw::StructExtractOp>(
            op.getLoc(), newOperand, op.getFieldAttr());
      })
      .Case([&](SigExtractOp op) {
        auto type = op.getType().getNestedType();
        return comb::createDynamicExtract(builder, op.getLoc(), newOperand,
                                          op.getLowBit(),
                                          type.getIntOrFloatBitWidth());
      })
      .Default([](Operation *) -> Value {
        llvm_unreachable("expected a projection op");
      });
}

namespace {
struct UndrivenToSymbolicPass
    : public llhd::impl::UndrivenToSymbolicPassBase<UndrivenToSymbolicPass> {
protected:
  void runOnOperation() override;
};
} // namespace

void UndrivenToSymbolicPass::runOnOperation() {
  SmallVector<SignalOp> sigOps;
  getOperation().walk([&](SignalOp sigOp) {
    // NOTE: verif.symbolic_value can only live in a hw.module, so for now
    // we only promote signals that live directly in the hw.module.
    if (isa<hw::HWModuleOp>(sigOp->getParentOp()))
      sigOps.push_back(sigOp);
  });

  for (auto sigOp : sigOps) {
    // Ignore signals with an initial value.
    if (sigOp.getInit())
      continue;
    SmallVector<ProbeOp> probes;
    SmallVector<Operation *> projections;
    // Get only signals that are only probed (through projections).
    if (!isOnlyProbedOrProjected(sigOp, probes, projections))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "Replacing undriven " << sigOp << "\n");

    OpBuilder builder(sigOp);
    Value symbol = verif::SymbolicValueOp::create(
        builder, sigOp.getLoc(), sigOp.getType().getNestedType(),
        sigOp.getNameAttr());
    DenseMap<Value, Value> substitutions;
    substitutions[sigOp] = symbol;
    for (auto *op : projections) {
      builder.setInsertionPoint(op);
      auto newOperand = substitutions.lookup(op->getOperand(0));
      auto newVal = buildHWProjection(builder, op, newOperand);
      substitutions[op->getResult(0)] = newVal;
    }
    for (auto probeOp : probes) {
      auto inputSignal = probeOp.getSignal();
      probeOp.replaceAllUsesWith(substitutions.lookup(inputSignal));
      probeOp.erase();
    }
    for (auto *op : llvm::reverse(projections))
      op->erase();
    sigOp.erase();
  }
}
