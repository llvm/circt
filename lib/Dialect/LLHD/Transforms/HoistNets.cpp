//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CaptureUtils.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-hoist-nets"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_HOISTNETSPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;

namespace {
struct HoistNetsPass : public llhd::impl::HoistNetsPassBase<HoistNetsPass> {
protected:
  void runOnOperation() override;

private:
  void hoistProbes(Region *region);
};
} // namespace

void HoistNetsPass::hoistProbes(Region *region) {
  Operation *parentOp = region->getParentOp();
  DominanceInfo dominance(parentOp);

  // Collect any probes of nets defined outside the region
  SmallVector<ProbeNetOp> probes;
  region->walk([&](ProbeNetOp probe) {
    if (dominance.properlyDominates(probe.getNet(), parentOp))
      probes.push_back(probe);
  });
  if (probes.empty())
    return;

  // ProcessOps can contain waits - if a probe is used across a wait we can't
  // just replace it with the hoisted version from the wrong timestep. Instead
  // we have to capture the value across the waits as a block argument
  if (auto processOp = dyn_cast<ProcessOp>(parentOp)) {
    SmallVector<WaitOp> waitOps;
    for (auto &block : *region)
      if (auto waitOp = dyn_cast<WaitOp>(block.getTerminator()))
        waitOps.push_back(waitOp);
    if (!waitOps.empty()) {
      Liveness liveness(processOp);
      for (auto probe : probes)
        captureAcrossWait(probe, waitOps, liveness, dominance);
    }
  }

  OpBuilder builder(parentOp);
  for (auto probe : probes) {
    auto *clone = builder.clone(*probe);
    probe.replaceAllUsesWith(clone);
    probe.erase();
  }
}

void HoistNetsPass::runOnOperation() {
  SmallVector<Region *> regions;
  getOperation()->walk([&](Operation *op) {
    if (isa<ProcessOp, FinalOp, CombinationalOp, scf::IfOp>(op))
      for (auto &region : op->getRegions())
        if (!region.empty())
          regions.push_back(&region);
  });
  for (auto *region : regions) {
    hoistProbes(region);
  }
}
