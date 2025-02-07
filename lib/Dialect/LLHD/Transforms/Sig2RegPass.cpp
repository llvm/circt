//===- Sig2RegPass.cpp - Implement the Sig2Reg Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement Pass to promote LLHD signals to SSA values.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-sig2reg"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_SIG2REG
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;

namespace {
struct Sig2RegPass : public circt::llhd::impl::Sig2RegBase<Sig2RegPass> {
  void runOnOperation() override;
};
} // namespace

static LogicalResult promote(llhd::SignalOp sigOp) {
  SmallVector<llhd::PrbOp> probes;
  llhd::DrvOp driveOp;
  for (auto *user : sigOp.getResult().getUsers()) {
    if (user->getBlock() != sigOp->getBlock()) {
      LLVM_DEBUG(
          { llvm::dbgs() << "Promotion failed: user in other block\n"; });
      return failure();
    }

    if (auto prbOp = dyn_cast<llhd::PrbOp>(user)) {
      probes.push_back(prbOp);
      continue;
    }

    if (auto drvOp = dyn_cast<llhd::DrvOp>(user)) {
      if (driveOp) {
        LLVM_DEBUG({ llvm::dbgs() << "Promotion failed: multiple drivers\n"; });
        return failure();
      }

      if (drvOp.getEnable()) {
        LLVM_DEBUG(
            { llvm::dbgs() << "Promotion failed: conditional driver\n"; });
        return failure();
      }

      driveOp = drvOp;
      continue;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Promotion failed: user that is not a probe or drive: "
                   << *user << "\n";
    });
    return failure();
  }

  Value replacement;
  if (driveOp) {
    auto timeOp = driveOp.getTime().getDefiningOp<llhd::ConstantTimeOp>();
    if (!timeOp)
      return failure();

    OpBuilder builder(driveOp);
    if (timeOp.getValue().getTime() == 0 && timeOp.getValue().getDelta() == 0)
      replacement = driveOp.getValue();
    else
      replacement = builder.create<llhd::DelayOp>(
          driveOp.getLoc(), driveOp.getValue(), timeOp.getValue());
  } else {
    replacement = sigOp.getInit();
  }

  for (auto prb : probes) {
    prb.getResult().replaceAllUsesWith(replacement);
    prb.erase();
  }

  if (driveOp)
    driveOp.erase();

  return success();
}

void Sig2RegPass::runOnOperation() {
  hw::HWModuleOp moduleOp = getOperation();

  for (auto sigOp :
       llvm::make_early_inc_range(moduleOp.getOps<llhd::SignalOp>())) {
    LLVM_DEBUG(
        { llvm::dbgs() << "\nAttempting to promote " << sigOp << "\n"; });
    if (failed(promote(sigOp)))
      continue;

    LLVM_DEBUG({ llvm::dbgs() << "Successfully promoted!\n"; });
    sigOp.erase();
  }
}
