//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Promote LLHD signals to SSA values.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/LLHDOps.h"
#include "circt/Dialect/LLHD/LLHDPasses.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llhd-sig2reg"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_SIG2REGPASS
#include "circt/Dialect/LLHD/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace llhd;

namespace {
struct Sig2RegPass : public llhd::impl::Sig2RegPassBase<Sig2RegPass> {
  void runOnOperation() override;
  void promote(SignalOp op);
  SmallVector<SignalOp> signalsToErase;
};
} // namespace

void Sig2RegPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Promoting signals to values in "
                          << getOperation().getModuleName() << "\n");
  for (auto op : getOperation().getOps<SignalOp>())
    promote(op);
  for (auto op : signalsToErase)
    op.erase();
  signalsToErase.clear();
}

void Sig2RegPass::promote(SignalOp op) {
  // Find the single drive, and check if there are any probes or multiple drives
  // of this signal. Bail on unknown users.
  DriveLikeOp drive;
  bool multipleDrives = false;
  bool anyProbe = false;
  for (auto *user : op->getUsers()) {
    if (isa<ProbeOp>(user)) {
      anyProbe = true;
      continue;
    }

    if (auto driveOp = dyn_cast<DriveLikeOp>(user)) {
      if (driveOp->getBlock() != op->getBlock()) {
        LLVM_DEBUG(llvm::dbgs() << "- Skipping " << op
                                << " (driven from different block)\n");
        ++numSkipped;
        return;
      }
      if (drive)
        multipleDrives = true;
      drive = driveOp;
      continue;
    }

    // Unknown user.
    LLVM_DEBUG(llvm::dbgs() << "- Skipping " << op << " (unsupported user "
                            << user->getName() << ")\n");
    ++numSkipped;
    return;
  }

  // Remove unprobed signals.
  if (!anyProbe) {
    LLVM_DEBUG(llvm::dbgs() << "- Removing unprobed " << op << "\n");
    ++numUnused;
    SmallPtrSet<Operation *, 8> erased;
    for (auto *user : llvm::make_early_inc_range(op->getUsers()))
      if (erased.insert(user).second)
        user->erase();
    signalsToErase.push_back(op);
    return;
  }

  // If there are multiple drives, we cannot come up with a clean replacement
  // value.
  if (multipleDrives) {
    LLVM_DEBUG(llvm::dbgs() << "- Skipping " << op << " (multiple drives)\n");
    ++numSkipped;
    return;
  }

  // We can't promote conditional drives.
  if (drive && drive.getEnable()) {
    LLVM_DEBUG(llvm::dbgs() << "- Skipping " << op << " (conditional drive)\n");
    ++numSkipped;
    return;
  }

  // Determine the value to replace the signal with.
  Value value;
  if (drive) {
    value = drive.getValue();
    if (drive.getTime()) {
      if (op.getInit()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Skipping " << op
                   << " (delayed drive to signal with initializer)\n");
        ++numSkipped;
        return;
      }
      TimeAttr delay;
      if (!matchPattern(drive.getTime(), m_Constant(&delay))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "- Skipping " << op << " (non-constant drive delay)\n");
        ++numSkipped;
        return;
      }
      OpBuilder builder(drive);
      value = DelayOp::create(builder, drive.getLoc(), value, delay);
    }
  } else {
    value = op.getInit();
    if (!value) {
      LLVM_DEBUG(llvm::dbgs()
                 << "- Skipping " << op << " (undriven with no initializer)\n");
      ++numSkipped;
      return;
    }

    // The signal captures its initial value once. Only a constant initializer
    // is guaranteed to equal the signal's value at all times.
    if (!matchPattern(value, m_Constant())) {
      LLVM_DEBUG(llvm::dbgs() << "- Skipping " << op
                              << " (undriven with non-constant initializer)\n");
      ++numSkipped;
      return;
    }
  }

  // Replace all probes with this value, and remove all users, which are only
  // probes and drives at this point.
  LLVM_DEBUG({
    llvm::dbgs() << "- Replacing " << op << " with ";
    value.printAsOperand(llvm::dbgs(), OpPrintingFlags());
    llvm::dbgs() << "\n";
  });
  SmallPtrSet<Operation *, 8> erased;
  for (auto *user : llvm::make_early_inc_range(op->getUsers())) {
    if (auto probeOp = dyn_cast<ProbeOp>(user))
      probeOp.replaceAllUsesWith(value);
    if (erased.insert(user).second)
      user->erase();
  }
  signalsToErase.push_back(op);
  ++numPromoted;
}
