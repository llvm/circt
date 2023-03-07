//===- MergeConnections.cpp - Merge expanded connections --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass merges expanded connections into one connection.
// LowerTypes fully expands aggregate connections even when semantically
// not necessary to expand because it is required for ExpandWhen.
//
// More specifically this pass folds the following patterns:
//   %dest(0) <= v0
//   %dest(1) <= v1
//   ...
//   %dest(n) <= vn
// into
//   %dest <= {vn, .., v1, v0}
// Also if v0, v1, .., vn are subfield op like %a(0), %a(1), ..., a(n), then we
// merge entire connections into %dest <= %a.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-register-optimizer"

using namespace circt;
using namespace firrtl;

// Instantiated for RegOp and RegResetOp
template <typename T>
static bool canErase(T op) {
  if (hasDontTouch(op.getResult()) ||
      (op.getAnnotationsAttr() && !op.getAnnotationsAttr().empty()))
    return false;
  return true;
}

namespace {

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

struct RegisterOptimizerPass
    : public RegisterOptimizerBase<RegisterOptimizerPass> {
  void runOnOperation() override;
  bool checkRegReset(RegResetOp reg, SmallVector<Operation *> &toErase);
  bool checkReg(RegOp reg, SmallVector<Operation *> &toErase);
};

} // namespace

bool RegisterOptimizerPass::checkReg(RegOp reg,
                                     SmallVector<Operation *> &toErase) {
  if (!canErase(reg))
    return false;
  auto con = getSingleConnectUserOf(reg.getResult());
  if (!con)
    return false;

  // Register is only written by itself, replace with invalid.
  if (con.getSrc() == reg) {
    auto inv =
        OpBuilder(reg).create<InvalidValueOp>(reg.getLoc(), reg.getType());
    reg.replaceAllUsesWith(inv.getResult());
    reg.erase();
    toErase.push_back(con);
    return true;
  }
  return false;
}

bool RegisterOptimizerPass::checkRegReset(RegResetOp reg,
                                          SmallVector<Operation *> &toErase) {
  if (!canErase(reg))
    return false;
  auto con = getSingleConnectUserOf(reg.getResult());
  if (!con)
    return false;

  // Register is only written by itself, and reset with a constant.
  if (con.getSrc() == reg && isConstant(reg.getResetValue())) {
    // constant obviously dominates the register.
    reg.replaceAllUsesWith(reg.getResetValue());
    reg.erase();
    toErase.push_back(con);
    return true;
  }
  // Register is only written by a constant, and reset with the same constant.
  if (con.getSrc() == reg.getResetValue() && isConstant(reg.getResetValue())) {
    // constant obviously dominates the register.
    reg.replaceAllUsesWith(reg.getResetValue());
    reg.erase();
    toErase.push_back(con);
    return true;
  }

  return false;
}

void RegisterOptimizerPass::runOnOperation() {
  auto mod = getOperation();
  LLVM_DEBUG(llvm::dbgs() << "===----- Running RegisterOptimizer "
                             "--------------------------------------===\n"
                          << "Module: '" << mod.getName() << "'\n";);

  bool changed = false;
  SmallVector<Operation *> toErase;
  for (auto &op : llvm::make_early_inc_range(*mod.getBodyBlock())) {
    if (auto reg = dyn_cast<RegResetOp>(&op))
      changed |= checkRegReset(reg, toErase);
    else if (auto reg = dyn_cast<RegOp>(&op))
      changed |= checkReg(reg, toErase);
  }
  for (auto *op : toErase)
    op->erase();

  if (!changed)
    return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createRegisterOptimizerPass() {
  return std::make_unique<RegisterOptimizerPass>();
}
