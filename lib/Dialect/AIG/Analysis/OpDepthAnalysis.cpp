//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the op depth (level) analysis.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/Analysis/OpDepthAnalysis.h"

using namespace circt;
using namespace aig;
using namespace analysis;

OpDepthAnalysis::OpDepthAnalysis(hw::HWModuleOp moduleOp,
                                 mlir::AnalysisManager &am)
    : module(moduleOp) {}

SmallVector<AndInverterOp> OpDepthAnalysis::getPOs() {
  SmallVector<AndInverterOp> po;
  for (auto op : module.getOps<AndInverterOp>()) {
    bool isPO = false;
    for (auto *user : op->getUsers()) {
      if (!isa<AndInverterOp>(user)) {
        isPO = true;
        break;
      }
    }
    if (isPO)
      po.push_back(op);
  }
  return po;
}

size_t OpDepthAnalysis::updateLevel(AndInverterOp op, bool isRoot) {
  if (!isRoot) {
    if (auto it = opDepths.find(op); it != opDepths.end()) {
      return it->second;
    }
  }

  /// PI is level 0, so the minimum level of an AndInverterOp is 1
  size_t maxDepth = 1;
  for (auto fanin : op.getOperands()) {
    if (auto faninOp = fanin.getDefiningOp<AndInverterOp>()) {
      size_t faninDepth = updateLevel(faninOp);
      maxDepth = std::max(maxDepth, faninDepth + 1);
    }
  }
  opDepths[op] = maxDepth;
  return maxDepth;
}