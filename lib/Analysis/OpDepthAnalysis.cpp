//===- OpCountAnalysis.cpp - operation count analyses -----------*- C++ -*-===//
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

#include "circt/Analysis/OpDepthAnalysis.h"
#include "mlir/IR/Operation.h"

using namespace circt;
using namespace aig;
using namespace analysis;

OpDepthAnalysis::OpDepthAnalysis(hw::HWModuleOp moduleOp,
                                 mlir::AnalysisManager &am)
    : module(moduleOp) {
  // llvm::dbgs() << "OpDepthAnalysis Init\n";
  // updateAllLevel();
}

SmallVector<AndInverterOp> OpDepthAnalysis::getPOs() {
  SmallVector<AndInverterOp> po;
  for (auto op : module.getOps<AndInverterOp>()) {
    bool isPO = true;
    for (auto *user : op->getUsers()) {
      if (isa<AndInverterOp>(user)) {
        isPO = false;
        break;
      }
    }
    if (isPO)
      po.push_back(op);
  }
  return po;
}

void OpDepthAnalysis::updateAllLevel() {
  auto po = getPOs();
  for (auto &op : po) {
    currDepth = std::max(currDepth, updateLevel(op));
  }

  for (auto &op : po) {
    if (auto it = opDepths.find(op);
        it != opDepths.end() && it->second == currDepth) {
      setCriticalPath(op);
      break;
    }
  }
}

size_t OpDepthAnalysis::updateLevel(AndInverterOp op, bool isRoot) {
  if (auto it = opDepths.find(op); !isRoot && it != opDepths.end()) {
    return it->second;
  }

  /// PI is level 0, so the minimum level of an AndInverterOp is 1
  size_t maxDepth = 1;
  for (auto fanin : op.getOperands()) {
    auto faninOp = fanin.getDefiningOp<AndInverterOp>();
    if (faninOp) {
      size_t faninDepth = updateLevel(faninOp);
      maxDepth = std::max(maxDepth, faninDepth + 1);
    }
  }
  opDepths[op] = maxDepth;
  return maxDepth;
}

void OpDepthAnalysis::setCriticalPath(AndInverterOp op) {
  size_t clevel = opDepths[op];
  for (auto fanin : op.getOperands()) {
    auto faninOp = fanin.getDefiningOp<AndInverterOp>();
    if (faninOp && opDepths[faninOp] + 1 == clevel) {
      setCriticalPath(faninOp);
      // break; // TODO: Should break when there are multiple critical paths?
    }
  }
}