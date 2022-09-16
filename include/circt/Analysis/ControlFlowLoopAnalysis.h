//===- ControlFlowLoopAnalysis.h - CF Loop Analysis -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform loop analysis on
// structures expressed as a CFG
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_CONTROL_FLOW_LOOP_ANALYSIS_H
#define CIRCT_ANALYSIS_CONTROL_FLOW_LOOP_ANALYSIS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallSet.h"

/// TODO can we reuse parts of Polygeist's implementation?
namespace circt {
namespace analysis {

/// Container that holds information about a cfg loop.
struct LoopInfo {
  SmallPtrSet<Block *, 2> loopLatches;
  SmallPtrSet<Block *, 4> inLoop;
  SmallPtrSet<Block *, 2> exitBlocks;
  Block *loopHeader;
};

struct ControlFlowLoopAnalysis {
  // Construct the analysis from a FuncOp.
  ControlFlowLoopAnalysis(Region &region);
  LogicalResult analyzeRegion();

  bool isLoopHeader(Block *b);
  bool isLoopElement(Block *b);
  LoopInfo *getLoopInfoForHeader(Block *b);
  LoopInfo *getLoopInfo(Block *b);

  SmallVector<LoopInfo> topLevelLoops;

private:
  bool hasBackedge(Block *);
  LogicalResult collectLoopInfo(Block *entry, LoopInfo &loopInfo);

private:
  Region &region;
  mlir::DominanceInfo domInfo;
};

} // namespace analysis
} // namespace circt

#endif // CIRCT_ANALYSIS_CONTROL_FLOW_LOOP_ANALYSIS_H
