//===- OpDepthAnalysis.h - operation depth analyses -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines AIG operation depth analysis.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_ANALYSIS_OPDEPTH_ANALYSIS_H
#define CIRCT_ANALYSIS_OPDEPTH_ANALYSIS_H

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/HW/HWOps.h"

namespace mlir {
class AnalysisManager;
} // namespace mlir
namespace circt {
namespace aig {
namespace analysis {

class OpDepthAnalysis {
public:
  OpDepthAnalysis(hw::HWModuleOp moduleOp, mlir::AnalysisManager &am);

  /// Get the depth of operations of a specific name
  size_t getOpDepth(AndInverterOp op) const { return opDepths.at(op); }

  const DenseMap<AndInverterOp, size_t> &getOpDepthMap() const {
    return opDepths;
  }

  size_t updateLevel(AndInverterOp op, bool isRoot = false);

  SmallVector<AndInverterOp> getPOs();

private:
  DenseMap<AndInverterOp, size_t> opDepths;
  hw::HWModuleOp module;
};

} // namespace analysis
} // namespace aig
} // namespace circt

#endif // CIRCT_ANALYSIS_OPDEPTH_ANALYSIS_H
