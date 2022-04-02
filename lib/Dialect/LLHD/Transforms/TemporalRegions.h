//===- TemporalRegions.h - LLHD temporal regions analysis -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an Analysis for Behavioral LLHD to find the temporal
// regions of an LLHD process
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LLHD_TRANSFORMS_TEMPORALREGIONS_H
#define DIALECT_LLHD_TRANSFORMS_TEMPORALREGIONS_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/Operation.h"

namespace circt {
namespace llhd {

class TemporalRegionAnalysis {
public:
  explicit TemporalRegionAnalysis(Operation *op) { recalculate(op); }

public:
  void recalculate(Operation *);

  int getBlockTR(Block *);
  SmallVector<Block *, 8> getBlocksInTR(int);
  SmallVector<Block *, 8> getExitingBlocksInTR(int);
  SmallVector<int, 8> getTRSuccessors(int);
  Block *getTREntryBlock(int);

public:
  bool hasSingleExitBlock(int tr) {
    return getExitingBlocksInTR(tr).size() == 1;
  }

  bool isOwnTRSuccessor(int tr) {
    auto succs = getTRSuccessors(tr);
    return std::find(succs.begin(), succs.end(), tr) != succs.end();
  }

  unsigned getNumTemporalRegions() { return numTRs; }
  unsigned getNumTRSuccessors(int tr) { return getTRSuccessors(tr).size(); }
  unsigned numBlocksInTR(int tr) { return getBlocksInTR(tr).size(); }

public:
  using BlockMapT = DenseMap<Block *, int>;
  using TRMapT = DenseMap<int, SmallVector<Block *, 8>>;

private:
  unsigned numTRs;
  BlockMapT blockMap;
  TRMapT trMap;
};

} // namespace llhd
} // namespace circt

#endif // DIALECT_LLHD_TRANSFORMS_TEMPORALREGIONS_H
