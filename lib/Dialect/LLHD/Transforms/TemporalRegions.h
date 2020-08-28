//===- TemporalRegions.h - LLHD temporal regions analysis -------*- C++ -*-===//
//
// This file defines an Analysis for Behavioral LLHD to find the temporal
// regions of an LLHD process
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LLHD_TRANSFORMS_TEMPORALREGIONS_H
#define DIALECT_LLHD_TRANSFORMS_TEMPORALREGIONS_H

#include "mlir/IR/Operation.h"

namespace mlir {
namespace llhd {

struct TemporalRegionAnalysis {
  using BlockMapT = DenseMap<Block *, int>;
  using TRMapT = DenseMap<int, SmallVector<Block *, 8>>;

  explicit TemporalRegionAnalysis(Operation *op) { recalculate(op); }

  void recalculate(Operation *);

  unsigned getNumTemporalRegions() { return numTRs; }

  int getBlockTR(Block *);
  SmallVector<Block *, 8> getBlocksInTR(int);

  SmallVector<Block *, 8> getExitingBlocksInTR(int);
  Block *getTREntryBlock(int);
  bool hasSingleExitBlock(int tr) {
    return getExitingBlocksInTR(tr).size() == 1;
  }
  bool isOwnTRSuccessor(int tr) {
    auto succs = getTRSuccessors(tr);
    return std::find(succs.begin(), succs.end(), tr) != succs.end();
  }

  SmallVector<int, 8> getTRSuccessors(int);
  unsigned getNumTRSuccessors(int tr) { return getTRSuccessors(tr).size(); }
  unsigned numBlocksInTR(int tr) { return getBlocksInTR(tr).size(); }

private:
  unsigned numTRs;
  BlockMapT blockMap;
  TRMapT trMap;
};

} // namespace llhd
} // namespace mlir

#endif // DIALECT_LLHD_TRANSFORMS_TEMPORALREGIONS_H
