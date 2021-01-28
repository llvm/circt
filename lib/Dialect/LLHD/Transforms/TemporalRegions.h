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

#include "mlir/IR/Operation.h"

namespace circt {
namespace llhd {

struct TemporalRegionAnalysis {
  using BlockMapT = llvm::DenseMap<mlir::Block *, int>;
  using TRMapT = llvm::DenseMap<int, llvm::SmallVector<mlir::Block *, 8>>;

  explicit TemporalRegionAnalysis(mlir::Operation *op) { recalculate(op); }

  void recalculate(mlir::Operation *);

  unsigned getNumTemporalRegions() { return numTRs; }

  int getBlockTR(mlir::Block *);
  llvm::SmallVector<mlir::Block *, 8> getBlocksInTR(int);

  llvm::SmallVector<mlir::Block *, 8> getExitingBlocksInTR(int);
  mlir::Block *getTREntryBlock(int);
  bool hasSingleExitBlock(int tr) {
    return getExitingBlocksInTR(tr).size() == 1;
  }
  bool isOwnTRSuccessor(int tr) {
    auto succs = getTRSuccessors(tr);
    return std::find(succs.begin(), succs.end(), tr) != succs.end();
  }

  llvm::SmallVector<int, 8> getTRSuccessors(int);
  unsigned getNumTRSuccessors(int tr) { return getTRSuccessors(tr).size(); }
  unsigned numBlocksInTR(int tr) { return getBlocksInTR(tr).size(); }

private:
  unsigned numTRs;
  BlockMapT blockMap;
  TRMapT trMap;
};

} // namespace llhd
} // namespace circt

#endif // DIALECT_LLHD_TRANSFORMS_TEMPORALREGIONS_H
