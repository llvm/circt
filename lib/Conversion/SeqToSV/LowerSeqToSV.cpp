//===- SeqToSV.cpp - Seq to SV lowering -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "seq-to-sv"

using namespace circt;
using namespace seq;

namespace {
struct LowerSeqToSVPass : public impl::LowerSeqToSVBase<LowerSeqToSVPass> {
  void runOnOperation() override;

  // Expose the pass constructor with pass options.
  using LowerSeqToSVBase<LowerSeqToSVPass>::LowerSeqToSVBase;

  // Make the pass parameters publicly visible.
  using LowerSeqToSVBase<LowerSeqToSVPass>::disableRegRandomization;
  using LowerSeqToSVBase<
      LowerSeqToSVPass>::addVivadoRAMAddressConflictSynthesisBugWorkaround;
  using LowerSeqToSVBase<LowerSeqToSVPass>::numSubaccessRestored;
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
circt::createLowerSeqToSVPass(const LowerSeqToSVOptions &options) {
  return std::make_unique<LowerSeqToSVPass>(options);
}

void LowerSeqToSVPass::runOnOperation() {
  // TODO
  LLVM_DEBUG(llvm::dbgs() << "Lower Seq to SV\n");
}
