//===- LowerSeqToSV.cpp - Seq to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq ops to SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SeqToSV.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace circt;

namespace {
#define GEN_PASS_DEF_LOWERSEQFIRRTLINITTOSV
#include "circt/Conversion/Passes.h.inc"

struct SeqFIRRTLInitToSVPass
    : public impl::LowerSeqFIRRTLInitToSVBase<SeqFIRRTLInitToSVPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void SeqFIRRTLInitToSVPass::runOnOperation() {
  ModuleOp top = getOperation();
  OpBuilder builder(top.getBody(), top.getBody()->begin());
  // FIXME: getOrCreate
  builder.create<sv::MacroDeclOp>(top.getLoc(), "RANDOM", nullptr, nullptr);
}

std::unique_ptr<Pass> circt::createLowerSeqFIRRTLInitToSV() {
  return std::make_unique<SeqFIRRTLInitToSVPass>();
}
