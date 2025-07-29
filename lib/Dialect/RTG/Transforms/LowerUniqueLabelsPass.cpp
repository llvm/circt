//===- LowerUniqueLabelsPass.cpp - RTG LowerUniqueLabelsPass impl. --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers label_unique_decl operations to label_decl operations by
// creating a unique label string based on all the existing unique and
// non-unique label declarations in the module.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/PatternMatch.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_LOWERUNIQUELABELSPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Lower Unique Labels Pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerUniqueLabelsPass
    : public rtg::impl::LowerUniqueLabelsPassBase<LowerUniqueLabelsPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void LowerUniqueLabelsPass::runOnOperation() {
  Namespace labelNames;

  // Collect all the label names in a first iteration.
  getOperation()->walk([&](Operation *op) {
    if (auto labelDecl = dyn_cast<LabelDeclOp>(op))
      labelNames.add(labelDecl.getFormatString());
    else if (auto labelDecl = dyn_cast<LabelUniqueDeclOp>(op))
      labelNames.add(labelDecl.getFormatString());
  });

  // Lower the unique labels in a second iteration.
  getOperation()->walk([&](LabelUniqueDeclOp op) {
    // Convert 'rtg.label_unique_decl' to 'rtg.label_decl' by choosing a unique
    // name based on the set of names we collected during elaboration.
    IRRewriter rewriter(op);
    auto newName = labelNames.newName(op.getFormatString());
    rewriter.replaceOpWithNewOp<LabelDeclOp>(op, newName, ValueRange());
    ++numLabelsLowered;
  });
}
