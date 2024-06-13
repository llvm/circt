//===- FlattenWires.cpp - Flattens all wires ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Replace all wires with their inputs.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/Naming.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Pass/Pass.h"

using namespace circt;
using namespace mlir;
using namespace verif;

namespace {
/// Flatten wires by unconditionally replacing them with their inputs
/// Wires create an alias for a set of operations, they are usually removed
/// through canonicalization at some point.Some wires are however
/// maintained.This pass unconditionally replaces all wires with their inputs,
/// making it easier to reason about in contexts where wires don't exist.
struct WireOpConversionPattern : OpConversionPattern<hw::WireOp> {
  using OpConversionPattern<hw::WireOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::WireOp wire, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If the wire has a name or an `sv.namehint` attribute, propagate it as an
    // `sv.namehint` to the expression.
    if (auto *inputOp = adaptor.getInput().getDefiningOp())
      if (auto name = chooseName(wire, inputOp))
        rewriter.modifyOpInPlace(
            inputOp, [&] { inputOp->setAttr("sv.namehint", name); });

    rewriter.replaceOp(wire, adaptor.getInput());
    return success();
  }
};

// Eagerly replace all wires with their inputs
struct PrepareForFormalPass : PrepareForFormalBase<PrepareForFormalPass> {
  void runOnOperation() override;
};
} // namespace

void PrepareForFormalPass::runOnOperation() {
  // Set target: We don't want any wires left in our output
  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addIllegalOp<hw::WireOp>();

  // Create the operation rewrite patters
  RewritePatternSet patterns(&getContext());
  patterns.add<WireOpConversionPattern>(patterns.getContext());

  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::verif::createPrepareForFormalPass() {
  return std::make_unique<PrepareForFormalPass>();
}
