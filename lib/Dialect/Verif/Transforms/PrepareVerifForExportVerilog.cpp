//===- PrepareVerifForExportVerilog.cpp - Formal Preparations --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Prepare verif dialect for the verilog export.
//
//===----------------------------------------------------------------------===//
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;

namespace circt {
namespace verif {
#define GEN_PASS_DEF_PREPAREVERIFFOREXPORTVERILOG
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace mlir;
using namespace verif;

namespace {
struct PrepareVerifForExportVerilog
    : circt::verif::impl::PrepareVerifForExportVerilogBase<
          PrepareVerifForExportVerilog> {
  void runOnOperation() override;
};

struct FormalOpConversionPattern : public OpConversionPattern<verif::FormalOp> {
  using OpConversionPattern<FormalOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FormalOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the ports for all the symbolic values
    SmallVector<hw::PortInfo> ports;
    for (auto sym : op.getBody().front().getOps<verif::SymbolicValueOp>()) {
      ports.push_back(
          hw::PortInfo({{rewriter.getStringAttr("symbolic_value_" +
                                                std::to_string(ports.size())),
                         sym.getType(), hw::ModulePort::Input}}));
    }

    auto moduleOp =
        rewriter.create<hw::HWModuleOp>(op.getLoc(), op.getNameAttr(), ports);

    // Insert before output op
    rewriter.setInsertionPointToStart(moduleOp.getBodyBlock());

    // Clone body except for symbolic values which we replace with module
    // arguments
    size_t i = 0;
    for (auto &innerOp : op.getBody().front().getOperations()) {
      if (dyn_cast<verif::SymbolicValueOp>(innerOp)) {
        rewriter.replaceAllUsesWith(innerOp.getResult(0),
                                    moduleOp.getArgumentForInput(i));
        i++;
      } else {
        rewriter.clone(innerOp);
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

void PrepareVerifForExportVerilog::runOnOperation() {
  auto &context = getContext();
  mlir::ConversionTarget target(context);
  target.addLegalDialect<hw::HWDialect, verif::VerifDialect>();
  target.addIllegalOp<verif::FormalOp, verif::SymbolicValueOp>();

  RewritePatternSet patterns(&context);
  patterns.add<FormalOpConversionPattern>(&context);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
