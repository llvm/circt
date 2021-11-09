//===- MSFTPasses.cpp - Implement MSFT passes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/ExportTcl.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/SV/SVOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"

using namespace circt;
using namespace msft;

namespace circt {
namespace msft {
#define GEN_PASS_CLASSES
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace msft
} // namespace circt

//===----------------------------------------------------------------------===//
// Lower MSFT to HW.
//===----------------------------------------------------------------------===//

NamedAttribute createOutputFileAttr(MLIRContext *context,
                                    StringRef outputFile) {
  auto outputFileName = Identifier::get("output_file", context);
  auto outputFileAttr =
      hw::OutputFileAttr::getFromFilename(context, outputFile);
  return NamedAttribute(outputFileName, outputFileAttr);
}

namespace {
/// Lower MSFT's InstanceOp to HW's. Currently trivial since `msft.instance` is
/// currently a subset of `hw.instance`.
struct InstanceOpLowering : public OpConversionPattern<InstanceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;
};
} // anonymous namespace

LogicalResult
InstanceOpLowering::matchAndRewrite(InstanceOp msftInst, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Operation *referencedModule = msftInst.getReferencedModule();
  if (!referencedModule)
    return rewriter.notifyMatchFailure(msftInst,
                                       "Could not find referenced module");
  if (!hw::isAnyModule(referencedModule))
    return rewriter.notifyMatchFailure(
        msftInst, "Referenced module was not an HW module");
  auto hwInst = rewriter.create<hw::InstanceOp>(
      msftInst.getLoc(), referencedModule, msftInst.instanceNameAttr(),
      SmallVector<Value>(adaptor.getOperands().begin(),
                         adaptor.getOperands().end()),
      msftInst.parameters().getValueOr(ArrayAttr()), msftInst.sym_nameAttr());
  rewriter.replaceOp(msftInst, hwInst.getResults());
  return success();
}

namespace {
/// Lower MSFT's ModuleOp to HW's.
struct ModuleOpLowering : public OpConversionPattern<MSFTModuleOp> {
public:
  ModuleOpLowering(MLIRContext *context, StringRef outputFile)
      : OpConversionPattern::OpConversionPattern(context),
        outputFile(outputFile) {}

  LogicalResult
  matchAndRewrite(MSFTModuleOp mod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  StringRef outputFile;
};
} // anonymous namespace

LogicalResult
ModuleOpLowering::matchAndRewrite(MSFTModuleOp mod, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
  if (mod.body().empty()) {
    std::string comment;
    llvm::raw_string_ostream(comment)
        << "// Module not generated: \"" << mod.getName() << "\" params "
        << mod.parameters();
    // TODO: replace this with proper comment op when it's created.
    rewriter.replaceOpWithNewOp<sv::VerbatimOp>(mod, comment);
    return success();
  }

  auto outputFileAttr = createOutputFileAttr(rewriter.getContext(), outputFile);

  auto hwmod = rewriter.replaceOpWithNewOp<hw::HWModuleOp>(
      mod, mod.getNameAttr(), mod.getPorts(), /*parameters=*/ArrayAttr(),
      /*attributes=*/ArrayRef(outputFileAttr));
  rewriter.eraseBlock(hwmod.getBodyBlock());
  rewriter.inlineRegionBefore(mod.getBody(), hwmod.getBody(),
                              hwmod.getBody().end());
  return success();
}
namespace {

/// Lower MSFT's ModuleExternOp to HW's.
struct ModuleExternOpLowering : public OpConversionPattern<MSFTModuleExternOp> {
public:
  ModuleExternOpLowering(MLIRContext *context, StringRef outputFile)
      : OpConversionPattern::OpConversionPattern(context),
        outputFile(outputFile) {}

  LogicalResult
  matchAndRewrite(MSFTModuleExternOp mod, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final;

private:
  StringRef outputFile;
};
} // anonymous namespace

LogicalResult ModuleExternOpLowering::matchAndRewrite(
    MSFTModuleExternOp mod, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto outputFileAttr = createOutputFileAttr(rewriter.getContext(), outputFile);

  rewriter.replaceOpWithNewOp<hw::HWModuleExternOp>(
      mod, mod.getNameAttr(), mod.getPorts(), mod.verilogName().getValueOr(""),
      mod.parameters(), /*attributes=*/ArrayRef(outputFileAttr));

  return success();
}

namespace {
/// Lower MSFT's OutputOp to HW's.
struct OutputOpLowering : public OpConversionPattern<OutputOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp out, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<hw::OutputOp>(out, out.getOperands());
    return success();
  }
};
} // anonymous namespace

namespace {
struct LowerToHWPass : public LowerToHWBase<LowerToHWPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void LowerToHWPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();

  // Check output file pass options have been specified.
  if (verilogFile.empty()) {
    top.emitError("verilogFile pass option was not specified");
    return signalPassFailure();
  }
  if (tclFile.empty()) {
    top.emitError("tclFile pass option was not specified");
    return signalPassFailure();
  }

  // Traverse MSFT location attributes and export the required Tcl into
  // templated `sv::VerbatimOp`s with symbolic references to the instance paths.
  hw::SymbolCache symCache;
  populateSymbolCache(top, symCache);
  for (auto hwmod : top.getBody()->getOps<msft::MSFTModuleOp>())
    if (failed(exportQuartusTcl(hwmod, symCache, tclFile)))
      return signalPassFailure();

  // The `hw::InstanceOp` (which `msft::InstanceOp` lowers to) convenience
  // builder gets its argNames and resultNames from the `hw::HWModuleOp`. So we
  // have to lower `msft::MSFTModuleOp` before we lower `msft::InstanceOp`.

  // Convert everything except instance ops first.

  ConversionTarget target(*ctxt);
  target.addIllegalOp<MSFTModuleOp, MSFTModuleExternOp, OutputOp>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<sv::SVDialect>();

  RewritePatternSet patterns(ctxt);
  patterns.insert<ModuleOpLowering>(ctxt, verilogFile);
  patterns.insert<ModuleExternOpLowering>(ctxt, verilogFile);
  patterns.insert<OutputOpLowering>(ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();

  // Then, convert the InstanceOps
  target.addIllegalDialect<MSFTDialect>();
  RewritePatternSet instancePatterns(ctxt);
  instancePatterns.insert<InstanceOpLowering>(ctxt);
  if (failed(applyPartialConversion(top, target, std::move(instancePatterns))))
    signalPassFailure();
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createLowerToHWPass() {
  return std::make_unique<LowerToHWPass>();
}
} // namespace msft
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/MSFT/MSFTPasses.h.inc"
} // namespace

void circt::msft::registerMSFTPasses() { registerPasses(); }
