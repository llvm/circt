//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Support/ConversionPatternSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-impl-runtime"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_IMPLRUNTIMEPASS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;
using func::FuncOp;

//===----------------------------------------------------------------------===//
// String Formatting
//===----------------------------------------------------------------------===//

namespace {
/// A helper for discovering and generating formatting functions as we convert
/// string formatting ops.
struct FormattingFuncs {
  FormattingFuncs(ModuleOp op, SymbolTable &symbolTable)
      : builder(op), symbolTable(symbolTable) {
    builder.setInsertionPointToStart(op.getBody());
    auto ptrType = LLVM::LLVMPointerType::get(builder.getContext());
    auto voidType = LLVM::LLVMVoidType::get(builder.getContext());
    funcType = LLVM::LLVMFunctionType::get(voidType, {ptrType});
  }

  LLVM::GlobalOp getStringGlobal(const Twine &str);
  LLVM::LLVMFuncOp getFormatLiteral();

  OpBuilder builder;
  SymbolTable &symbolTable;
  LLVM::LLVMFunctionType funcType;

  DenseMap<StringAttr, LLVM::GlobalOp> stringGlobals;
  LLVM::LLVMFuncOp formatLiteral;

  static constexpr StringLiteral formatLiteralName = "arc.format.literal";
};
} // namespace

LLVM::GlobalOp FormattingFuncs::getStringGlobal(const Twine &str) {
  auto attr = builder.getStringAttr(str + StringRef("", 1));
  auto &global = stringGlobals[attr];
  if (!global) {
    auto type = LLVM::LLVMArrayType::get(builder.getI8Type(), attr.size());
    global =
        LLVM::GlobalOp::create(builder, builder.getUnknownLoc(), type,
                               /*isConstant=*/true, LLVM::Linkage::Internal,
                               /*name=*/"arc.str", attr);
    symbolTable.insert(global);
  }
  return global;
}

LLVM::LLVMFuncOp FormattingFuncs::getFormatLiteral() {
  if (formatLiteral)
    return formatLiteral;

  formatLiteral = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                           formatLiteralName, funcType);
  SymbolTable::setSymbolVisibility(formatLiteral,
                                   SymbolTable::Visibility::Private);
  symbolTable.insert(formatLiteral);
  return formatLiteral;
}

static LogicalResult convert(sim::FormatLitOp op,
                             sim::FormatLitOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             FormattingFuncs &funcs) {
  auto *context = rewriter.getContext();
  auto loc = op.getLoc();
  auto ptrType = LLVM::LLVMPointerType::get(context);

  // Materialize a reference to the formatting function.
  auto func = funcs.getFormatLiteral();
  auto funcRef =
      LLVM::AddressOfOp::create(rewriter, loc, ptrType, func.getSymNameAttr());

  // Materialize a reference to the literal.
  auto literalRef = LLVM::AddressOfOp::create(
      rewriter, loc, funcs.getStringGlobal(op.getLiteral()));

  // Package everything up in a struct.
  auto type = LLVM::LLVMStructType::getLiteral(context, {ptrType, ptrType});
  Value value = LLVM::UndefOp::create(rewriter, loc, type);
  value = LLVM::InsertValueOp::create(rewriter, loc, value, funcRef,
                                      ArrayRef<int64_t>{0});
  value = LLVM::InsertValueOp::create(rewriter, loc, value, literalRef,
                                      ArrayRef<int64_t>{1});
  // rewriter.replaceAllUsesWith(op, {funcRef, literalRef});
  // rewriter.eraseOp(op);
  rewriter.replaceOp(op, value);
  return success();
}

static LogicalResult convert(sim::PrintFormattedProcOp op,
                             sim::PrintFormattedProcOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter,
                             FormattingFuncs &funcs) {
  auto loc = op.getLoc();

  // Extract the formatting function and data references.
  auto funcRef = LLVM::ExtractValueOp::create(rewriter, loc, adaptor.getInput(),
                                              ArrayRef<int64_t>{0});
  auto dataRef = LLVM::ExtractValueOp::create(rewriter, loc, adaptor.getInput(),
                                              ArrayRef<int64_t>{1});

  // Call the formatting function with the data reference.
  LLVM::CallOp::create(rewriter, loc, funcs.funcType,
                       ValueRange{funcRef, dataRef});
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct ImplRuntimePass
    : public arc::impl::ImplRuntimePassBase<ImplRuntimePass> {
  void runOnOperation() override;
};
} // namespace

void ImplRuntimePass::runOnOperation() {
  // Collect the existing formatting functions.
  auto &symbolTable = getAnalysis<SymbolTable>();
  FormattingFuncs formattingFuncs(getOperation(), symbolTable);

  // Setup the type conversion.
  TypeConverter converter;
  converter.addConversion([](sim::FormatStringType type) {
    auto ptrType = LLVM::LLVMPointerType::get(type.getContext());
    return LLVM::LLVMStructType::getLiteral(type.getContext(),
                                            {ptrType, ptrType});
  });

  // Gather the conversion patterns.
  ConversionPatternSet patterns(&getContext(), converter);
  patterns.add<sim::FormatLitOp>(convert, formattingFuncs);
  patterns.add<sim::PrintFormattedProcOp>(convert, formattingFuncs);

  // Setup the legal ops.
  ConversionTarget target(getContext());
  target.addIllegalOp<sim::FormatLitOp>();
  target.addIllegalOp<sim::PrintFormattedProcOp>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  // Disable pattern rollback to use the faster one-shot dialect conversion.
  ConversionConfig config;
  config.allowPatternRollback = false;

  // Perform the conversion.
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns),
                                    config)))
    return signalPassFailure();
}
