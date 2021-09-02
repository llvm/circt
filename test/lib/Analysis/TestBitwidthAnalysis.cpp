//===- TestBitwidthAnalysis.cpp - Test bitwidth analysis results ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and testing bitwidth analysis
// results.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/BitwidthAnalysis.h"
#include "mlir/Pass/Pass.h"

using namespace circt;
using namespace mlir;
using namespace llvm;

namespace {

// Add bitwidths of operation results to all operations.
void addBitwidthAttributes(MLIRContext *ctx, const BitwidthAnalysis &res,
                           Operation *op, ValueRange values) {
  SmallVector<Attribute> resWidths;
  llvm::transform(
      values, std::back_inserter(resWidths), [&](auto v) -> Attribute {
        if (auto width = res.valueWidth(v))
          return IntegerAttr::get(IntegerType::get(ctx, 64), width.getValue());
        else
          return StringAttr::get(ctx, "N/A");
      });
  if (resWidths.size() != 0)
    op->setAttr("result bits", ArrayAttr::get(ctx, resWidths));
};

struct TestBitwidthAnalysisPass
    : public PassWrapper<TestBitwidthAnalysisPass,
                         OperationPass<mlir::ModuleOp>> {
  StringRef getArgument() const final { return "test-bitwidth-analysis"; }
  StringRef getDescription() const final {
    return "Test bitwidth analysis results.";
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto ctx = module.getContext();
    module.walk([&](mlir::FuncOp func) {
      auto analysisResult = BitwidthAnalysis(func, 32);
      func.walk([&](Operation *op) {
        addBitwidthAttributes(ctx, analysisResult, op, op->getResults());
      });
    });
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace test {
void registerTestBitwidthAnalysisPass() {
  PassRegistration<TestBitwidthAnalysisPass>();
}
} // namespace test
} // namespace circt
