//===- ApplyLoweringOptions.cpp - Test pass for adding lowering options ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass for testing purposes to apply lowering options ot a module.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ExportVerilog.h"
#include "../PassDetail.h"
#include "ExportVerilogInternals.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombVisitors.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVVisitors.h"
#include "circt/Support/LoweringOptions.h"

using namespace circt;


namespace {

struct testApplyLoweringOptionPass
    : public TestApplyLoweringOptionBase<testApplyLoweringOptionPass> {
  testApplyLoweringOptionPass(StringRef _options) {
    options = _options.str();
  }
  testApplyLoweringOptionPass() = default;
  void runOnOperation() override {
    if (!options.hasValue()) {
        markAllAnalysesPreserved();
        return;
    }
    LoweringOptions opts(options, [this](llvm::Twine tw) {
        getOperation().emitError(tw);
        signalPassFailure();
    });
    opts.setAsAttribute(getOperation());
  }  
};
}

std::unique_ptr<mlir::Pass>
circt::createTestApplyLoweringOptionPass(std::string options) {
  return std::make_unique<testApplyLoweringOptionPass>(options);
}

std::unique_ptr<mlir::Pass>
circt::createTestApplyLoweringOptionPass() {
  return std::make_unique<testApplyLoweringOptionPass>();
}
