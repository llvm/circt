//===- SVTraceIVerilog.cpp - Generator Callout Pass -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass adds the necessary instrumentation to a HWModule to trigger
// tracing in an iverilog simulation.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

using namespace circt;
using namespace sv;
using namespace hw;

//===----------------------------------------------------------------------===//
// SVTraceIVerilogPass
//===----------------------------------------------------------------------===//

namespace {

struct SVTraceIVerilogPass
    : public sv::SVTraceIVerilogBase<SVTraceIVerilogPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

void SVTraceIVerilogPass::runOnOperation() {
  ModuleOp module = getOperation();

  for (auto mod : module.getOps<hw::HWModuleOp>()) {
    if (!targetModuleName.empty() &&
        mod.getName() != targetModuleName.getValue())
      continue;

    OpBuilder builder(mod.getBodyBlock(), mod.getBodyBlock()->begin());
    std::string traceMacro;
    llvm::raw_string_ostream ss(traceMacro);
    auto modName = mod.getName();
    ss << "initial begin\n  $dumpfile (\"" << directoryName.getValue()
       << modName << ".vcd\");\n  $dumpvars (0, " << modName
       << ");\n  #1;\nend\n";
    builder.create<sv::VerbatimOp>(mod.getLoc(), ss.str());
  }
}

std::unique_ptr<Pass> circt::sv::createSVTraceIVerilogPass() {
  return std::make_unique<SVTraceIVerilogPass>();
}
