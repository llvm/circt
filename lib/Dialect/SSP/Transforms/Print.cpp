//===- Print.cpp - Print pass -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the Print (as a DOT graph) pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SSP/SSPAttributes.h"
#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Dialect/SSP/SSPPasses.h"
#include "circt/Dialect/SSP/Utilities.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Pass/Pass.h"

#include "circt/Scheduling/Utilities.h"

namespace circt {
namespace ssp {
#define GEN_PASS_DEF_PRINT
#include "circt/Dialect/SSP/SSPPasses.h.inc"
} // namespace ssp
} // namespace circt

using namespace circt;
using namespace scheduling;
using namespace ssp;

namespace {
struct PrintPass : public circt::ssp::impl::PrintBase<PrintPass> {
  explicit PrintPass(raw_ostream &os) : os(os) {}
  void runOnOperation() override;
  raw_ostream &os;
};
} // end anonymous namespace

template <typename ProblemT>
static void printInstance(InstanceOp instOp, raw_ostream &os) {
  auto prob = loadProblem<ProblemT>(instOp);
  dumpAsDOT(prob, os);
}

void PrintPass::runOnOperation() {
  auto moduleOp = getOperation();
  for (auto instOp : moduleOp.getOps<InstanceOp>()) {
    StringRef probName = instOp.getProblemName();
    if (probName == "Problem")
      printInstance<Problem>(instOp, os);
    else if (probName == "CyclicProblem")
      printInstance<CyclicProblem>(instOp, os);
    else if (probName == "ChainingProblem")
      printInstance<ChainingProblem>(instOp, os);
    else if (probName == "SharedResourcesProblem")
      printInstance<SharedResourcesProblem>(instOp, os);
    else if (probName == "ModuloProblem")
      printInstance<ModuloProblem>(instOp, os);
    else if (probName == "ChainingCyclicProblem")
      printInstance<ChainingCyclicProblem>(instOp, os);
    else {
      auto instName = instOp.getSymName().value_or("unnamed");
      llvm::errs() << "ssp-print-instance: Unknown problem class '" << probName
                   << "' in instance '" << instName << "'\n";
      return signalPassFailure();
    }
  }
}

std::unique_ptr<mlir::Pass> circt::ssp::createPrintPass() {
  return std::make_unique<PrintPass>(llvm::errs());
}
