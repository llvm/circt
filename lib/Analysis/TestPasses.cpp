//===- TestPasses.cpp - Test passes for the analysis infrastructure -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements test passes for the analysis infrastructure.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace circt::analysis;

//===----------------------------------------------------------------------===//
// DependenceAnalysis passes.
//===----------------------------------------------------------------------===//

namespace {
struct TestDependenceAnalysisPass
    : public PassWrapper<TestDependenceAnalysisPass, FunctionPass> {
  void runOnFunction() override;
  StringRef getArgument() const override { return "test-dependence-analysis"; }
  StringRef getDescription() const override {
    return "Perform dependence analysis and emit results as attributes";
  }
};
} // namespace

void TestDependenceAnalysisPass::runOnFunction() {
  MLIRContext *context = &getContext();

  MemoryDependenceResult results;
  getMemoryAccessDependences(getFunction(), results);

  for (auto sourceToDeps : results) {
    SmallVector<Attribute> deps;

    for (auto dep : sourceToDeps.second) {
      if (dep.dependenceType != mlir::DependenceResult::HasDependence)
        continue;

      SmallVector<Attribute> comps;
      for (auto comp : dep.dependenceComponents) {
        SmallVector<Attribute> vector;
        vector.push_back(IntegerAttr::get(IntegerType::get(context, 64),
                                          comp.lb.getValue()));
        vector.push_back(IntegerAttr::get(IntegerType::get(context, 64),
                                          comp.ub.getValue()));
        comps.push_back(ArrayAttr::get(context, vector));
      }

      deps.push_back(ArrayAttr::get(context, comps));
    }

    auto dependences = ArrayAttr::get(context, deps);
    sourceToDeps.first->setAttr("dependences", dependences);
  }
}

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace circt {
namespace test {
void registerAnalysisTestPasses() {
  mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<TestDependenceAnalysisPass>();
  });
}
} // namespace test
} // namespace circt
