//===- CheckInstanceChoice.cpp - Check instance choice configurations -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/FIRRTLInstanceInfo.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/ErrorHandling.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CHECKINSTANCECHOICE
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using namespace mlir;

namespace {
class CheckInstanceChoicePass
    : public circt::firrtl::impl::CheckInstanceChoiceBase<
          CheckInstanceChoicePass> {
public:
  using CheckInstanceChoiceBase::CheckInstanceChoiceBase;

  void runOnOperation() override {
    auto circuit = getOperation();
    auto &instanceInfo = getAnalysis<InstanceInfo>();
    auto &instanceGraph = getAnalysis<InstanceGraph>();

    // Check for nested instance choices by walking all modules
    for (auto module : circuit.getOps<FModuleOp>()) {
      // Skip modules that are not instantiated within an instance choice
      if (!instanceInfo.anyInstanceInInstanceChoice(module))
        continue;

      // Find the instance graph node for this module
      auto *node = instanceGraph.lookup(module.getModuleNameAttr());
      assert(node && "module not found in instance graph");
      // Walk the children
      for (auto *child : *node) {
        auto childOp = child->getInstance();
        if (!isa<InstanceChoiceOp>(childOp))
          continue;
        // Error.
        auto diag = childOp->emitOpError(
            "instance choice within another instance choice "
            "is not allowed");

        // Backtrace to the parent instance choice.
        while (true) {
          for (auto *use : node->uses()) {
            if (auto parent = dyn_cast<InstanceChoiceOp>(use->getInstance())) {
              diag.attachNote(parent->getLoc())
                  << "is the parent instance choice";
              return signalPassFailure();
            }
            auto *parent = use->getParent();
            if (!parent ||
                !instanceInfo.anyInstanceInInstanceChoice(parent->getModule()))
              continue;
            node = parent;
            break;
          }
        }
        llvm::llvm_unreachable_internal(
            "should have found a parent instance choice");
      }
    }

    markAllAnalysesPreserved();
  }
};
} // namespace
