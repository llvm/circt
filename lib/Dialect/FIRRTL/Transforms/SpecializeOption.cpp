//===- SpecializeOption.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_SPECIALIZEOPTION
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {
struct SpecializeOptionPass
    : public circt::firrtl::impl::SpecializeOptionBase<SpecializeOptionPass> {
  using SpecializeOptionBase::numInstances;
  using SpecializeOptionBase::selectDefaultInstanceChoice;

  void runOnOperation() override {
    auto circuit = getOperation();

    DenseMap<StringAttr, OptionCaseOp> selected;
    if (auto choiceAttr = circuit.getSelectInstChoiceAttr()) {
      for (auto attr : choiceAttr.getAsRange<StringAttr>()) {
        const auto optionAndCase = attr.getValue().str();
        size_t eq = optionAndCase.find("=");
        if (eq == std::string::npos) {
          mlir::emitError(circuit.getLoc(),
                          "invalid option format: \"" + optionAndCase + '"');
          return signalPassFailure();
        }

        std::string optionName = optionAndCase.substr(0, eq);
        auto optionOp = circuit.lookupSymbol<OptionOp>(optionName);
        if (!optionOp) {
          mlir::emitError(circuit.getLoc(), "unknown option \"")
              << optionName << '"';
          return signalPassFailure();
        }

        std::string caseName = optionAndCase.substr(eq + 1);
        auto caseOp = optionOp.lookupSymbol<OptionCaseOp>(caseName);
        if (!caseOp) {
          mlir::emitError(circuit.getLoc(), "invalid option case \"")
              << caseName << '"';
          return signalPassFailure();
        }
        selected[StringAttr::get(&getContext(), optionName)] = caseOp;
      }
    }

    bool failed = false;
    mlir::parallelForEach(
        &getContext(), circuit.getOps<FModuleOp>(), [&](auto module) {
          module.walk([&](firrtl::InstanceChoiceOp inst) {
            auto it = selected.find(inst.getOptionNameAttr());
            FlatSymbolRefAttr target;
            if (it == selected.end()) {
              if (!selectDefaultInstanceChoice) {
                inst.emitError("missing specialization for option ")
                    << inst.getOptionNameAttr();
                failed = true;
                return;
              }
              target = inst.getDefaultTargetAttr();
            } else
              target = inst.getTargetOrDefaultAttr(it->second);

            ImplicitLocOpBuilder builder(inst.getLoc(), inst);
            auto newInst = builder.create<InstanceOp>(
                inst->getResultTypes(), target, inst.getNameAttr(),
                inst.getNameKindAttr(), inst.getPortDirectionsAttr(),
                inst.getPortNamesAttr(), inst.getAnnotationsAttr(),
                inst.getPortAnnotationsAttr(), builder.getArrayAttr({}),
                UnitAttr{}, UnitAttr{}, inst.getInnerSymAttr());
            inst.replaceAllUsesWith(newInst);
            inst.erase();

            ++numInstances;
          });
        });

    bool analysisPreserved = numInstances == 0;
    circuit->walk([&](OptionOp optionOp) {
      optionOp->erase();
      analysisPreserved = false;
    });
    if (analysisPreserved)
      markAllAnalysesPreserved();

    if (failed)
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
firrtl::createSpecializeOptionPass(bool selectDefaultInstanceChoice) {
  auto pass = std::make_unique<SpecializeOptionPass>();
  pass->selectDefaultInstanceChoice = selectDefaultInstanceChoice;
  return pass;
}
