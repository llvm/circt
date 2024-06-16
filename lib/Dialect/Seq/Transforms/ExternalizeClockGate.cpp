//===- ExternalizeClockGate.cpp - Convert clock gate to extern module -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_EXTERNALIZECLOCKGATE
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;
using namespace hw;

namespace {
struct ExternalizeClockGatePass
    : public circt::seq::impl::ExternalizeClockGateBase<
          ExternalizeClockGatePass> {
  using ExternalizeClockGateBase<
      ExternalizeClockGatePass>::ExternalizeClockGateBase;
  void runOnOperation() override;
};
} // anonymous namespace

void ExternalizeClockGatePass::runOnOperation() {
  SymbolTable &symtbl = getAnalysis<SymbolTable>();

  // Collect all clock gate ops.
  DenseMap<HWModuleOp, SmallVector<ClockGateOp>> gatesInModule;
  for (auto module : getOperation().getOps<HWModuleOp>()) {
    module.walk([&](ClockGateOp op) { gatesInModule[module].push_back(op); });
  }

  if (gatesInModule.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  // Generate the external module declaration.
  auto builder = OpBuilder::atBlockBegin(getOperation().getBody());
  auto i1Type = builder.getI1Type();

  SmallVector<PortInfo, 4> modulePorts;
  modulePorts.push_back({{builder.getStringAttr(inputName), i1Type,
                          ModulePort::Direction::Input}});
  modulePorts.push_back({{builder.getStringAttr(outputName), i1Type,
                          ModulePort::Direction::Output}});
  modulePorts.push_back({{builder.getStringAttr(enableName), i1Type,
                          ModulePort::Direction::Input}});
  bool hasTestEnable = !testEnableName.empty();
  if (hasTestEnable)
    modulePorts.push_back({{builder.getStringAttr(testEnableName), i1Type,
                            ModulePort::Direction::Input}});

  auto externModuleOp = builder.create<HWModuleExternOp>(
      getOperation().getLoc(), builder.getStringAttr(moduleName), modulePorts,
      moduleName);
  symtbl.insert(externModuleOp);

  // Replace all clock gates with an instance of the external module.
  SmallVector<Value, 4> instPorts;
  for (auto &[module, clockGatesToReplace] : gatesInModule) {
    DenseMap<Value, Value> fromClocks;
    Value cstFalse;
    for (auto ckgOp : clockGatesToReplace) {
      ImplicitLocOpBuilder builder(ckgOp.getLoc(), ckgOp);

      Value enable = ckgOp.getEnable();
      Value testEnable = ckgOp.getTestEnable();

      // If the clock gate has no test enable operand but the module does, add a
      // constant 0 input.
      if (hasTestEnable && !testEnable) {
        if (!cstFalse) {
          cstFalse = builder.create<ConstantOp>(i1Type, 0);
        }
        testEnable = cstFalse;
      }

      // If the clock gate has a test enable operand but the module does not,
      // add a `comb.or` to merge the two enable conditions.
      if (!hasTestEnable && testEnable) {
        enable = builder.createOrFold<comb::OrOp>(enable, testEnable, true);
        testEnable = {};
      }

      Value input = ckgOp.getInput();
      if (hw::type_isa<seq::ClockType>(input.getType())) {
        auto it = fromClocks.try_emplace(input, Value{});
        if (it.second) {
          ImplicitLocOpBuilder builder(input.getLoc(), &getContext());
          builder.setInsertionPointAfterValue(input);
          it.first->second = builder.create<seq::FromClockOp>(input);
        }
        instPorts.push_back(it.first->second);
      } else {
        instPorts.push_back(input);
      }
      instPorts.push_back(enable);
      if (testEnable)
        instPorts.push_back(testEnable);

      auto instOp = builder.create<InstanceOp>(
          externModuleOp, builder.getStringAttr(instName), instPorts,
          builder.getArrayAttr({}), ckgOp.getInnerSymAttr());

      Value output = instOp.getResult(0);
      if (hw::type_isa<seq::ClockType>(input.getType()))
        output = builder.create<seq::ToClockOp>(output);
      ckgOp.replaceAllUsesWith(output);
      ckgOp.erase();

      instPorts.clear();
      ++numClockGatesConverted;
    }
  }
}

std::unique_ptr<Pass> circt::seq::createExternalizeClockGatePass(
    const ExternalizeClockGateOptions &options) {
  return std::make_unique<ExternalizeClockGatePass>(options);
}
