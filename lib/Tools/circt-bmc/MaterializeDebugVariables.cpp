//===- MaterializeDebugVariables.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/LLVM.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_MATERIALIZEDEBUGVARIABLES
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

namespace {
struct MaterializeDebugVariablesPass
    : public circt::impl::MaterializeDebugVariablesBase<
          MaterializeDebugVariablesPass> {
  using MaterializeDebugVariablesBase::MaterializeDebugVariablesBase;

  void runOnOperation() override {
    for (auto module : getOperation().getOps<HWModuleOp>())
      materializeDebugVariables(module);
  }

private:
  static void materializeDebugVariables(HWModuleOp module) {
    auto *body = module.getBodyBlock();
    DenseSet<Value> trackedValues;
    for (auto varOp : body->getOps<debug::VariableOp>())
      trackedValues.insert(varOp.getValue());

    OpBuilder builder = OpBuilder::atBlockBegin(body);
    for (auto &port : module.getPortList()) {
      if (port.isOutput())
        continue;
      if (isa<seq::ClockType>(port.type))
        continue;
      if (!port.name || port.name.getValue().empty())
        continue;

      auto value = body->getArgument(port.argNum);
      if (!trackedValues.insert(value).second)
        continue;
      debug::VariableOp::create(builder, value.getLoc(), port.name, value,
                                /*scope=*/Value{});
    }

    auto materializeRegAnchor = [&](auto regOp, StringRef regName) {
      auto regResult = regOp.getResult();
      if (!trackedValues.insert(regResult).second)
        return;
      auto regNameAttr = builder.getStringAttr(regName);
      OpBuilder regBuilder(regOp);
      regBuilder.setInsertionPointAfter(regOp);
      debug::VariableOp::create(regBuilder, regResult.getLoc(), regNameAttr,
                                regResult, /*scope=*/Value{});
    };

    unsigned regIndex = 0;
    for (auto regOp : body->getOps<seq::CompRegOp>()) {
      if (regOp.getName() && !regOp.getName()->empty())
        materializeRegAnchor(regOp, regOp.getName().value());
      else
        materializeRegAnchor(regOp, ("reg_" + Twine(regIndex)).str());
      ++regIndex;
    }

    for (auto regOp : body->getOps<seq::FirRegOp>()) {
      if (!regOp.getName().empty())
        materializeRegAnchor(regOp, regOp.getName());
      else
        materializeRegAnchor(regOp, ("reg_" + Twine(regIndex)).str());
      ++regIndex;
    }
  }
};
} // namespace
