//===- MaterializeDebugVariables.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
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

    DenseSet<StringAttr> outputPortNames;
    for (auto &port : module.getPortList())
      if (port.isOutput())
        outputPortNames.insert(port.name);

    OpBuilder builder = OpBuilder::atBlockBegin(body);
    StringRef stateSuffix = "_state";
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

      StringAttr debugName = port.name;
      auto portName = port.name.getValue();
      if (portName.ends_with(stateSuffix)) {
        auto baseName = portName.drop_back(stateSuffix.size());
        auto nextNameAttr =
            builder.getStringAttr((Twine(baseName) + "_next").str());
        if (outputPortNames.contains(nextNameAttr))
          debugName = builder.getStringAttr(baseName);
      }

      debug::VariableOp::create(builder, value.getLoc(), debugName, value,
                                /*scope=*/Value{});
    }
  }
};
} // namespace
