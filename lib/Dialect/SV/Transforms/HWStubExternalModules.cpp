//===- HWStubExternalModules.cpp - HW Module Stubbing Pass ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts external modules to empty normal modules.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace sv {
#define GEN_PASS_DEF_HWSTUBEXTERNALMODULES
#include "circt/Dialect/SV/SVPasses.h.inc"
} // namespace sv
} // namespace circt

using namespace circt;
//===----------------------------------------------------------------------===//
// HWStubExternalModules Pass
//===----------------------------------------------------------------------===//

namespace {
struct HWStubExternalModulesPass
    : public circt::sv::impl::HWStubExternalModulesBase<
          HWStubExternalModulesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void HWStubExternalModulesPass::runOnOperation() {
  auto topModule = getOperation().getBody();
  OpBuilder builder(topModule->getParentOp()->getContext());
  builder.setInsertionPointToEnd(topModule);

  for (auto &op : llvm::make_early_inc_range(*topModule))
    if (auto module = dyn_cast<hw::HWModuleExternOp>(op)) {
      hw::ModulePortInfo ports(module.getPortList());
      auto nameAttr = module.getNameAttr();
      auto newModule = hw::HWModuleOp::create(
          builder, module.getLoc(), nameAttr, ports, module.getParameters());
      auto outputOp = newModule.getBodyBlock()->getTerminator();
      OpBuilder innerBuilder(outputOp);
      SmallVector<Value, 8> outputs;
      // All output ports need values, use x
      for (auto &p : ports.getOutputs()) {
        outputs.push_back(
            sv::ConstantXOp::create(innerBuilder, outputOp->getLoc(), p.type));
      }
      outputOp->setOperands(outputs);

      // Done with the old module.
      module.erase();
    }
}

std::unique_ptr<Pass> circt::sv::createHWStubExternalModulesPass() {
  return std::make_unique<HWStubExternalModulesPass>();
}
