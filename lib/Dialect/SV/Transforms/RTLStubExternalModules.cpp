//===- RTLStubExternalModules.cpp - RTL Module Stubbing Pass --------------===//
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

#include "SVPassDetail.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// RTLStubExternalModules Pass
//===----------------------------------------------------------------------===//

namespace {
struct RTLStubExternalModulesPass
    : public sv::RTLStubExternalModulesBase<RTLStubExternalModulesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

void RTLStubExternalModulesPass::runOnOperation() {
  auto topModule = getOperation().getBody();
  OpBuilder builder(topModule->getParentOp()->getContext());
  builder.setInsertionPointToEnd(topModule);

  for (auto &op : llvm::make_early_inc_range(*topModule))
    if (auto module = dyn_cast<rtl::RTLModuleExternOp>(op)) {
      SmallVector<rtl::ModulePortInfo> ports = module.getPorts();
      auto nameAttr = module.getNameAttr();
      auto newModule =
          builder.create<rtl::RTLModuleOp>(module.getLoc(), nameAttr, ports);
      auto outputOp = newModule.getBodyBlock()->getTerminator();
      OpBuilder innerBuilder(outputOp);
      SmallVector<Value, 8> outputs;
      // All output ports need values, use x
      for (auto &p : ports) {
        if (p.isOutput())
          outputs.push_back(
              innerBuilder.create<sv::ConstantXOp>(outputOp->getLoc(), p.type));
      }
      outputOp->setOperands(outputs);

      // Now update instances to drop parameters
      auto useRange = SymbolTable::getSymbolUses(module, getOperation());
      if (useRange)
        for (auto &user : *useRange)
          if (auto inst = dyn_cast<rtl::InstanceOp>(user.getUser()))
            inst->removeAttr("parameters");

      // Done with the old module.
      module.erase();
    }
}

std::unique_ptr<Pass> circt::sv::createRTLStubExternalModulesPass() {
  return std::make_unique<RTLStubExternalModulesPass>();
}
