//===- DebugInfo.cpp - Debug info analysis --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/DebugInfo.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "di"

using namespace mlir;
using namespace circt;

namespace circt {
namespace detail {

struct DebugInfoBuilder {
  DebugInfoBuilder(DebugInfo &di) : di(di) {}
  DebugInfo &di;

  void visitRoot(Operation *op);
  void visitModule(Operation *op, DIModule &module);

  DIModule *createModule() {
    return new (di.moduleAllocator.Allocate()) DIModule;
  }

  DIInstance *createInstance() {
    return new (di.instanceAllocator.Allocate()) DIInstance;
  }

  DIVariable *createVariable() {
    return new (di.variableAllocator.Allocate()) DIVariable;
  }

  DIModule &getOrCreateModule(StringAttr moduleName) {
    auto &slot = di.moduleNodes[moduleName];
    if (!slot) {
      slot = createModule();
      slot->name = moduleName;
    }
    return *slot;
  }
};

void DebugInfoBuilder::visitRoot(Operation *op) {
  op->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto moduleOp = dyn_cast<hw::HWModuleOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Collect DI for module " << moduleOp.getNameAttr() << "\n");
      auto &module = getOrCreateModule(moduleOp.getNameAttr());
      module.op = op;

      // Add variables for each of the ports.
      auto inputValues = moduleOp.getBody().getArguments();
      auto outputValues =
          moduleOp.getBodyBlock()->getTerminator()->getOperands();
      for (auto &port : moduleOp.getPortList()) {
        auto value = port.isOutput() ? outputValues[port.argNum]
                                     : inputValues[port.argNum];
        auto *var = createVariable();
        var->name = port.name;
        var->loc = port.loc;
        var->value = value;
        module.variables.push_back(var);
      }

      // Visit the ops in the module.
      visitModule(op, module);
      return WalkResult::skip();
    }

    if (auto moduleOp = dyn_cast<hw::HWModuleExternOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Collect DI for extern module "
                              << moduleOp.getNameAttr() << "\n");
      auto &module = getOrCreateModule(moduleOp.getNameAttr());
      module.op = op;
      module.isExtern = true;

      // Add variables for each of the ports.
      for (auto &port : moduleOp.getPortList()) {
        auto *var = createVariable();
        var->name = port.name;
        var->loc = port.loc;
        module.variables.push_back(var);
      }

      return WalkResult::skip();
    }

    return WalkResult::advance();
  });
}

void DebugInfoBuilder::visitModule(Operation *op, DIModule &module) {
  op->walk([&](Operation *op) {
    if (auto instOp = dyn_cast<hw::InstanceOp>(op)) {
      auto &childModule =
          getOrCreateModule(instOp.getModuleNameAttr().getAttr());
      auto *instance = createInstance();
      instance->name = instOp.getInstanceNameAttr();
      instance->op = instOp;
      instance->module = &childModule;
      module.instances.push_back(instance);

      // TODO: What do we do with the port assignments? These should be tracked
      // somewhere.
      return;
    }

    if (auto wireOp = dyn_cast<hw::WireOp>(op)) {
      auto *var = createVariable();
      var->name = wireOp.getNameAttr();
      var->loc = wireOp.getLoc();
      var->value = wireOp;
      module.variables.push_back(var);
      return;
    }
  });
}

} // namespace detail
} // namespace circt

DebugInfo::DebugInfo(Operation *op) : operation(op) {
  detail::DebugInfoBuilder(*this).visitRoot(op);
}
