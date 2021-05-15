//===- esi-tester.cpp - The ESI test driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program exercises some ESI functionality which is intended to be for API
// use only.
//
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace circt;
using namespace circt::esi;

/// This is a test pass for verifying FuncOp's eraseResult method.
struct TestESIModWrap
    : public mlir::PassWrapper<TestESIModWrap, OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto mlirMod = getOperation();
    auto b = mlir::OpBuilder::atBlockEnd(mlirMod.getBody());

    SmallVector<rtl::HWModuleOp, 8> mods;
    for (Operation *mod : mlirMod.getOps<rtl::HWModuleExternOp>()) {
      SmallVector<ESIPortValidReadyMapping, 32> liPorts;
      findValidReadySignals(mod, liPorts);
      if (!liPorts.empty())
        if (!buildESIWrapper(b, mod, liPorts))
          signalPassFailure();
    }
  }
};

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<comb::CombDialect, esi::ESIDialect, rtl::HWDialect>();

  mlir::PassRegistration<TestESIModWrap>(
      "test-mod-wrap", "Test the ESI find and wrap functionality");

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "CIRCT modular optimizer driver", registry,
                        /*preloadDialectsInContext=*/true));
}
