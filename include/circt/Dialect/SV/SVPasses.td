//===-- SVPasses.td - SV pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVPASSES
#define CIRCT_DIALECT_SV_SVPASSES

include "mlir/Pass/PassBase.td"

def RTLCleanup : Pass<"rtl-cleanup", "rtl::RTLModuleOp"> {
  let summary = "Cleanup transformations for operations in rtl.module bodies";
  let description = [{
      This pass merges sv.alwaysff operations with the same condition, sv.ifdef
      nodes with the same condition, and perform other cleanups for the IR.
      This is a good thing to run before Verilog emission.
  }];

  let constructor = "circt::sv::createRTLCleanupPass()";
}

def RTLStubExternalModules : Pass<"rtl-stub-external-modules", 
                                  "mlir::ModuleOp"> {
  let summary = "transform external rtl modules to empty rtl modules";
  let description = [{
      This pass creates empty module bodies for external modules.  This is
      useful for linting to eliminate missing file errors.
  }];

  let constructor = "circt::sv::createRTLStubExternalModulesPass()";
  let dependentDialects = ["circt::sv::SVDialect"];

}

#endif // CIRCT_DIALECT_SV_SVPASSES
