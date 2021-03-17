//===-- RTLPasses.td - RTL pass definition file -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the RTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTL_RTLPASSES
#define CIRCT_DIALECT_RTL_RTLPASSES

include "mlir/Pass/PassBase.td"

def RTLGreyBoxer : Pass<"rtl-grey-boxer", "mlir::ModuleOp"> {
  let summary = "transform external rtl modules to empty rtl modules";
  let description = [{
      This pass creates empty module bodies for external modules.  This is useful for linting to eliminate missing file errors.
  }];

  let constructor = "circt::rtl::createRTLGreyBoxerPass()";
}

#endif // CIRCT_DIALECT_RTL_RTLPASSES
