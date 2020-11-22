//===-- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for all dialect conversions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_PASSES_TD
#define CIRCT_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// FIRRTLToRTL
//===----------------------------------------------------------------------===//

def LowerFIRRTLToRTLModule
  : Pass<"lower-firrtl-to-rtl-module", "mlir::ModuleOp"> {
  let summary = "Lower FIRRTL to RTL Modules";
  let description = [{
    Lower firrtl.module's to rtl.module.
  }];
  let constructor = "circt::firrtl::createLowerFIRRTLToRTLModulePass()";
  let dependentDialects = ["rtl::RTLDialect"];
}

def LowerFIRRTLToRTL : Pass<"lower-firrtl-to-rtl", "rtl::RTLModuleOp"> {
  let summary = "Lower FIRRTL to RTL";
  let description = [{
    Lower the contents of an firrtl.module to the RTL dialect.
  }];
  let constructor = "circt::firrtl::createLowerFIRRTLToRTLPass()";
  let dependentDialects = ["rtl::RTLDialect", "sv::SVDialect"];
}

#endif // CIRCT_CONVERSION_PASSES_TD
