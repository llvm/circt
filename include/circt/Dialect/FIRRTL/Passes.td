//===-- Passes.td - FIRRTL pass definition file ------------*- tablegen -*-===//
//
// This file contains definitions for passes that work on the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_PASSES_TD
#define CIRCT_DIALECT_FIRRTL_PASSES_TD

include "mlir/Pass/PassBase.td"

/// Lower firrtl.module port types to ground types.
def LowerFIRRTLTypes : Pass<"lower-firrtl-types", "firrtl::FModuleOp"> {
  let summary = "Lower FIRRTL types to ground types";
  let constructor = "circt::firrtl::createLowerFIRRTLTypesPass()";
}

/// Lower firrtl.module's to rtl.module.
def LowerFIRRTLToRTLModule
  : Pass<"lower-firrtl-to-rtl-module", "ModuleOp"> {
  let summary = "Lower FIRRTL to RTL Modules";
  let constructor = "circt::firrtl::createLowerFIRRTLToRTLModulePass()";
}

/// Lower the contents of an firrtl.module to the RTL dialect.
def LowerFIRRTLToRTL : Pass<"lower-firrtl-to-rtl", "rtl::RTLModuleOp"> {
  let summary = "Lower FIRRTL to RTL";
  let constructor = "circt::firrtl::createLowerFIRRTLToRTLPass()";
}

#endif // CIRCT_DIALECT_FIRRTL_PASSES_TD
