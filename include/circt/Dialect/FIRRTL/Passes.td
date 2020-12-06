//===-- Passes.td - FIRRTL pass definition file ------------*- tablegen -*-===//
//
// This file contains definitions for passes that work on the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_PASSES_TD
#define CIRCT_DIALECT_FIRRTL_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerFIRRTLTypes : Pass<"lower-firrtl-types", "firrtl::FModuleOp"> {
  let summary = "Lower FIRRTL types to ground types";
  let description = [{
    Lower firrtl.module port types to ground types.
  }];
  let constructor = "circt::firrtl::createLowerFIRRTLTypesPass()";
}

#endif // CIRCT_DIALECT_FIRRTL_PASSES_TD
