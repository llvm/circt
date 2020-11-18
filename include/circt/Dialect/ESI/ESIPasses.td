//===-- ESIPasses.td - ESI pass definition file ------------*- tablegen -*-===//
//
// This file contains definitions for passes that work on the ESI dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIPASSES_TD
#define CIRCT_DIALECT_ESI_ESIPASSES_TD

include "mlir/Pass/PassBase.td"

/// Lower firrtl.module port types to ground types.
def LowerESIToRTL: Pass<"lower-esi-to-rtl", "mlir::ModuleOp"> {
  let summary = "Lower ESI to RTL where possible and SV elsewhere";
  let constructor = "circt::esi::createESILoweringPass()";
  let dependentDialects = ["circt::rtl::RTLDialect", "circt::sv::SVDialect"];
}

#endif // CIRCT_DIALECT_ESI_ESIPASSES_TD
