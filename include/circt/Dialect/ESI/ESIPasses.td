//===-- ESIPasses.td - ESI pass definition file ------------*- tablegen -*-===//
//
// This file contains definitions for passes that work on the ESI dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIPASSES_TD
#define CIRCT_DIALECT_ESI_ESIPASSES_TD

include "mlir/Pass/PassBase.td"

def LowerESIToPhysical: Pass<"lower-esi-to-physical", "mlir::ModuleOp"> {
  let summary = "Lower ESI abstract Ops to ESI physical ops.";
  let constructor = "circt::esi::createESIPhysicalLoweringPass()";
}

def LowerESIPorts: Pass<"lower-esi-ports", "mlir::ModuleOp"> {
  let summary = "Lower ESI input and/or output ports.";
  let constructor = "circt::esi::createESIPortLoweringPass()";
  let dependentDialects = ["circt::sv::SVDialect"];
}

def LowerESItoRTL: Pass<"lower-esi-to-rtl", "mlir::ModuleOp"> {
  let summary = "Lower ESI to RTL where possible and SV elsewhere.";
  let constructor = "circt::esi::createESItoRTLPass()";
  let dependentDialects = ["circt::rtl::RTLDialect"];
}

#endif // CIRCT_DIALECT_ESI_ESIPASSES_TD
