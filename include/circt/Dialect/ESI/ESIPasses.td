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
  let constructor = "circt::esi::createESILoweringPass()";
}

#endif // CIRCT_DIALECT_ESI_ESIPASSES_TD
