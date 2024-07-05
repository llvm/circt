#ifndef CIRCT_DIALECT_SIM_SEQPASSES
#define CIRCT_DIALECT_SIM_SEQPASSES

include "mlir/Pass/PassBase.td"

def ProceduralizeSim : Pass<"sim-proceduralize", "hw::HWModuleOp"> {
  let summary = "Create procedural regions for simulation operations";
  let dependentDialects = ["circt::hw::HWDialect, circt::seq::SeqDialect, mlir::scf::SCFDialect"];
}

#endif // CIRCT_DIALECT_SIM_SEQPASSES