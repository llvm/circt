#ifndef CIRCT_DIALECT_SIM_SEQPASSES
#define CIRCT_DIALECT_SIM_SEQPASSES

include "mlir/Pass/PassBase.td"

def ProceduralizeSim : Pass<"sim-proceduralize", "hw::HWModuleOp"> {
  let summary = "Transform non-procedural to procedural operations.";
  let description = [{Transform non-procedural simulation operations with clock
                      and enable to procedural operations wrapped in
                      a procedural region.}];
  let dependentDialects = ["circt::hw::HWDialect, circt::seq::SeqDialect, mlir::scf::SCFDialect"];
}

#endif // CIRCT_DIALECT_SIM_SEQPASSES