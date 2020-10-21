//===- RTL.td - RTL dialect definition ---------------------*- tablegen -*-===//
//
// This is the top level file for the RTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RTL_TD
#define RTL_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

def RTLDialect : Dialect {
  let name = "rtl";

  let summary = "Types and operations for rtl dialect";
  let description = [{
    This dialect defines the `rtl` dialect, which is intended to be a generic
    representation of RTL outside of a particular use-case.
  }];

  let cppNamespace = "::circt::rtl";
}

include "Types.td"

// Base class for the operation in this dialect.
class RTLOp<string mnemonic, list<OpTrait> traits = []> :
    Op<RTLDialect, mnemonic, traits>;

include "Combinatorial.td"
include "Statements.td"
include "Structure.td"

#endif // RTL_TD
