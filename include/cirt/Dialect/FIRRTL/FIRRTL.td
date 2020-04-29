//===- FIRRTL.td - FIRRTL dialect definition ---------------*- tablegen -*-===//
//
// This is the top level file for the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FIRRTL_TD
#define FIRRTL_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffects.td"

def FIRRTLDialect : Dialect {
  let name = "firrtl";

  let summary = "Types and operations for firrtl dialect";
  let description = [{
    This dialect defines the `firrtl` dialect, which is used to lower from
    Chisel code to Verilog.  For more information, see the
    [FIRRTL GitHub page](https://github.com/freechipsproject/firrtl).
  }];

  let cppNamespace = "firrtl";
}

// Base class for the operation in this dialect
class FIRRTLOp<string mnemonic, list<OpTrait> traits = []> :
    Op<FIRRTLDialect, mnemonic, traits>;

include "Types.td"
include "OpStructure.td"
include "OpDeclarations.td"
include "OpExpressions.td"
include "OpStatements.td"

#endif // FIRRTL_TD
