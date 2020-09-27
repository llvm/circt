//===- SV.td - SystemVerilog dialect definition ------------*- tablegen -*-===//
//
// This is the top level file for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SV_TD
#define SV_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SVDialect : Dialect {
  let name = "sv";

  let summary = "Types and operations for SV dialect";
  let description = [{
    This dialect defines the `sv` dialect, which represents various
    SystemVerilog-specific constructs in an AST-like representation.
  }];

  let cppNamespace = "::circt::sv";
}

// Base class for the operation in this dialect.
class SVOp<string mnemonic, list<OpTrait> traits = []> :
    Op<SVDialect, mnemonic, traits>;

include "Expressions.td"
include "Statements.td"

#endif // SV_TD
