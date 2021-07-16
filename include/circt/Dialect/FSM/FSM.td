//===- FSM.td - FSM dialect definition file -------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_TD
#define CIRCT_DIALECT_FSM_TD

include "mlir/IR/OpBase.td"

def FSMDialect : Dialect {
  let name = "fsm";
  let summary = "Types and operations for FSM dialect";
  let description = [{
    This dialect defines the `fsm` dialect, which is intended to represent
    finite-state machines.
  }];

  let cppNamespace = "::circt::fsm";
}

// Base class for the operation in this dialect.
class FSMOp<string mnemonic, list<OpTrait> traits = []> :
    Op<FSMDialect, mnemonic, traits>;

include "FSMOps.td"

#endif // CIRCT_DIALECT_FSM_TD
