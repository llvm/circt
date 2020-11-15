//===- ESI.td - ESI dialect top level tblgen ---------------*- tablegen -*-===//
//
// Top level tablegen file for the ESI dialect
//
//===----------------------------------------------------------------------===//


#ifndef ESI_TD
#define ESI_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ESI_Dialect : Dialect {
  let name = "esi";
  let cppNamespace = "::circt::esi";
}

class ESI_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ESI_Dialect, mnemonic, traits>;

include "ESIPorts.td"
include "ESIOps.td"

#endif // ESI_TD
