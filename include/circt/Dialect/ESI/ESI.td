//===- ESI.td - ESI dialect top level tblgen ---------------*- tablegen -*-===//
//
// Top level tablegen file for the ESI dialect
//
//===----------------------------------------------------------------------===//


#ifndef ESI_TD
#define ESI_TD

include "mlir/IR/OpBase.td"

def ESI_Dialect : Dialect {
    let name = "esi";
    let cppNamespace = "::circt::esi";
}

include "ESITypes.td"
include "ESIOps.td"

#endif // ESI_TD
