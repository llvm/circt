// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef ESI_TD
#define ESI_TD

include "mlir/IR/OpBase.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def ESI_Dialect : Dialect {
    let name = "esi";
    let cppNamespace = "::circt::esi";
}

include "ESITypes.td"
include "ESIOps.td"

#endif // ESI_TD
