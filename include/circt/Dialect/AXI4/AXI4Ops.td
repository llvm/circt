//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4OPS_TD
#define CIRCT_DIALECT_AXI4_AXI4OPS_TD

include "circt/Dialect/AXI4/AXI4Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class AXI4Op<string mnemonic, list<Trait> traits = []> :
    Op<AXI4Dialect, mnemonic, traits>;

// Operations will be added here.

#endif // CIRCT_DIALECT_AXI4_AXI4OPS_TD
