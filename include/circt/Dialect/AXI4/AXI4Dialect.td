//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4DIALECT_TD
#define CIRCT_DIALECT_AXI4_AXI4DIALECT_TD

include "mlir/IR/OpBase.td"

def AXI4Dialect : Dialect {
  let name = "axi4";

  let summary = "Types and operations for the AXI4 dialect";
  let description = [{
    This dialect defines the `axi4` dialect, which models the AMBA AXI4 bus
    protocol.
  }];

  let cppNamespace = "::circt::axi4";
}

#endif // CIRCT_DIALECT_AXI4_AXI4DIALECT_TD
