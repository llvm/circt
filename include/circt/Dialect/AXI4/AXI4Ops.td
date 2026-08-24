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
include "circt/Dialect/AXI4/AXI4Types.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class AXI4Op<string mnemonic, list<Trait> traits = []> :
    Op<AXI4Dialect, mnemonic, traits>;

// Constrains each `!axi4.port` result to at most one use
def PortResultsAtMostOneUse : NativeOpTrait<"PortResultsAtMostOneUse"> {
  let cppNamespace = "::circt::axi4::OpTrait";
}

def AbstractManagerOp : AXI4Op<"abstract_manager", [PortResultsAtMostOneUse]> {
  let summary = "A stand-alone AXI4 manager endpoint";
  let description = [{
    Models a manager endpoint with no concrete implementation, for abstract
    network modelling. Returns the `!axi4.port` it drives.

    Example:
    ```mlir
    %mgr = axi4.abstract_manager %clk, %rst_ni : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset);
  let results = (outs PortType:$port);

  let assemblyFormat =
      "$clock `,` $reset attr-dict `:` qualified(type($port))";
}

def AbstractSubordinateOp : AXI4Op<"abstract_subordinate"> {
  let summary = "A stand-alone AXI4 subordinate endpoint";
  let description = [{
    Models a subordinate endpoint with no concrete implementation, for abstract
    network modelling. Takes the upstream `!axi4.port` it responds to.

    Example:
    ```mlir
    axi4.abstract_subordinate %clk, %rst_ni, %mgr : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$upstream);

  let assemblyFormat =
      "$clock `,` $reset `,` $upstream attr-dict `:` qualified(type($upstream))";
}

#endif // CIRCT_DIALECT_AXI4_AXI4OPS_TD
