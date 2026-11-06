//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the passes for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4PASSES_TD
#define CIRCT_DIALECT_AXI4_AXI4PASSES_TD

include "mlir/Pass/PassBase.td"

def VerifyAXI4Networks : Pass<"verify-axi4-networks", "mlir::ModuleOp"> {
  let summary = "Globally verify AXI4 networks";
  let description = [{
    Verifies non-local properties of an AXI4 network:
    - Ensures every `!axi4.port` value has at most one consumer, and that ports
      do not connect endpoints in different clock or reset domains.
    - Ensures a network occupies a single clock and reset domain.
    - Warns where an endpoint can handle fewer requests than the port reaching
      it can have concurrently outstanding. This isn't a correctness condition,
      but flags bottlenecks that might limit throughput.
  }];

  let dependentDialects = ["circt::axi4::AXI4Dialect"];
}

#endif // CIRCT_DIALECT_AXI4_AXI4PASSES_TD
