//===- Physical.td - Physical dialect definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Physical dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PHYSICAL_TD
#define PHYSICAL_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PhysicalDialect : Dialect {
  let name = "physical";

  let summary = "A dialect for physical design";
  let description = [{
    The `physical` design dialect houses stuff applicable to physical design
    -- placement and routing (PnR). By constraining the PnR tool, we make the
    tools job easier leading to more predictable outcomes. If done well,
    those outcomes improve on what the PnR engine does unconstrained.
  }];

  let cppNamespace = "::circt::physical";
}

// Base class for the operation in this dialect.
class PhysicalOp<string mnemonic, list<OpTrait> traits = []> :
    Op<PhysicalDialect, mnemonic, traits>;

#endif // PHYSICAL_TD
