//===- MSFT.td - MSFT dialect definition -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the MSFT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MSFT_TD
#define MSFT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MSFTDialect : Dialect {
  let name = "msft";

  let summary = "Microsoft internal support dialect";
  let description = [{
    Umbrella dialect for everything needed to support Microsoft development
    but not thoroughly discussed. Most (if not everything) in this dialect is
    a candidate for generalization and re-homing.
  }];

  let cppNamespace = "::circt::msft";
}

// Base class for the operation in this dialect.
class MSFTOp<string mnemonic, list<OpTrait> traits = []> :
    Op<MSFTDialect, mnemonic, traits>;

#endif // MSFT_TD
