//===- APInt.td - APInt dialect definition -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the APInt dialect.
//
//===----------------------------------------------------------------------===//

#ifndef APINT_TD
#define APINT_TD

include "mlir/IR/OpBase.td"

def APIntDialect : Dialect {
  let name = "apint";

  let summary = "Types and operations for apint dialect";
  let description = [{
    This dialect defines the `apint` dialect, which is intended to represent
    lossless arbitrary precision integer arithmetics. This is achieved by
    adjusting the bit widths to prevent under- and overflows while considering
    necessary sign-extensions for operations with different signedness.
    Hence, an addition with operands of type `ui2` and `ui2` yields in a result
    type `ui3` whereas an addition of `si2` and `ui2` yields in `si4`.
  }];

  let cppNamespace = "::circt::apint";
  let extraClassDeclaration = [{
    /// Register all APInt types.
    void registerTypes();
  }];
}

// Base class for the operation in this dialect.
class APIntOp<string mnemonic, list<OpTrait> traits = []> :
    Op<APIntDialect, mnemonic, traits>;

include "circt/Dialect/APInt/APIntTypes.td"
include "circt/Dialect/APInt/APIntOps.td"

#endif // APINT_TD
