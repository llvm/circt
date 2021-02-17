//===- Comb.td - Comb dialect definition --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Comb dialect.
//
//===----------------------------------------------------------------------===//

#ifndef COMB_TD
#define COMB_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def CombDialect : Dialect {
  let name = "comb";

  let summary = "Types and operations for comb dialect";
  let description = [{
    This dialect defines the `comb` dialect, which is intended to be a generic
    representation of combinational logic outside of a particular use-case.
  }];

  let cppNamespace = "::circt::comb";
}

// TODO: eliminate the dependence on the RTL dialect here.

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def RTLIntegerType : Type<CPred<"rtl::isRTLIntegerType($_self)">,
                          "an integer bitvector of one or more bits",
                          "::mlir::IntegerType">;

// Type constraint that indicates that an operand/result may only be a valid
// non-directional type.
def RTLNonInOutType : Type<CPred<"!rtl::hasRTLInOutType($_self)">,
                           "a type without inout">;

// Base class for the operation in this dialect.
class CombOp<string mnemonic, list<OpTrait> traits = []> :
    Op<CombDialect, mnemonic, traits>;

include "Combinational.td"

#endif // COMB_TD
