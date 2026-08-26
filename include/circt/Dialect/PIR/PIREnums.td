//===- PIREnums.td - Enums for PIR dialect -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines PIR dialect specific enums.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PIR_PIRENUMS_TD
#define CIRCT_DIALECT_PIR_PIRENUMS_TD

include "mlir/IR/EnumAttr.td"

let cppNamespace = "circt::pir" in {

//----------------------------------------------------------------------------//
// Special definition for small enums
//----------------------------------------------------------------------------//

class I8EnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I8, sym, str, val>;

class I8EnumAttr<string name, string summary, list<I8EnumAttrCase> cases> :
    IntEnumAttr<I8, name, summary, cases> {
  let underlyingType = "uint8_t";
}

//----------------------------------------------------------------------------//
// Property Sampling Type (always vs. initial)
//----------------------------------------------------------------------------//

/// Always refers to properties that must hold in all samples of a trace
def Always : I8EnumAttrCase<"Always", 1, "always">;
/// Initial refers to proeprties that must only hold in the first step of a trace
def Initial : I8EnumAttrCase<"Initial", 0, "initial">;

def SamplingTypeAttr : I8EnumAttr<"SamplingType", "Sampling stategy for a PIR property",
                                   [Always, Initial]>;
} // namspace circt::PIR

#endif // CIRCT_DIALECT_PIR_PIRENUMS_TD
