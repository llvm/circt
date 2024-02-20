//===- HWEnums.td - Enums for HW dialect -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines HW dialect specific enums.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWENUM_TD
#define CIRCT_DIALECT_HW_HWENUM_TD

include "mlir/IR/EnumAttr.td"

let cppNamespace = "circt::hw" in {

//----------------------------------------------------------------------------//
// Parameter Expression Opcodes.
//----------------------------------------------------------------------------//

/// Fully Associative Expression Opcodes.
def PEO_Add : I32EnumAttrCase<"Add", 0, "add">;
def PEO_Mul : I32EnumAttrCase<"Mul", 1, "mul">;
def PEO_And : I32EnumAttrCase<"And", 2, "and">;
def PEO_Or  : I32EnumAttrCase<"Or",  3, "or">;
def PEO_Xor : I32EnumAttrCase<"Xor", 4, "xor">;

// Binary Expression Opcodes.
def PEO_Shl  : I32EnumAttrCase<"Shl" , 5, "shl">;
def PEO_ShrU : I32EnumAttrCase<"ShrU", 6, "shru">;
def PEO_ShrS : I32EnumAttrCase<"ShrS", 7, "shrs">;
def PEO_DivU : I32EnumAttrCase<"DivU", 8, "divu">;
def PEO_DivS : I32EnumAttrCase<"DivS", 9, "divs">;
def PEO_ModU : I32EnumAttrCase<"ModU",10, "modu">;
def PEO_ModS : I32EnumAttrCase<"ModS",11, "mods">;

// Unary Expression Opcodes.
def PEO_CLog2 : I32EnumAttrCase<"CLog2", 12, "clog2">;

// String manipulation Opcodes.
def PEO_StrConcat : I32EnumAttrCase<"StrConcat", 13, "str.concat">;

def PEOAttr  : I32EnumAttr<"PEO", "Parameter Expression Opcode",
                           [PEO_Add, PEO_Mul, PEO_And, PEO_Or, PEO_Xor,
                            PEO_Shl, PEO_ShrU, PEO_ShrS,
                            PEO_DivU, PEO_DivS, PEO_ModU, PEO_ModS,
                            PEO_CLog2, PEO_StrConcat]>;

//----------------------------------------------------------------------------//
// Edge behavior for trigger blocks.
//----------------------------------------------------------------------------//

/// AtPosEdge triggers on a rise from 0 to 1/X/Z, or X/Z to 1.
def AtPosEdge: I32EnumAttrCase<"AtPosEdge", 0, "posedge">;
/// AtNegEdge triggers on a drop from 1 to 0/X/Z, or X/Z to 0.
def AtNegEdge: I32EnumAttrCase<"AtNegEdge", 1, "negedge">;
/// AtEdge(v) is syntactic sugar for "AtPosEdge(v) or AtNegEdge(v)".
def AtEdge   : I32EnumAttrCase<"AtEdge", 2, "edge">;

def EventControlAttr : I32EnumAttr<"EventControl", "edge control trigger",
                                   [AtPosEdge, AtNegEdge, AtEdge]>;

}

#endif // CIRCT_DIALECT_HW_HWENUM_TD
