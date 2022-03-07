//===- MooreTypes.td - Moore type definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SystemVerilog types for the Moore dialect. Includable in other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOORETYPES
#define CIRCT_DIALECT_MOORE_MOORETYPES

include "circt/Dialect/Moore/MooreDialect.td"

//===----------------------------------------------------------------------===//
// Integer atom types
//===----------------------------------------------------------------------===//

def MooreIntType : DialectType<MooreDialect,
  CPred<"$_self.isa<moore::IntType>()">,
  "an SystemVerilog int", "::circt::moore::IntType">;

//===----------------------------------------------------------------------===//
// LValue / RValue predicates
//===----------------------------------------------------------------------===//

class MooreLValueTypeOf<list<Type> allowedTypes>
  : ContainerType<AnyTypeOf<allowedTypes>,
    CPred<"$_self.isa<moore::LValueType>()">,
    "$_self.cast<moore::LValueType>().getNestedType()", "Moore LValue type">;

def MooreLValueType : MooreLValueTypeOf<[AnyType]>;
def MooreRValueType : AnyTypeOf<[MooreIntType]>;

#endif // CIRCT_DIALECT_MOORE_MOORETYPES
