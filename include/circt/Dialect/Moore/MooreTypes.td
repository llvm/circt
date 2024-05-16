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
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class MooreTypeDef<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<MooreDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

class MooreType<Pred condition, string description, string cppClassName>
  : DialectType<MooreDialect, condition, description, cppClassName>;


//===----------------------------------------------------------------------===//
// Unit Types
//===----------------------------------------------------------------------===//

def VoidType : MooreTypeDef<"Void", [], "moore::PackedType"> {
  let mnemonic = "void";
  let summary = "the SystemVerilog `void` type";
}

def StringType : MooreTypeDef<"String", [], "moore::UnpackedType"> {
  let mnemonic = "string";
  let summary = "the SystemVerilog `string` type";
}

def ChandleType : MooreTypeDef<"Chandle", [], "moore::UnpackedType"> {
  let mnemonic = "chandle";
  let summary = "the SystemVerilog `chandle` type";
}

def EventType : MooreTypeDef<"Event", [], "moore::UnpackedType"> {
  let mnemonic = "event";
  let summary = "the SystemVerilog `event` type";
}


//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

def IntType : MooreTypeDef<"Int", [], "moore::PackedType"> {
  let typeName = "moore.int";
  let summary = "a simple bit vector type";
  let description = [{
    The `!moore.iN` and `!moore.lN` types represent a two-valued or four-valued
    simple bit vector of width `N`. The predefined SystemVerilog integer types
    map to this as follows:

    | Verilog    | Moore Dialect |
    |------------|---------------|
    | `bit`      | `!moore.i1`   |
    | `logic`    | `!moore.l1`   |
    | `reg`      | `!moore.l1`   |
    | `byte`     | `!moore.i8`   |
    | `shortint` | `!moore.i16`  |
    | `int`      | `!moore.i32`  |
    | `integer`  | `!moore.l32`  |
    | `longint`  | `!moore.i64`  |
    | `time`     | `!moore.l64`  |
  }];
  let parameters = (ins "unsigned":$width, "Domain":$domain);

  let extraClassDeclaration = [{
    /// Create a signless `bit [width-1:0]` type.
    static IntType getInt(MLIRContext *context, unsigned width) {
      return get(context, width, Domain::TwoValued);
    }

    /// Create a signless `logic [width-1:0]` type.
    static IntType getLogic(MLIRContext *context, unsigned width) {
      return get(context, width, Domain::FourValued);
    }
  }];
}


//===----------------------------------------------------------------------===//
// RealType
//===----------------------------------------------------------------------===//

def RealType : MooreTypeDef<"Real", [], "moore::UnpackedType"> {
  let mnemonic = "real";
  let summary = "a SystemVerilog real type";
  let description = [{
    This type represents the SystemVerilog real type. Since the Moore dialect
    does not fully handle real-valued expressions properly yet, we coalesce the
    `shortreal`, `real`, and `realtime` types in the SystemVerilgo standard to
    this common `!moore.real` type. The standard specifies these types to be of
    at least 32, 64, and 64 bits, respectively. The `!moore.real` type is 64
    bits wide.

    | Verilog     | Moore Dialect |
    |-------------|---------------|
    | `shortreal` | `!moore.real` |
    | `real`      | `!moore.real` |
    | `realtime`  | `!moore.real` |
  }];
}


//===----------------------------------------------------------------------===//
// Constraints
//===----------------------------------------------------------------------===//

def UnpackedType : MooreType<CPred<"llvm::isa<moore::UnpackedType>($_self)">,
  "unpacked type", "moore::UnpackedType">;

/// A simple bit vector type.
def SimpleBitVectorType : MooreType<CPred<[{
    llvm::isa<moore::IntType>($_self)
  }]>, "simple bit vector type", "moore::IntType">;

/// A single two or four-valued bit.
def AnySingleBitType : MooreType<CPred<[{
    llvm::isa<moore::IntType>($_self) &&
    llvm::cast<moore::IntType>($_self).getWidth() == 1
  }]>, "single bit type", "moore::IntType">;

/// A single two-valued bit.
def BitType : MooreType<CPred<[{
    llvm::isa<moore::IntType>($_self) &&
    llvm::cast<moore::IntType>($_self).getWidth() == 1 &&
    llvm::cast<moore::IntType>($_self).getDomain() == moore::Domain::TwoValued
  }]>, "`bit` type", "moore::IntType"> {
  let builderCall = [{
    IntType::getInt($_builder.getContext(), 1)
  }];
}

#endif // CIRCT_DIALECT_MOORE_MOORETYPES
