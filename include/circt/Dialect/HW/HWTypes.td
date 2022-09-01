//===- HWTypes.td - HW data type definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data types for the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWTYPES
#define CIRCT_DIALECT_HW_HWTYPES

include "circt/Dialect/HW/HWDialect.td"

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWIntegerType : DialectType<HWDialect,
    CPred<"::circt::hw::isHWIntegerType($_self)">,
    "an integer bitvector of one or more bits", "::mlir::IntegerType">;

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWValueType : DialectType<HWDialect,
    CPred<"::circt::hw::isHWValueType($_self)">, "a known primitive element">;

// Type constraint that indicates that an operand/result may only be a valid
// non-directional type.
def HWNonInOutType : DialectType<HWDialect,
    CPred<"!circt::hw::hasHWInOutType($_self)">, "a type without inout">;

def InOutType : DialectType<HWDialect,
    CPred<"::circt::hw::type_isa<circt::hw::InOutType>($_self)">,
          "InOutType", "InOutType">;

// A handle to refer to circt::hw::ArrayType in ODS.
def ArrayType : DialectType<HWDialect,
    CPred<"::circt::hw::type_isa<circt::hw::ArrayType>($_self)">,
          "an ArrayType", "::circt::hw::TypeAliasOr<circt::hw::ArrayType>">;

// A handle to refer to circt::hw::StructType in ODS.
def StructType : DialectType<HWDialect,
    CPred<"::circt::hw::type_isa<circt::hw::StructType>($_self)">,
          "a StructType", "::circt::hw::TypeAliasOr<circt::hw::StructType>">;

// A handle to refer to circt::hw::UnionType in ODS.
def UnionType : DialectType<HWDialect,
    CPred<"::circt::hw::type_isa<circt::hw::UnionType>($_self)">,
          "a UnionType", "::circt::hw::TypeAliasOr<hw::UnionType>">;

// A handle to refer to circt::hw::EnumType in ODS.
def EnumType : DialectType<HWDialect,
    CPred<"::circt::hw::type_isa<circt::hw::EnumType>($_self)">,
          "a EnumType", "::circt::hw::TypeAliasOr<circt::hw::EnumType>">;

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

/// Points to a name within a module.
def HWInnerRefAttr : Attr<
  CPred<"$_self.isa<::circt::hw::InnerRefAttr>()">,
        "name reference attribute"> {
  let returnType = "::circt::hw::InnerRefAttr";
  let storageType = "::circt::hw::InnerRefAttr";
  let convertFromStorage = "$_self";
}

/// A flat symbol reference or a reference to a name within a module.
def NameRefAttr : Attr<
  CPred<"$_self.isa<::mlir::FlatSymbolRefAttr, ::circt::hw::InnerRefAttr>()">,
  "name reference attribute">;

// Like a FlatSymbolRefArrayAttr, but can also refer to names inside modules.
def NameRefArrayAttr : TypedArrayAttrBase<NameRefAttr,
  "name reference array attribute"> {
  let constBuilderCall = "$_builder.getArrayAttr($0)";
}

#endif // CIRCT_DIALECT_HW_HWTYPES
