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
    CPred<"hw::isHWIntegerType($_self)">,
    "an integer bitvector of one or more bits", "::mlir::IntegerType">;

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWValueType : DialectType<HWDialect,
    CPred<"hw::isHWValueType($_self)">, "a known primitive element">;

// Type constraint that indicates that an operand/result may only be a valid
// non-directional type.
def HWNonInOutType : DialectType<HWDialect,
    CPred<"!hw::hasHWInOutType($_self)">, "a type without inout">;

def InOutType : DialectType<HWDialect,
    CPred<"hw::type_isa<hw::InOutType>($_self)">, "InOutType", "InOutType">;

// A handle to refer to hw::ArrayType in ODS.
def ArrayType : DialectType<HWDialect,
    CPred<"hw::type_isa<hw::ArrayType>($_self)">, "an ArrayType",
    "TypeAliasOr<ArrayType>">;

// A handle to refer to hw::StructType in ODS.
def StructType : DialectType<HWDialect,
    CPred<"hw::type_isa<hw::StructType>($_self)">, "a StructType",
    "TypeAliasOr<StructType>">;

// A handle to refer to hw::UnionType in ODS.
def UnionType : DialectType<HWDialect,
    CPred<"hw::type_isa<hw::UnionType>($_self)">, "a UnionType",
    "TypeAliasOr<UnionType>">;

#endif // CIRCT_DIALECT_HW_HWTYPES
