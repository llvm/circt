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

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWIntegerType : Type<CPred<"hw::isHWIntegerType($_self)">,
                         "an integer bitvector of one or more bits",
                         "::mlir::IntegerType">;

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWValueType : Type<CPred<"hw::isHWValueType($_self)">,
                       "a known primitive element">;

// Type constraint that indicates that an operand/result may only be a valid
// non-directional type.
def HWNonInOutType : Type<CPred<"!hw::hasHWInOutType($_self)">,
                          "a type without inout">;

// A handle to refer to hw::ArrayType in ODS.
def ArrayType : Type<CPred<"hw::type_isa<hw::ArrayType>($_self)">,
    "an ArrayType", "::circt::hw::TypeAliasOr<ArrayType>">;

// A handle to refer to hw::StructType in ODS.
def StructType : Type<CPred<"hw::type_isa<hw::StructType>($_self)">,
    "a StructType", "::circt::hw::TypeAliasOr<StructType>">;

// A handle to refer to hw::UnionType in ODS.
def UnionType : Type<CPred<"hw::type_isa<hw::UnionType>($_self)">,
    "a UnionType", "::circt::hw::TypeAliasOr<UnionType>">;

#endif // CIRCT_DIALECT_HW_HWTYPES
