//===- APIntTypes.td - APInt data type definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data types for the APInt dialect.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def APIntType : DialectType<APIntDialect,
    And<[
      CPred<"$_self.isa<::mlir::IntegerType>()">,
      CPred<"!$_self.isSignlessInteger()">
    ]>,
    "arbitrary precision integer with signedness semantics",
    "::mlir::IntegerType">;
