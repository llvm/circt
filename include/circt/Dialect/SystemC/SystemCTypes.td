//===- SystemCTypes.td - SystemC type definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types for the SystemC dialect. Includable in other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES
#define CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES

include "circt/Dialect/SystemC/SystemCDialect.td"

class SystemCType<Pred condition, string description, string cppClassName>
  : DialectType<SystemCDialect, condition, description, cppClassName>;

// A handle to refer to circt::systemc::FuncHandleType in ODS.
def FuncHandleType : SystemCType<
    CPred<"::circt::hw::type_isa<circt::systemc::FuncHandleType>($_self)">,
    "FuncHandleType",
    "::circt::hw::TypeAliasOr<circt::systemc::FuncHandleType>">,
  BuildableType<"::circt::systemc::FuncHandleType::get($_builder.getContext())">;

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES
