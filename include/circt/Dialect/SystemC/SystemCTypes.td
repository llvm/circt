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

// A constraint for a mlir::FunctionType with zero inputs.
def NullaryFunctionConstraint : CPred<
  "::circt::hw::type_isa<mlir::FunctionType>($_self) && " #
  "::circt::hw::type_cast<mlir::FunctionType>($_self).getNumInputs() == 0">;

// A constraint for a mlir::FunctionType with zero results.
def VoidFunctionConstraint : CPred<
  "::circt::hw::type_isa<mlir::FunctionType>($_self) && " #
  "::circt::hw::type_cast<mlir::FunctionType>($_self).getNumResults() == 0">;

// A handle to refer to a mlir::FunctionType with zero inputs and results,
// i.e., a nullary void function.
def NullaryVoidFunctionType : SystemCType<
    And<[NullaryFunctionConstraint, VoidFunctionConstraint]>,
    "FunctionType with no inputs and results",
    "::circt::hw::TypeAliasOr<mlir::FunctionType>">,
  BuildableType<"mlir::FunctionType::get($_builder.getContext(), {}, {})">;

// A handle to refer to circt::systemc::InputType in ODS.
def InputType : SystemCType<
    CPred<"::circt::hw::type_isa<circt::systemc::InputType>($_self)">,
    "InputType",
    "::circt::hw::TypeAliasOr<circt::systemc::InputType>">;

// A handle to refer to circt::systemc::InOutType in ODS.
def InOutType : SystemCType<
    CPred<"::circt::hw::type_isa<circt::systemc::InOutType>($_self)">,
    "InOutType",
    "::circt::hw::TypeAliasOr<circt::systemc::InOutType>">;

// A handle to refer to circt::systemc::OutputType in ODS.
def OutputType : SystemCType<
    CPred<"::circt::hw::type_isa<circt::systemc::OutputType>($_self)">,
    "OutputType",
    "::circt::hw::TypeAliasOr<circt::systemc::OutputType>">;

// A handle to refer to circt::systemc::SignalType in ODS.
def SignalType : SystemCType<
    CPred<"::circt::hw::type_isa<circt::systemc::SignalType>($_self)">,
    "SignalType",
    "::circt::hw::TypeAliasOr<circt::systemc::SignalType>">;

#endif // CIRCT_DIALECT_SYSTEMC_SYSTEMCTYPES
