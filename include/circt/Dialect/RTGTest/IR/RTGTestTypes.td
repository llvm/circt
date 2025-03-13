//===- RTGTestTypes.td - RTGTest types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTGTest types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTGTEST_IR_RTGTESTTYPES_TD
#define CIRCT_DIALECT_RTGTEST_IR_RTGTESTTYPES_TD

include "circt/Dialect/RTGTest/IR/RTGTestDialect.td"
include "circt/Dialect/RTG/IR/RTGInterfaces.td"
include "circt/Dialect/RTG/IR/RTGTypes.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class RTGTestTypeDef<string name, list<Trait> traits = []> 
  : TypeDef<RTGTestDialect, name, traits>;

def CPUType : RTGTestTypeDef<"CPU", [ContextResourceTypeInterface]> {
  let summary = "handle to a specific CPU";
  let description = [{
    This type implements a specific context resource to test RTG operations
    taking context resources as operands (such as `on_context`) and other things
    requiring a concrete instance of a `ContextResourceTypeInterface`.
  }];

  let mnemonic = "cpu";
  let assemblyFormat = "";
}

def IntegerRegisterType : TypeDef<RTGTestDialect, "IntegerRegister", [
  RegisterTypeInterface
]> {
  let summary = "represents an integer register";

  let mnemonic = "ireg";
  let assemblyFormat = "";
}

def RegisterType : AnyTypeOf<[IntegerRegisterType]>;

#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTESTTYPES_TD
