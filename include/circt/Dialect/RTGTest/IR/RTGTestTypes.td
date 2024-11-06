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
include "mlir/IR/AttrTypeBase.td"

class RTGTestTypeDef<string name, list<Trait> traits = []> 
  : TypeDef<RTGTestDialect, name, traits> { }


#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTESTTYPES_TD
