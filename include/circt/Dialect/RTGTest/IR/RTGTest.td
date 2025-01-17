//===- RTGTest.td - RTGTest top-level definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the RTGTest dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTGTEST_IR_RTGTEST_TD
#define CIRCT_DIALECT_RTGTEST_IR_RTGTEST_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

include "circt/Dialect/RTG/IR/RTGInterfaces.td"

include "circt/Dialect/RTGTest/IR/RTGTestDialect.td"
include "circt/Dialect/RTGTest/IR/RTGTestAttributes.td"
include "circt/Dialect/RTGTest/IR/RTGTestTypes.td"
include "circt/Dialect/RTGTest/IR/RTGTestAttributes.td"
include "circt/Dialect/RTGTest/IR/RTGTestOps.td"

#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTEST_TD
