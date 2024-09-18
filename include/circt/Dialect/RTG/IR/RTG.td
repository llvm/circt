//===- RTGDialect.td - RTG dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGDIALECT_TD
#define CIRCT_DIALECT_RTG_IR_RTGDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def RTGDialect : Dialect {
  let name = "rtg";

  let summary = "types and operations for random test generation";
  let description = [{
    This dialect defines the `rtg` dialect, which is intended to encode
    constraints and randomization constructs for random test generation. Such a
    test can, for example, be a sequence of ISA instructions to test a CPU core.
  }];

  let cppNamespace = "::circt::rtg";
}

include "circt/Dialect/RTG/IR/RTGOps.td"

#endif // CIRCT_DIALECT_RTG_IR_RTGDIALECT_TD
