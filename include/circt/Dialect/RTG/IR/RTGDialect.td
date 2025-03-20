//===- RTGDialect.td - RTG dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGDIALECT_TD
#define CIRCT_DIALECT_RTG_IR_RTGDIALECT_TD

include "mlir/IR/DialectBase.td"

def RTGDialect : Dialect {
  let name = "rtg";

  let summary = "types and operations for random test generation";
  let description = [{
    This dialect defines the `rtg` dialect, which is intended to encode
    constraints and randomization constructs for random test generation. Such a
    test can, for example, be a sequence of ISA instructions to test a CPU core.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let cppNamespace = "::circt::rtg";

  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // CIRCT_DIALECT_RTG_IR_RTGDIALECT_TD
