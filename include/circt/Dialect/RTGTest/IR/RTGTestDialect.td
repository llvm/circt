//===- RTGTestDialect.td - RTG Test dialect definition -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RTG Test dialect. This dialect provides a minimal
// set of operations to test the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTGTEST_IR_RTGTESTDIALECT_TD
#define CIRCT_DIALECT_RTGTEST_IR_RTGTESTDIALECT_TD

include "mlir/IR/DialectBase.td"

def RTGTestDialect : Dialect {
  let name = "rtgtest";

  let summary = "types and operations for random test generation testing";
  let description = [{
    This dialect defines the `rtgtest` dialect, which provides a set of 
    operation definitions to test the RTG dialect.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let hasConstantMaterializer = 1;

  let dependentDialects = ["::circt::rtg::RTGDialect"];

  let cppNamespace = "::circt::rtgtest";

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTESTDIALECT_TD
