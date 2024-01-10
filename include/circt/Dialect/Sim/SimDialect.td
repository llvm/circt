//===- SimDialect.td - Sim dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains the SimDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMDIALECT
#define CIRCT_DIALECT_SIM_SIMDIALECT

def SimDialect : Dialect {
  let name = "sim";
  let cppNamespace = "::circt::sim";

  let summary = "Types and operations for the `sim` dialect";
  let description = [{
    The `sim` dialect is intented to model simulator-specific operations.
  }];

  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
}

#endif // CIRCT_DIALECT_SIM_SIMDIALECT
