//===- SSPDialect.td - SSP dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SSP_SSPDIALECT_TD
#define CIRCT_DIALECT_SSP_SSPDIALECT_TD

include "mlir/IR/OpBase.td"

def SSPDialect : Dialect {
  let name = "ssp";
  let summary = "Static scheduling problem instances and solutions.";
  let description = [{
    A dialect to abstractly represent instances and solutions of static
    scheduling problems.
  }];
  let cppNamespace = "::circt::ssp";
  let useDefaultAttributePrinterParser = true;
  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

#endif // CIRCT_DIALECT_SSP_SSPDIALECT_TD
