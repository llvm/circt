//===- OpLib.td - OpLib Definitions ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_OPLIB_DIALECT_TD
#define CIRCT_OPLIB_DIALECT_TD

include "mlir/IR/DialectBase.td"

def OpLib_Dialect : Dialect {
  let name = "oplib";
  let cppNamespace = "::circt::oplib";
  let summary = "Operator library dialect";

  let useDefaultAttributePrinterParser = true;

  let extraClassDeclaration = [{
    /// Register all SSP attributes.
    void registerAttributes();
  }];
}

include "circt/Dialect/OpLib/OpLibAttributes.td"
include "circt/Dialect/OpLib/OpLibOps.td"

#endif // CIRCT_OPLIB_DIALECT_TD
