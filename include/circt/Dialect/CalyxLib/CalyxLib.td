//===- CalyxLib.td - CalyxLib Definitions ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CALYXLIB_DIALECT_TD
#define CIRCT_CALYXLIB_DIALECT_TD

include "mlir/IR/DialectBase.td"

def CalyxLib_Dialect : Dialect {
  let name = "calyxlib";
  let cppNamespace = "::circt::calyxlib";
  let summary = "Operator library for passes that lower to Calyx";

  let extraClassDeclaration = [{
    /// Register all SSP attributes.
    void registerAttributes();
  }];
}

include "circt/Dialect/CalyxLib/CalyxLibAttributes.td"
include "circt/Dialect/CalyxLib/CalyxLibOps.td"

#endif // CIRCT_CALYXLIB_DIALECT_TD
