//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_LLHDDIALECT_TD
#define CIRCT_DIALECT_LLHD_LLHDDIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def LLHDDialect : Dialect {
  let name = "llhd";
  let cppNamespace = "::circt::llhd";
  let dependentDialects = ["circt::hw::HWDialect"];

  let description = [{
    A low-level hardware description dialect in MLIR.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register all LLHD types.
    void registerTypes();
    /// Register all LLHD attributes.
    void registerAttributes();
  }];
}

#endif // CIRCT_DIALECT_LLHD_LLHDDIALECT_TD
