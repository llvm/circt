//===- LLHDDialect.td - LLHD dialect definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_LLHDDIALECT_TD
#define CIRCT_DIALECT_LLHD_IR_LLHDDIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LLHD dialect definition
//===----------------------------------------------------------------------===//

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

class LLHDOp<string mnemonic, list<Trait> traits = []>
    : Op<LLHDDialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_LLHD_IR_LLHDDIALECT_TD
