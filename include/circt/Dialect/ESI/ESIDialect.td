//===- ESIDialect.td - ESI dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_DIALECT_TD
#define CIRCT_DIALECT_ESI_DIALECT_TD

include "mlir/IR/OpBase.td"

def ESI_Dialect : Dialect {
  let name = "esi";
  let cppNamespace = "::circt::esi";

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register all ESI types.
    void registerTypes();
  }];
}

class ESI_Op<string mnemonic, list<Trait> traits = []> :
    Op<ESI_Dialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_ESI_DIALECT_TD
