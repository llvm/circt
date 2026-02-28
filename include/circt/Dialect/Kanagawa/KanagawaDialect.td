//===- KanagawaDialect.td - Kanagawa dialect definition ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_KANAGAWA_DIALECT_TD
#define CIRCT_DIALECT_KANAGAWA_DIALECT_TD

include "mlir/IR/OpBase.td"

def KanagawaDialect : Dialect {
  let name = "kanagawa";
  let cppNamespace = "::circt::kanagawa";

  let summary = "Types and operations for Kanagawa dialect";
  let description = [{
    The `kanagawa` dialect is intended to support porting and eventual open sourcing
    of an internal hardware development language.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];

  // Needed for kanagawa.pipeline_header
  let dependentDialects = ["seq::SeqDialect", "hw::HWDialect"];
}

#endif // CIRCT_DIALECT_KANAGAWA_DIALECT_TD
