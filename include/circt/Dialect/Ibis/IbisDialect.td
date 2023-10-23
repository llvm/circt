//===- IbisDialect.td - Ibis dialect definition ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_DIALECT_TD
#define CIRCT_DIALECT_IBIS_DIALECT_TD

include "mlir/IR/OpBase.td"

def IbisDialect : Dialect {
  let name = "ibis";
  let cppNamespace = "::circt::ibis";

  let summary = "Types and operations for Ibis dialect";
  let description = [{
    The `ibis` dialect is intended to support porting and eventual open sourcing
    of an internal hardware development language.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  // Opt-out of properties for now, must migrate by LLVM 19.  #5273.
  let usePropertiesForAttributes = 0;

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];

  // Needed for ibis.pipeline_header
  let dependentDialects = ["seq::SeqDialect", "hw::HWDialect"];
}

#endif // CIRCT_DIALECT_IBIS_DIALECT_TD
