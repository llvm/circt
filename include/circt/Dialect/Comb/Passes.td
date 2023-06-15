//===-- Passes.td - Comb pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_COMB_PASSES_TD
#define CIRCT_DIALECT_COMB_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerComb : Pass<"lower-comb", "mlir::ModuleOp"> {
  let summary = "Lowers some comb ops";
  let description = [{
    Lowers truth tables.
  }];

  let options = [
    Option<"assignToWire", "assign-to-wire", "bool", "false",
      "Assign each lowered op to a wire">,
  ];

  let dependentDialects = ["circt::hw::HWDialect"];
}

#endif // CIRCT_DIALECT_COMB_PASSES_TD
