//===- SimAttributes.td - Attributes for Sim dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SIM_SIMATTRIBUTES_TD
#define CIRCT_DIALECT_SIM_SIMATTRIBUTES_TD

include "circt/Dialect/Sim/SimDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def NeverTriggerAttr : AttrDef<SimDialect, "NeverTrigger",
  [TypedAttrInterface]> {
  let mnemonic = "never";
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";
  let genVerifyDecl = true;
}

#endif // CIRCT_DIALECT_SIM_SIMATTRIBUTES_TD
