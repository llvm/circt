//===- RTGTypes.td - RTG types -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTG types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGTYPES_TD
#define CIRCT_DIALECT_RTG_IR_RTGTYPES_TD

include "circt/Dialect/RTG/IR/RTGDialect.td"
include "mlir/IR/AttrTypeBase.td"

class RTGTypeDef<string name> : TypeDef<RTGDialect, name> { }

def SnippetType : RTGTypeDef<"Snippet"> {
  let summary = "";
  let description = [{
  }];

  let mnemonic = "snippet";
  let assemblyFormat = "";
}

def ResourceType : RTGTypeDef<"Resource"> {
  let summary = "";
  let description = [{
  }];

  let mnemonic = "resource";
  let assemblyFormat = "";
}

#endif // CIRCT_DIALECT_RTG_IR_RTGTYPES_TD
