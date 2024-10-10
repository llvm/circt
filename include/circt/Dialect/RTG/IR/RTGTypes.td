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

def SequenceType : RTGTypeDef<"Sequence"> {
  let summary = "";
  let description = [{
  }];

  let mnemonic = "sequence";
  let parameters = (ins
    OptionalArrayRefParameter<"::mlir::Type",
                      "types of the sequence inputs">:$argTypes);
  let assemblyFormat = "(`<` $argTypes^ `>`)?";
}

def ResourceType : RTGTypeDef<"Resource"> {
  let summary = "";
  let description = [{
  }];

  let mnemonic = "resource";
  let assemblyFormat = "";
}

def ResourceSetType : RTGTypeDef<"ResourceSet"> {
  let summary = "";
  let description = [{
  }];

  let mnemonic = "resource_set";
  let assemblyFormat = "";
}

def ContextResourceType : RTGTypeDef<"ContextResource"> {
  let summary = "";
  let description = [{
  }];

  let mnemonic = "context_resource";
  let assemblyFormat = "";
}

def ContextResourceSetType : RTGTypeDef<"ContextResourceSet"> {
  let summary = "";
  let description = [{
  }];

  let mnemonic = "context_resource_set";
  let parameters = (ins 
    "::circt::rtg::ResourceTypeInterface":$contextResourceType);
  let assemblyFormat = "`<` $contextResourceType `>`";
}

#endif // CIRCT_DIALECT_RTG_IR_RTGTYPES_TD
