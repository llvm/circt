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
include "circt/Dialect/RTG/IR/RTGInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class RTGTypeDef<string name, list<Trait> traits = []> :
  TypeDef<RTGDialect, name, traits> { }

def SnippetType : RTGTypeDef<"Snippet"> {
  let summary = "";
  let description = [{
  }];

  let mnemonic = "snippet";
  let assemblyFormat = "";
}

def ResourceType : RTGTypeDef<"Resource", [
  DeclareTypeInterfaceMethods<ResourceTypeInterface>
]> {
  let summary = "";
  let description = [{
  }];

  let parameters = (ins 
    ArrayRefParameter<"::circt::rtg::ResourceTypeInterface", "set elements">:$elements);

  let mnemonic = "resource";
  let assemblyFormat = "`<` $elements `>`";
}

def ResourceSetType : RTGTypeDef<"ResourceSet"> {
  let summary = "";
  let description = [{
  }];

  let parameters = (ins 
    ArrayRefParameter<"::circt::rtg::ResourceTypeInterface", "set elements">:$elements);

  let mnemonic = "resource_set";
  let assemblyFormat = "`<` $elements `>`";
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
    ArrayRefParameter<"::circt::rtg::ResourceTypeInterface", "set elements">:$elements);
  let assemblyFormat = "`<` $elements `>`";
}

def AnyResourceType : Type<CPred<"llvm::isa<::circt::rtg::ResourceTypeInterface>($_self)">, "any resource type">;
def AnyResourceSetType : AnyTypeOf<[ResourceSetType, ContextResourceSetType]>;

#endif // CIRCT_DIALECT_RTG_IR_RTGTYPES_TD
