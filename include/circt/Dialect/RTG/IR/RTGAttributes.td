//===- RTGAttributes.td - RTG attributes -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTG attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_TD
#define CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_TD

include "circt/Dialect/RTG/IR/RTGInterfaces.td"
include "circt/Dialect/RTG/IR/RTGDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"

class RTGAttrDef<string name, list<Trait> traits = []> 
  : AttrDef<RTGDialect, name, traits>;

def DefaultContextAttr : RTGAttrDef<"DefaultContext", [
  DeclareAttrInterfaceMethods<ContextResourceAttrInterface>,
]> {
  let summary = "the default context of its type";
  let description = [{
    A target must specify a default value for each context resource type it
    uses. The tests matched against that target then start in that context
    initially and 'on_context' operations can be used to switch to another
    context within a test. Essentially, this attribute is used within tests to
    refer to the default context set in the target.
  }];

  let mnemonic = "default";
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";
}

def AnyContextAttr : RTGAttrDef<"AnyContext", [
  DeclareAttrInterfaceMethods<ContextResourceAttrInterface>,
]> {
  let summary = "any single context of its type";
  let description = [{
    This attribute can be used to refer to any context of its type.
  }];

  let mnemonic = "any_context";
  let parameters = (ins AttributeSelfTypeParameter<"">:$type);
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// Attributes for ISA targets
//===----------------------------------------------------------------------===//

class RTGISAAttrDef<string name, list<Trait> traits = []> 
  : RTGAttrDef<name, traits> { let mnemonic = "isa." # !tolower(name); }

def ImmediateAttr : RTGISAAttrDef<"Immediate", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>,
]> {
  let summary = "an ISA immediate value";
  let description = [{
    This represents an ISA immediate of arbitrary but fixed bit-width. The type
    of this attribute must always be an `ImmediateType` of matching bit-width.
  }];

  let parameters = (ins "llvm::APInt":$value);

  let hasCustomAssemblyFormat = true;
  let genStorageClass = false;
}

def VirtualRegisterConfigAttr : RTGISAAttrDef<"VirtualRegisterConfig", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>,
]> {
  let summary = "an allowed register configuration for a virtual register";

  let parameters = (ins ArrayRefParameter<"rtg::RegisterAttrInterface",
                                          "allowed registers">:$allowedRegs);

  let mnemonic = "virtual_register_config";
  let assemblyFormat = "`[` $allowedRegs `]`";

  let genVerifyDecl = true;
}

#endif // CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_TD
