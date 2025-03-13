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

class ImmediateAttrBase<int width> : RTGAttrDef<"Imm" # width, [
  DeclareAttrInterfaceMethods<TypedAttrInterface>,
]> {
  let summary = "represents a " # width # "-bit immediate value";

  let mnemonic = "imm" # width;
  let parameters = (ins "uint32_t":$value);

  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    llvm::APInt getAPInt() const {
      return llvm::APInt(}] # width # [{, getValue());
    }
  }];

  let extraClassDefinition = [{
    Type $cppClass::getType() const {
      return Imm}] # width # [{Type::get(getContext());
    }

    LogicalResult $cppClass::verify(
      ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
      uint32_t value) {

      if (32 - llvm::countl_zero(value) > }] # width # [{)
        return emitError() << "cannot represent " << value << " with }] #
          width # [{ bits";

      return success();
    }
  }];

  let genVerifyDecl = 1;
}


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
}

#endif // CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_TD
