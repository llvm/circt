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

#endif // CIRCT_DIALECT_RTG_IR_RTGATTRIBUTES_TD
