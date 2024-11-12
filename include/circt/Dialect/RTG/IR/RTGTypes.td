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

def ModeType : RTGTypeDef<"Mode"> {
  let summary = "";
  let description = [{
  }];

  let mnemonic = "mode";
  let assemblyFormat = "";
}

def ContextResourceType : RTGTypeDef<"ContextResource"> {
  let summary = "";
  let description = [{
  }];

  let mnemonic = "context_resource";
  let assemblyFormat = "";
}

def SetType : RTGTypeDef<"Set"> {
  let summary = "";
  let description = [{
  }];

  let parameters = (ins "::mlir::Type":$elementType);

  let mnemonic = "set";
  let assemblyFormat = "`<` $elementType `>`";
}

class SetTypeOf<Type elementType> : ContainerType<elementType, SetType.predicate, "llvm::cast<rtg::SetType>($_self).getElementType()", "">;

def BagType : RTGTypeDef<"Bag"> {
  let summary = "";
  let description = [{
  }];

  let parameters = (ins "::mlir::Type":$elementType);

  let mnemonic = "bag";
  let assemblyFormat = "`<` $elementType `>`";
}

class BagTypeOf<Type elementType> : ContainerType<elementType, BagType.predicate, "llvm::cast<rtg::BagType>($_self).getElementType()", "">;

def TargetType : RTGTypeDef<"Target"> {
  let summary = "";
  let description = [{
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::StringAttr", "target entry names">:$entryNames,
    ArrayRefParameter<"mlir::Type", "target entry types">:$entryTypes);

  let mnemonic = "target";

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif // CIRCT_DIALECT_RTG_IR_RTGTYPES_TD
