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

def DictType : RTGTypeDef<"Dict"> {
  let summary = "";
  let description = [{
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::StringAttr", "dict entry names">:$entryNames,
    ArrayRefParameter<"mlir::Type", "dict entry types">:$entryTypes);

  let mnemonic = "dict";

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def MemoryType : RTGTypeDef<"Memory"> {
  let summary = "";
  let description = [{
    A memory is a resource provided by the target. The test can only declare a
    memory that is a subset or an alias to a memory provided by the target.
    Each memory has its size, adress bitwidth, and alignment specified as an
    attribute. Additionally, memories have an implicit field storing the base
    address.
    Further information about the memory cannot be specified in the memory type
    or its runtime value directly, but it can be wrapped inside a dictionary.
    TODO: what about the memory slot bitwidth?
  }];

  let parameters = (ins "uint64_t":$size, "uint64_t":$addr_bitwidth, "uint64_t":$alignment);

  let mnemonic = "memory";
  let assemblyFormat = "`<` `size` $size `,` `addr_width` $addr_bitwidth `,` `align` $alignment `>`";
}

#endif // CIRCT_DIALECT_RTG_IR_RTGTYPES_TD
