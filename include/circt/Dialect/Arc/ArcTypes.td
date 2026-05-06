//===- ArcTypes.td - Arc dialect types ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCTYPES_TD
#define CIRCT_DIALECT_ARC_ARCTYPES_TD

include "circt/Dialect/Arc/ArcDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "circt/Dialect/HW/HWTypeInterfaces.td"

class ArcTypeDef<string name, list<Trait> traits = []> : TypeDef<ArcDialect, name, traits> { }

def StateType : ArcTypeDef<"State"> {
  let mnemonic = "state";
  let parameters = (ins "::mlir::Type":$type);
  let assemblyFormat = "`<` $type `>`";
  let genVerifyDecl = 1;
  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$type), [{
      return $_get(type.getContext(), type);
    }]>
  ];

  let extraClassDeclaration = [{
    unsigned getBitWidth();
    unsigned getByteWidth() { return (getBitWidth() + 7) / 8; }
  }];
}

def MemoryType : ArcTypeDef<"Memory"> {
  let mnemonic = "memory";
  let parameters = (ins "unsigned":$numWords,
                        "::mlir::IntegerType":$wordType,
                        "::mlir::IntegerType":$addressType);
  let assemblyFormat = "`<` $numWords `x` $wordType `,` $addressType `>`";

  let extraClassDeclaration = [{
    unsigned getStride();
  }];
}

def StorageType : ArcTypeDef<"Storage"> {
  let mnemonic = "storage";
  let parameters = (ins OptionalParameter<"unsigned">:$size);
  let assemblyFormat = "(`<` $size^ `>`)?";
}

def SimModelInstance : ArcTypeDef<"SimModelInstance"> {
  let mnemonic = "sim.instance";
  let parameters = (ins "mlir::FlatSymbolRefAttr":$model);
  let assemblyFormat = "`<` $model `>`";
}

def ArrayRefType : ArcTypeDef<"ArrayRef", [
  DeclareTypeInterfaceMethods<BitWidthTypeInterface>
]> {
  let summary = "array reference";
  let description = [{
    An array reference is a reference (pointer) to an array of values.
  }];
  let mnemonic = "arrayref";
  let parameters = (ins "::mlir::Type":$elementType, "::mlir::IntegerAttr":$sizeAttr);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType, "size_t":$size), [{
      auto *ctx = elementType.getContext();
      auto intType = mlir::IntegerType::get(ctx, 64);
      auto sizeAttr = mlir::IntegerAttr::get(intType, size);
      return $_get(ctx, elementType, sizeAttr);
    }]>
  ];

  let assemblyFormat = "`<` custom<ArcArrayRef>($sizeAttr, $elementType) `>`";

  let extraClassDeclaration = [{
    size_t getNumElements() const;

    std::optional<size_t> getByteWidth() const {
      std::optional<size_t> bitWidth = getBitWidth();
      if (!bitWidth.has_value()) {
        return std::nullopt;
      }
      return (*bitWidth + 7) / 8;
    }
  }];
}

#endif // CIRCT_DIALECT_ARC_ARCTYPES_TD
