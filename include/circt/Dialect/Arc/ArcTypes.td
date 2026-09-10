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
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"
include "circt/Dialect/HW/HWTypeInterfaces.td"

class ArcTypeDef<string name, list<Trait> traits = []> :
  TypeDef<ArcDialect, name, traits> { }

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

def CoroutinePCType : ArcTypeDef<"CoroutinePC"> {
  let summary = "Program counter of a coroutine";
  let description = [{
    An opaque program counter value indicating where to resume execution of a
    coroutine. The concrete size is determined by lowering.
  }];
  let mnemonic = "coroutine_pc";
  let parameters = (ins "mlir::FlatSymbolRefAttr":$coroutine);
  let assemblyFormat = "`<` $coroutine `>`";
}

def CoroutineStateType : ArcTypeDef<"CoroutineState"> {
  let summary = "State persisted between re-entries into a coroutine";
  let description = [{
    An opaque value representing the local state within a coroutine that
    persists between re-entries into that coroutine. The concrete layout is
    determined by lowering and is likely a union of structs, with each struct
    corresponding to a unique resumption block in the coroutine.
  }];
  let mnemonic = "coroutine_state";
  let parameters = (ins "mlir::FlatSymbolRefAttr":$coroutine);
  let assemblyFormat = "`<` $coroutine `>`";
}

def ArrayRefType : ArcTypeDef<"ArrayRef", [
  DeclareTypeInterfaceMethods<BitWidthTypeInterface>,
  DeclareTypeInterfaceMethods<ShapedTypeInterface>
]> {
  let summary = "array reference";
  let description = [{
    An array reference is a reference (pointer) to an array of values.
  }];
  let mnemonic = "arrayref";
  let parameters = (ins
    "::mlir::Type":$elementType,
    "uint64_t":$size
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType, "uint64_t":$size), [{
      auto *ctx = elementType.getContext();
      return $_get(ctx, elementType, size);
    }]>
  ];

  let assemblyFormat = "`<` $size `` custom<XInDimList>($elementType) `>`";

  let genVerifyDecl = 1;

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
