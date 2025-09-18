//===- MooreTypes.td - Moore type definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SystemVerilog types for the Moore dialect. Includable in other dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOORETYPES
#define CIRCT_DIALECT_MOORE_MOORETYPES

include "circt/Dialect/Moore/MooreDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

class MooreTypeDef<string name, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<MooreDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

class MooreType<Pred condition, string description, string cppClassName>
  : DialectType<MooreDialect, condition, description, cppClassName>;


//===----------------------------------------------------------------------===//
// Unit Types
//===----------------------------------------------------------------------===//

def VoidType : MooreTypeDef<"Void", [], "moore::PackedType"> {
  let mnemonic = "void";
  let summary = "the SystemVerilog `void` type";
}

def StringType : MooreTypeDef<"String", [], "moore::UnpackedType"> {
  let mnemonic = "string";
  let summary = "the SystemVerilog `string` type";
}

def ChandleType : MooreTypeDef<"Chandle", [], "moore::UnpackedType"> {
  let mnemonic = "chandle";
  let summary = "the SystemVerilog `chandle` type";
}

def EventType : MooreTypeDef<"Event", [], "moore::UnpackedType"> {
  let mnemonic = "event";
  let summary = "the SystemVerilog `event` type";
}


//===----------------------------------------------------------------------===//
// TimeType
//===----------------------------------------------------------------------===//

def TimeType : MooreTypeDef<"Time", [], "moore::PackedType"> {
  let mnemonic = "time";
  let summary = "the SystemVerilog `time` and `realtime` type";
  let description = [{
    The `!moore.time` type represents a time value. Internally, the value is
    represented as femtoseconds in `i64`.

    Time values are a problematic part of the SystemVerilog language, since they
    are multiplied by a scaling factor given by the current scope's `timeunit`
    and then mapped to an 64 bit four-valued integer for `time`, or a real for
    `realtime`. This means that a time value flowing from one module with one
    timeunit to another module with a different timeunit will read as two
    distinct delay values in the two modules.

    To work around these issues, the Moore dialect maps all time values to
    femtoseconds and represents them as an integer value. This allows us to have
    time values exchanged between modules correctly, while locally casting from
    and to `integer` and `real` with the correct local scaling factor.
  }];
}


//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

def IntType : MooreTypeDef<"Int", [], "moore::PackedType"> {
  let typeName = "moore.int";
  let summary = "a simple bit vector type";
  let description = [{
    The `!moore.iN` and `!moore.lN` types represent a two-valued or four-valued
    simple bit vector of width `N`. The predefined SystemVerilog integer types
    map to this as follows:

    | Verilog    | Moore Dialect |
    |------------|---------------|
    | `bit`      | `!moore.i1`   |
    | `logic`    | `!moore.l1`   |
    | `reg`      | `!moore.l1`   |
    | `byte`     | `!moore.i8`   |
    | `shortint` | `!moore.i16`  |
    | `int`      | `!moore.i32`  |
    | `integer`  | `!moore.l32`  |
    | `longint`  | `!moore.i64`  |
    | `time`     | `!moore.time` |
  }];
  let parameters = (ins "unsigned":$width, "Domain":$domain);

  let extraClassDeclaration = [{
    /// Create a signless `bit [width-1:0]` type.
    static IntType getInt(MLIRContext *context, unsigned width) {
      return get(context, width, Domain::TwoValued);
    }

    /// Create a signless `logic [width-1:0]` type.
    static IntType getLogic(MLIRContext *context, unsigned width) {
      return get(context, width, Domain::FourValued);
    }

    /// Get a two-valued version of this integer.
    IntType getTwoValued() {
      if (getDomain() == Domain::TwoValued)
        return *this;
      return getInt(getContext(), getWidth());
    }

    /// Get a four-valued version of this integer.
    IntType getFourValued() {
      if (getDomain() == Domain::FourValued)
        return *this;
      return getLogic(getContext(), getWidth());
    }
  }];
}


//===----------------------------------------------------------------------===//
// RealType
//===----------------------------------------------------------------------===//

def RealType : MooreTypeDef<"Real", [], "moore::UnpackedType"> {
  let typeName = "moore.real";
  let summary = "a SystemVerilog real type";
  let parameters = (ins "RealWidth":$width);
  let description = [{
    This type represents the SystemVerilog real type. We coalesce the
    `f32`, `f64`, and `realtime` types in the SystemVerilog standard to
    this common `!moore.real` type. The standard specifies these types to be of
    at least 32, 64, and 64 bits, respectively.

    | Verilog     | Moore Dialect      |
    |-------------|--------------------|
    | `shortreal` | `!moore.f32`       |
    | `real`      | `!moore.f64`       |
    | `realtime`  | `!moore.time`      |
  }];
  let extraClassDeclaration = [{
    // Convenience helpers
    static RealType getShortReal(MLIRContext *ctx) { return get(ctx, RealWidth::f32); }
    static RealType getReal     (MLIRContext *ctx) { return get(ctx, RealWidth::f64); }
  }];
}


//===----------------------------------------------------------------------===//
// Arrays
//===----------------------------------------------------------------------===//

def ArrayType : MooreTypeDef<"Array", [], "moore::PackedType"> {
  let mnemonic = "array";
  let summary = "a packed array type";
  let description = [{
    A packed array with a fixed number of elements. This type represents packed
    range dimensions (`[a:b]`) in SystemVerilog.

    | Verilog   | Moore Dialect         |
    |-----------|-----------------------|
    | `T [3:0]` | `!moore.array<4 x T>` |
    | `T [2:4]` | `!moore.array<3 x T>` |
  }];
  let parameters = (ins "unsigned":$size, "PackedType":$elementType);
  let assemblyFormat = [{
    `<` $size `x` $elementType `>`
  }];
  let builders = [
    AttrBuilderWithInferredContext<
      (ins "unsigned":$size, "PackedType":$elementType), [{
        return $_get(elementType.getContext(), size, elementType);
      }]>
  ];
}

def UnpackedArrayType : MooreTypeDef<"UnpackedArray", [],
  "moore::UnpackedType"
> {
  let mnemonic = "uarray";
  let summary = "an unpacked array type";
  let description = [{
    An unpacked array with a fixed number of elements. This type represents
    unpacked range dimensions (`[a:b]`) and unpacked array dimensions (`[a]`) in
    SystemVerilog.

    | Verilog       | Moore Dialect          |
    |---------------|------------------------|
    | `T foo [3:0]` | `!moore.uarray<4 x T>` |
    | `T foo [2:4]` | `!moore.uarray<3 x T>` |
    | `T foo [2]`   | `!moore.uarray<2 x T>` |
  }];
  let parameters = (ins "unsigned":$size, "UnpackedType":$elementType);
  let assemblyFormat = [{
    `<` $size `x` $elementType `>`
  }];
  let builders = [
    AttrBuilderWithInferredContext<
      (ins "unsigned":$size, "UnpackedType":$elementType), [{
        return $_get(elementType.getContext(), size, elementType);
      }]>
  ];
}

def OpenArrayType : MooreTypeDef<"OpenArray", [], "moore::PackedType"> {
  let mnemonic = "open_array";
  let summary = "an open packed array type";
  let description = [{
    A packed array with an unspecified number of elements. This type represents
    unsized/open packed arrays (`[]`) in SystemVerilog.

    | Verilog | Moore Dialect          |
    |---------|------------------------|
    | `T []`  | `!moore.open_array<T>` |
  }];
  let parameters = (ins "PackedType":$elementType);
  let assemblyFormat = [{
    `<` $elementType `>`
  }];
  let builders = [
    AttrBuilderWithInferredContext<
      (ins "PackedType":$elementType), [{
        return $_get(elementType.getContext(), elementType);
      }]>
  ];
}

def OpenUnpackedArrayType : MooreTypeDef<"OpenUnpackedArray", [],
  "moore::UnpackedType"
> {
  let mnemonic = "open_uarray";
  let summary = "an open unpacked array type";
  let description = [{
    An unpacked array with an unspecified number of elements. This type
    represents unsized/open unpacked arrays (`[]`) in SystemVerilog.

    | Verilog    | Moore Dialect           |
    |------------|-------------------------|
    | `T foo []` | `!moore.open_uarray<T>` |
  }];
  let parameters = (ins "UnpackedType":$elementType);
  let assemblyFormat = [{
    `<` $elementType `>`
  }];
  let builders = [
    AttrBuilderWithInferredContext<
      (ins "UnpackedType":$elementType), [{
        return $_get(elementType.getContext(), elementType);
      }]>
  ];
}

def AssocArrayType : MooreTypeDef<"AssocArray", [], "moore::UnpackedType"> {
  let mnemonic = "assoc_array";
  let summary = "an associative array type";
  let description = [{
    An associative array. This type represents associative arrays (`[T]`) in
    SystemVerilog.

    | Verilog     | Moore Dialect              |
    |-------------|----------------------------|
    | `T foo [K]` | `!moore.assoc_array<T, K>` |
  }];
  let parameters = (ins "UnpackedType":$elementType, "UnpackedType":$indexType);
  let assemblyFormat = [{
    `<` $elementType `,` $indexType `>`
  }];
  let builders = [
    AttrBuilderWithInferredContext<
      (ins "UnpackedType":$elementType, "UnpackedType":$indexType), [{
        return $_get(elementType.getContext(), elementType, indexType);
      }]>
  ];
}

def QueueType : MooreTypeDef<"Queue", [], "moore::UnpackedType"> {
  let mnemonic = "queue";
  let summary = "a queue type";
  let description = [{
    A queue with an optional upper bound on the number of elements that it can
    hold. This type represents queues (`[$]` and `[$:a]`) in SystemVerilog. A
    `bound` of 0 indicates an unbounded queue.

    | Verilog        | Moore Dialect         |
    |----------------|-----------------------|
    | `T foo [$]`    | `!moore.queue<T>`     |
    | `T foo [$:42]` | `!moore.queue<T, 42>` |
  }];
  let parameters = (ins
    "UnpackedType":$elementType,
    "unsigned":$bound
  );
  let assemblyFormat = [{
    `<` $elementType `,` $bound `>`
  }];
  let builders = [
    AttrBuilderWithInferredContext<
      (ins "UnpackedType":$elementType, "unsigned":$bound), [{
        return $_get(elementType.getContext(), elementType, bound);
      }]>
  ];
}

//===----------------------------------------------------------------------===//
// Structs
//===----------------------------------------------------------------------===//

// Common pieces of struct-like types.
class StructLikeType<
  string name, list<Trait> traits = [], string baseCppClass = "::mlir::Type"
> : MooreTypeDef<name, traits, baseCppClass> {
  let parameters = (ins ArrayRefParameter<"StructLikeMember">:$members);
  let assemblyFormat = [{
    `<` custom<Members>($members) `>`
  }];
  let extraClassDeclaration = [{
    std::optional<DenseMap<Attribute, Type>> getSubelementIndexMap();
    Type getTypeAtIndex(Attribute index);
    std::optional<uint32_t> getFieldIndex(StringAttr nameField);
  }];
}

def StructType : StructLikeType<"Struct", [
  DeclareTypeInterfaceMethods<DestructurableTypeInterface
>], "moore::PackedType"> {
  let mnemonic = "struct";
  let summary = "a packed struct type";
  let description = [{
    A packed struct. All members are guaranteed to be packed as well.
  }];
  let genVerifyDecl = 1;
}

def UnpackedStructType : StructLikeType<
  "UnpackedStruct", [
  DeclareTypeInterfaceMethods<DestructurableTypeInterface
>], "moore::UnpackedType"
> {
  let mnemonic = "ustruct";
  let summary = "an unpacked struct type";
  let description = [{
    An unpacked struct.
  }];
}

def UnionType : StructLikeType<"Union", [
  DeclareTypeInterfaceMethods<DestructurableTypeInterface
>], "moore::PackedType"> {
  let mnemonic = "union";
  let summary = "a packed union type";
  let description = [{
    A packed union. All members are guaranteed to be packed as well.
  }];
  let genVerifyDecl = 1;
}


def UnpackedUnionType : StructLikeType<
  "UnpackedUnion", [
  DeclareTypeInterfaceMethods<DestructurableTypeInterface
>], "moore::UnpackedType"
> {
  let mnemonic = "uunion";
  let summary = "an unpacked union type";
  let description = [{
    An unpacked union.
  }];
}

//===----------------------------------------------------------------------===//
// Reference type wrapper
//===----------------------------------------------------------------------===//

def RefType : MooreTypeDef<"Ref", [
  DeclareTypeInterfaceMethods<DestructurableTypeInterface
>], "moore::UnpackedType">{
  let mnemonic = "ref";
  let description = [{
    A wrapper is used to wrap any SystemVerilog type. It's aimed to work for
    'moore.variable', 'moore.blocking_assign', and 'moore.read', which are
    related to memory, like alloca/write/read.
  }];
  let parameters = (ins "UnpackedType":$nestedType);
  let assemblyFormat = [{
    `<` $nestedType `>`
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "UnpackedType":$nestedType), [{
      return $_get(nestedType.getContext(), nestedType);
    }]>
  ];

  let extraClassDeclaration = [{
    Domain getDomain() { return getNestedType().getDomain(); }
    std::optional<unsigned> getBitSize() {
      return getNestedType().getBitSize();
    };
    std::optional<DenseMap<Attribute, Type>> getSubelementIndexMap();
    Type getTypeAtIndex(Attribute index);
    std::optional<uint32_t> getFieldIndex(StringAttr nameField);
  }];
}

//===----------------------------------------------------------------------===//
// Format String
//===----------------------------------------------------------------------===//

def FormatStringType : MooreTypeDef<"FormatString"> {
  let mnemonic = "format_string";
  let summary = "a format string type";
  let description = [{
    An interpolated string produced by one of the string formatting operations.
    It is used to parse format strings present in Verilog source text and
    represent them as a sequence of IR operations that specify the formatting of
    individual arguments.
  }];
}

//===----------------------------------------------------------------------===//
// Constraints
//===----------------------------------------------------------------------===//

def ModuleType : MooreType<CPred<"llvm::isa<hw::ModuleType>($_self)">,
  "module type", "hw::ModuleType">;

def PackedType : MooreType<CPred<"llvm::isa<moore::PackedType>($_self)">,
  "packed type", "moore::PackedType">;

def UnpackedType : MooreType<CPred<"llvm::isa<moore::UnpackedType>($_self)">,
  "unpacked type", "moore::UnpackedType">;

/// A packed type with a known bit size.
def PackedTypeWithKnownSize : MooreType<CPred<[{
    llvm::isa<moore::PackedType>($_self) &&
    llvm::cast<moore::PackedType>($_self).getBitSize().has_value()
  }]>, "packed type with known size", "moore::PackedType">;

/// A simple bit vector type.
def SimpleBitVectorType : MooreType<CPred<[{
    llvm::isa<moore::IntType>($_self)
  }]>, "simple bit vector type", "moore::IntType">;

/// A single two or four-valued bit.
def AnySingleBitType : MooreType<CPred<[{
    moore::isIntType($_self, 1)
  }]>, "single bit type", "moore::IntType">;

/// A single two-valued bit.
def BitType : MooreType<CPred<[{
    moore::isIntType($_self, 1, moore::Domain::TwoValued)
  }]>, "`bit` type", "moore::IntType"> {
  let builderCall = [{
    IntType::getInt($_builder.getContext(), 1)
  }];
}

/// A two-valued integer.
def TwoValuedIntType : MooreType<CPred<[{
    moore::isIntType($_self, moore::Domain::TwoValued)
  }]>, "two-valued integer type", "moore::IntType">;

/// A four-valued integer.
def FourValuedIntType : MooreType<CPred<[{
    moore::isIntType($_self, moore::Domain::FourValued)
  }]>, "four-valued integer type", "moore::IntType">;

/// A 64-bit four-valued integer.
def FourValuedI64 : MooreType<CPred<[{
    moore::isIntType($_self, 64, moore::Domain::FourValued)
  }]>, "64-bit four-valued integer type", "moore::IntType"> {
  let builderCall = "IntType::getLogic($_builder.getContext(), 64)";
}

/// A 32-bit four-valued integer.
def FourValuedI32 : MooreType<CPred<[{
    moore::isIntType($_self, 32, moore::Domain::FourValued)
  }]>, "32-bit four-valued integer type", "moore::IntType"> {
  let builderCall = "IntType::getLogic($_builder.getContext(), 32)";
}

// A 64-bit two-valued integer.
def TwoValuedI64 : MooreType<CPred<[{
    moore::isIntType($_self, 64, moore::Domain::TwoValued)
  }]>, "64-bit two-valued integer type", "moore::IntType"> {
  let builderCall = "IntType::getInt($_builder.getContext(), 64)";
}

// A 32-bit two-valued integer.
def TwoValuedI32 : MooreType<CPred<[{
    moore::isIntType($_self, 32, moore::Domain::TwoValued)
  }]>, "32-bit two-valued integer type", "moore::IntType"> {
  let builderCall = "IntType::getInt($_builder.getContext(), 32)";
}

// A 32-bit real value.
def RealF32 : MooreType<CPred<[{
    moore::isRealType($_self, 32)
  }]>, "32-bit real type, aka. shortreal", "moore::RealType"> {
  let builderCall = "RealType::getShortReal($_builder.getContext())";
}

// A 64-bit real value.
def RealF64 : MooreType<CPred<[{
    moore::isRealType($_self, 64)
  }]>, "64-bit real type, aka. real", "moore::RealType"> {
  let builderCall = "RealType::getReal($_builder.getContext())";
}

/// A packed or unpacked array type with a fixed size.
def AnyStaticArrayType : MooreType<
  Or<[ArrayType.predicate, UnpackedArrayType.predicate]>,
  "packed or unpacked static array type",
  "moore::UnpackedType">;

/// A packed or unpacked struct type.
def AnyStructType : MooreType<
  Or<[StructType.predicate, UnpackedStructType.predicate]>,
  "packed or unpacked struct type",
  "moore::UnpackedType">;

/// A packed or unpacked union type.
def AnyUnionType : MooreType<
  Or<[UnionType.predicate, UnpackedUnionType.predicate]>,
  "packed or unpacked union type",
  "moore::UnpackedType">;

/// A ref type with the specified constraints on the nested type.
class SpecificRefType<Type type> : ConfinedType<RefType,
  [SubstLeaves<"$_self", "llvm::cast<moore::RefType>($_self).getNestedType()",
               type.predicate>],
  "ref of " # type.summary, "moore::RefType"
> {
  Type nestedType = type;
}

/// Struct references.
def StructRefType : SpecificRefType<StructType>;
def UnpackedStructRefType : SpecificRefType<UnpackedStructType>;
def AnyStructRefType : SpecificRefType<AnyStructType>;

/// Union references.
def UnionRefType : SpecificRefType<UnionType>;
def UnpackedUnionRefType : SpecificRefType<UnpackedUnionType>;
def AnyUnionRefType : SpecificRefType<AnyUnionType>;

#endif // CIRCT_DIALECT_MOORE_MOORETYPES
