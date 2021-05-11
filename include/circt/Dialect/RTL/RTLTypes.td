//===- RTLTypes.td - RTL data type definitions -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data types for the RTL dialect.
//
//===----------------------------------------------------------------------===//

// Base class for other typedefs. Provides dialact-specific defaults.
class RTLType<string name> : TypeDef<RTLDialect, name> { }

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def RTLIntegerType : DialectType<RTLDialect, CPred<"isRTLIntegerType($_self)">,
                                 "an integer bitvector of one or more bits",
                                 "::mlir::IntegerType">;

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def RTLValueType : DialectType<RTLDialect, CPred<"isRTLValueType($_self)">,
                               "a known primitive element">;

// Type constraint that indicates that an operand/result may only be a valid
// non-directional type.
def RTLNonInOutType : DialectType<RTLDialect, CPred<"!hasRTLInOutType($_self)">,
                                  "a type without inout">;

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//

// A simple fixed size array.
def ArrayType : RTLType<"Array"> {
  let summary = "fixed-sized array";
  let description = [{
    Fixed sized RTL arrays are roughly similar to C arrays. On the wire (vs.
    in a memory), arrays are always packed. Memory layout is not defined as
    it does not need to be since in silicon there is not implicit memory
    sharing.
  }];

  let mnemonic = "array";
  let parameters = (ins "::mlir::Type":$elementType, "size_t":$size);
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static ArrayType get(Type elementType, size_t size) {
      return get(elementType.getContext(), elementType, size);
    }
  }];
}

// An 'unpacked' array of fixed size.
def UnpackedArrayType : RTLType<"UnpackedArray"> {
  let summary = "SystemVerilog 'unpacked' fixed-sized array";
  let description = [{
    Unpacked arrays are a more flexible array representation than packed arrays,
    and are typically used to model memories.  See SystemVerilog Spec 7.4.2.
  }];

  let mnemonic = "uarray";
  let parameters = (ins "::mlir::Type":$elementType, "size_t":$size);
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static UnpackedArrayType get(Type elementType, size_t size) {
      return get(elementType.getContext(), elementType, size);
    }
  }];
}

def InOutType : RTLType<"InOut"> {
  let summary = "inout type";
  let description = [{
    InOut type is used for model operations and values that have "connection"
    semantics, instead of typical dataflow behavior.  This is used for wires
    and inout ports in Verilog.
  }];

  let mnemonic = "inout";
  let parameters = (ins "::mlir::Type":$elementType);
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    static InOutType get(Type elementType) {
      return get(elementType.getContext(), elementType);
    }
  }];
}

def StructType : RTLType<"Struct"> {
  let summary = "RTL struct type";
  let description = [{
    Represents a structure of name, value pairs.
    !rtl.struct<fieldName1: Type1, fieldName2: Type2>
  }];
  let mnemonic = "struct";

  let parameters = (
    ins
    // An ArrayRef of something which requires allocation in the storage
    // constructor.
    ArrayRefOfSelfAllocationParameter<
      "::circt::rtl::StructType::FieldInfo",
      "struct fields">: $elements
  );

  let extraClassDeclaration = [{
    using FieldInfo = ::circt::rtl::detail::FieldInfo;
    mlir::Type getFieldType(mlir::StringRef fieldName);
    void getInnerTypes(mlir::SmallVectorImpl<mlir::Type>&);
  }];
}

def UnionType : RTLType<"Union"> {
  let summary = "An untagged union of types";
  let parameters = (
    ins
    // An ArrayRef of something which requires allocation in the storage
    // constructor.
    ArrayRefOfSelfAllocationParameter<
      "::circt::rtl::UnionType::FieldInfo",
      "union fields">: $elements
  );
  let mnemonic = "union";

  let extraClassDeclaration = [{
    using FieldInfo = ::circt::rtl::detail::FieldInfo;
    mlir::Type getFieldType(mlir::StringRef fieldName);
  }];
}

def TypeRefType : RTLType<"TypeRef"> {
  let summary = "An symbolic reference to a type declaration";
  let description = [{
    A TypeRef is parameterized by a SymbolRefAttr, which points to the
    declaration of this type. Upon construction, a TypeRef verifies the
    referent exists and stores the inner type. It also canonicalizes the inner
    type, resolving any nested TypeRefs. If different, the canonical type is
    also stored.
  }];

  let mnemonic = "typeref";

  let parameters = (ins
    "mlir::SymbolRefAttr":$ref,
    "mlir::Type":$canonicalType,
    "mlir::StringAttr":$name
  );

  let genStorageClass = 0;
  let genAccessors = 0;

  let extraClassDeclaration = [{
    static TypeRefType get(SymbolRefAttr ref);
    mlir::SymbolRefAttr getRef() const;
  }];
}
