//===- FIRRTLTypesImpl.td - FIRRTL data type definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data type implementations for the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLTYPESIMPL_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLTYPESIMPL_TD

include "FIRRTLDialect.td"
include "circt/Dialect/FIRRTL/FIRRTLTypeInterfaces.td"
include "circt/Dialect/HW/HWTypeInterfaces.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class FIRRTLImplType<string name,
                     list<Trait> traits = [],
                     string baseCppClass = "::circt::firrtl::FIRRTLBaseType">
    : TypeDef<FIRRTLDialect, name, traits, baseCppClass> {
  // Storage classes must be defined in C++ and
  // inherit from FIRRTLBaseTypeStorage
  let genStorageClass = false;

  // MLIR generates awkward accessor "getIsConst" for the "isConst" parameter,
  // which is common on FIRRTLBaseType anyway, so we generate the other
  // accessors manually
  let genAccessors = false;
}

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

def SIntImpl : FIRRTLImplType<"SInt",
                              [WidthQualifiedTypeTrait],
                              "::circt::firrtl::IntType"> {
  let summary = "A signed integer type, whose width may not be known.";
  let parameters = (ins "int32_t":$widthOrSentinel, "bool":$isConst);
  let storageClass = "WidthTypeStorage";
  let builders = [
    TypeBuilder<(ins "std::optional<int32_t>":$width, CArg<"bool", "false">:$isConst)>,
    TypeBuilder<(ins)>,
  ];
  let genVerifyDecl = true;
  let typeName = "firrtl.sint";
  let extraClassDeclaration = [{
    using WidthQualifiedTypeTrait<SIntType>::getWidth;
    using WidthQualifiedTypeTrait<SIntType>::hasWidth;
    int32_t getWidthOrSentinel() const;
    SIntType getConstType(bool isConst);
  }];
}

def UIntImpl : FIRRTLImplType<"UInt",
                              [WidthQualifiedTypeTrait],
                              "::circt::firrtl::IntType"> {
  let summary = "An unsigned integer type, whose width may not be known.";
  let parameters = (ins "int32_t":$widthOrSentinel, "bool":$isConst);
  let storageClass = "WidthTypeStorage";
  let builders = [
    TypeBuilder<(ins "std::optional<int32_t>":$width, CArg<"bool", "false">:$isConst)>,
    TypeBuilder<(ins)>,
  ];
  let genVerifyDecl = true;
  let typeName = "firrtl.uint";
  let extraClassDeclaration = [{
    using WidthQualifiedTypeTrait<UIntType>::getWidth;
    using WidthQualifiedTypeTrait<UIntType>::hasWidth;
    int32_t getWidthOrSentinel() const;
    UIntType getConstType(bool isConst);
  }];
}

def ClockTypeImpl : FIRRTLImplType<"Clock"> {
  let summary = "Clock signal";
  let parameters = (ins "bool":$isConst);
  let storageClass = "FIRRTLBaseTypeStorage";
  let typeName = "firrtl.clock";
  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, false);
    }]>
  ];
  let extraClassDeclaration = [{
    ClockType getConstType(bool isConst);
  }];
}

def ResetTypeImpl : FIRRTLImplType<"Reset"> {
  let summary = "Reset Signal";
  let parameters = (ins "bool":$isConst);
  let storageClass = "FIRRTLBaseTypeStorage";
  let typeName = "firrtl.reset";
  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, false);
    }]>
  ];
  let extraClassDeclaration = [{
    ResetType getConstType(bool isConst);
  }];
}

def AsyncResetTypeImpl : FIRRTLImplType<"AsyncReset"> {
  let summary = "AsyncReset signal";
  let parameters = (ins "bool":$isConst);
  let storageClass = "FIRRTLBaseTypeStorage";
  let typeName = "firrtl.asyncreset";
  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, false);
    }]>
  ];
  let extraClassDeclaration = [{
    AsyncResetType getConstType(bool isConst);
  }];
}

def AnalogTypeImpl : FIRRTLImplType<"Analog",
  [WidthQualifiedTypeTrait]> {
  let summary = "Analog signal";
  let parameters = (ins "int32_t":$widthOrSentinel, "bool":$isConst);
  let storageClass = "WidthTypeStorage";
  let typeName = "firrtl.analog";
  let builders = [
    TypeBuilder<(ins "std::optional<int32_t>":$width, CArg<"bool", "false">:$isConst)>,
    TypeBuilder<(ins)>,
  ];
  let extraClassDeclaration = [{
    int32_t getWidthOrSentinel() const;
    AnalogType getConstType(bool isConst);
  }];
  let genVerifyDecl = true;
}

class BaseVectorTypeImpl<string name, string ElementType, list<Trait> traits = [], string BaseType = ElementType>
  : FIRRTLImplType<name, traits # [DeclareTypeInterfaceMethods<FieldIDTypeInterface>], BaseType> {
  let summary = "a fixed size collection of elements, like an array.";
  let parameters = (ins
      TypeParameter<ElementType, "Type of vector elements">:$elementType,
     "size_t":$numElements,
     "bool":$isConst
  );
  let storageClass = name # "TypeStorage";
  let skipDefaultBuilders = true;
  let builders = [
    TypeBuilderWithInferredContext<(ins
      ElementType:$elementType,
      "size_t":$numElements,
       CArg<"bool", "false">:$isConst)
    >
  ];

  // Additional class declarations to emit.
  code firrtlExtraClassDeclaration = "";

  let extraClassDeclaration = [{
    using ElementType = }] # ElementType # [{;

    ElementType getElementType() const;
    size_t getNumElements() const;

    /// Return the recursive properties of the type.
    RecursiveTypeProperties getRecursiveTypeProperties() const;

    /// Const support.
    bool isConst();
    ElementType getElementTypePreservingConst();

    /// Return a 'const' or non-'const' version of this type.
    }] # name # [{Type getConstType(bool isConst);

  }] # firrtlExtraClassDeclaration;
}

def FVectorImpl : BaseVectorTypeImpl<"FVector","::circt::firrtl::FIRRTLBaseType"> {
  let typeName = "firrtl.vector";
  let firrtlExtraClassDeclaration = [{
    /// Return this type with any flip types recursively removed from itself.
    FIRRTLBaseType getPassiveType();

    /// Return this type with a 'const' modifiers dropped
    FVectorType getAllConstDroppedType();

    /// Return this type with any type alias types recursively removed from itself.
    FIRRTLBaseType getAnonymousType();
  }];
}

def OpenVectorImpl : BaseVectorTypeImpl<"OpenVector","::circt::firrtl::FIRRTLType"> {
  let genVerifyDecl = 1;
  let typeName = "firrtl.openvector";
}

class BaseBundleTypeImpl<string name, string ElementType, list<Trait> traits = [], string BaseType = ElementType>
 : FIRRTLImplType<name, traits # [DeclareTypeInterfaceMethods<FieldIDTypeInterface>], BaseType> {
  let summary = "an aggregate of named elements. This is effectively a struct.";
  let parameters = (ins "ArrayRef<BundleElement>":$elements, "bool":$isConst);
  let storageClass = name # "TypeStorage";
  let skipDefaultBuilders = true;
  let builders = [
    TypeBuilder<(ins "ArrayRef<BundleElement>":$elements, CArg<"bool", "false">:$isConst)>
  ];

  // Additional class declarations to emit.
  code firrtlExtraClassDeclaration = "";

  let extraClassDeclaration = [{
    using ElementType = }] # ElementType # [{;

    /// Each element of a bundle, which is a name and type.
    struct BundleElement {
      StringAttr name;
      bool isFlip;
      ElementType type;

      BundleElement(StringAttr name, bool isFlip, ElementType type)
          : name(name), isFlip(isFlip), type(type) {}

      bool operator==(const BundleElement &rhs) const {
        return name == rhs.name && isFlip == rhs.isFlip && type == rhs.type;
      }
      bool operator!=(const BundleElement &rhs) const {
        return !operator==(rhs);
      }

      friend llvm::hash_code hash_value(const BundleElement &arg) {
        return llvm::hash_combine(arg.name, arg.isFlip, arg.type);
      }
    };

    ArrayRef<BundleElement> getElements() const;

    size_t getNumElements() const { return getElements().size(); }

    /// Look up an element's index by name.  This returns None on failure.
    std::optional<unsigned> getElementIndex(StringAttr name);
    std::optional<unsigned> getElementIndex(StringRef name);

    /// Look up an element's name by index. This asserts if index is invalid.
    StringAttr getElementNameAttr(size_t index);
    StringRef getElementName(size_t index);

    /// Look up an element by name.  This returns None on failure.
    std::optional<BundleElement> getElement(StringAttr name);
    std::optional<BundleElement> getElement(StringRef name);

    /// Look up an element by index.  This asserts if index is invalid.
    BundleElement getElement(size_t index);

    /// Look up an element type by name.
    ElementType getElementType(StringAttr name);
    ElementType getElementType(StringRef name);

    /// Look up an element type by index.
    ElementType getElementType(size_t index) const;

    /// Return the recursive properties of the type.
    RecursiveTypeProperties getRecursiveTypeProperties() const;

    using iterator = ArrayRef<BundleElement>::iterator;
    iterator begin() const { return getElements().begin(); }
    iterator end() const { return getElements().end(); }

    /// Const support.
    bool isConst();
    ElementType getElementTypePreservingConst(size_t index);

    /// Return a 'const' or non-'const' version of this type.
    }] # name # [{Type getConstType(bool isConst);

  }] # firrtlExtraClassDeclaration;
}

def BundleImpl : BaseBundleTypeImpl<"Bundle","::circt::firrtl::FIRRTLBaseType"> {
  let typeName = "firrtl.bundle";
  let genVerifyDecl = 1;
  let firrtlExtraClassDeclaration = [{
    /// Return this type with any flip types recursively removed from itself.
    FIRRTLBaseType getPassiveType();

    /// Return this type with a 'const' modifiers dropped
    BundleType getAllConstDroppedType();

    /// Return this type with any type alias types recursively removed from itself.
    FIRRTLBaseType getAnonymousType();
  }];
}

def OpenBundleImpl : BaseBundleTypeImpl<"OpenBundle","::circt::firrtl::FIRRTLType"> {
  let typeName = "firrtl.openbundle";
  let genVerifyDecl = 1;
}

def FEnumImpl : FIRRTLImplType<"FEnum", [DeclareTypeInterfaceMethods<FieldIDTypeInterface>]> {
  let summary = "a sum type of named elements.";
  let parameters = (ins "ArrayRef<EnumElement>":$elements, "bool":$isConst);
  let storageClass = "FEnumTypeStorage";
  let genVerifyDecl = true;
  let typeName = "firrtl.enum";
  let skipDefaultBuilders = true;
  let builders = [
    TypeBuilder<(ins "ArrayRef<EnumElement>":$elements, CArg<"bool", "false">:$isConst)>
  ];
  let extraClassDeclaration = [{
    /// Each element of an enum, which is a name and type.
    struct EnumElement {
      StringAttr name;
      IntegerAttr value;
      FIRRTLBaseType type;

      EnumElement(StringAttr name, IntegerAttr value, FIRRTLBaseType type)
          : name(name), value(value), type(type) {}

      bool operator==(const EnumElement &rhs) const {
        return name == rhs.name && value == rhs.value && type == rhs.type;
      }
      bool operator!=(const EnumElement &rhs) const {
        return !operator==(rhs);
      }
      bool operator<(const EnumElement &rhs) const {
        return value.getValue().ult(rhs.value.getValue());
      }

      friend llvm::hash_code hash_value(const FEnumType::EnumElement &arg) {
        return mlir::hash_value(arg.name) ^ mlir::hash_value(arg.value) ^
               mlir::hash_value(arg.type);
      }
    };

    ArrayRef<EnumElement> getElements() const;

    size_t getNumElements() const { return getElements().size(); }

    FEnumType getConstType(bool isConst);

    /// Return this type with a 'const' modifiers dropped
    FEnumType getAllConstDroppedType();

    /// Return this type with any type alias types recursively removed from itself.
    FIRRTLBaseType getAnonymousType();

    /// Get the width of the the tag field.
    size_t getTagWidth();

    /// Look up an element's index by name.  This returns None on failure.
    std::optional<unsigned> getElementIndex(StringAttr name);
    std::optional<unsigned> getElementIndex(StringRef name);

    /// Look up an element's name by index. This asserts if index is invalid.
    StringAttr getElementNameAttr(size_t index);
    StringRef getElementName(size_t index);
    
    // Look up an element's value by index. This asserts if index is invalid.
    IntegerAttr getElementValueAttr(size_t index);
    APInt getElementValue(size_t index);

    // Look up an element's type by index. This asserts if index is invalid.
    FIRRTLBaseType getElementType(size_t index);

    /// Look up an element by name.  This returns None on failure.
    std::optional<EnumElement> getElement(StringAttr name);
    std::optional<EnumElement> getElement(StringRef name);

    /// Look up an element by index.  This asserts if index is invalid.
    EnumElement getElement(size_t index);

    /// Look up an element type by name.
    FIRRTLBaseType getElementType(StringAttr name);
    FIRRTLBaseType getElementType(StringRef name);

    /// Look up an element type by index.
    FIRRTLBaseType getElementType(size_t index) const;
    FIRRTLBaseType getElementTypePreservingConst(size_t index);

    /// Return the recursive properties of the type.
    RecursiveTypeProperties getRecursiveTypeProperties() const;

    using iterator = ArrayRef<EnumElement>::iterator;
    iterator begin() const { return getElements().begin(); }
    iterator end() const { return getElements().end(); }
  }];
}

def RefImpl : FIRRTLImplType<"Ref",
                             [],
                             "::circt::firrtl::FIRRTLType"> {
  let summary = "A reference to a signal elsewhere.";
  let description = [{
    A reference type, such as `firrtl.probe<uint<1>>` or `firrtl.rwprobe<uint<2>>`.

    Used for remote reads and writes of the wrapped base type.

    Parameterized over the referenced base type, with flips removed.

    Not a base type.

    Values of this type are used to capture dataflow paths,
    and do not represent a circuit element or entity.

    Generally read-only (probe), optionally forceable (rwprobe).
  }];
  let parameters = (ins TypeParameter<"::circt::firrtl::FIRRTLBaseType",
                                      "Type of reference target">:$type,
                        "bool":$forceable,
                        OptionalParameter<"::mlir::SymbolRefAttr">:$layer);
  let genAccessors = true;
  let genStorageClass = true;
  let genVerifyDecl = true;
  let typeName = "firrtl.ref";

  let skipDefaultBuilders = true;
  let builders = [
    TypeBuilderWithInferredContext<(ins "::circt::firrtl::FIRRTLBaseType":$type,
                                        CArg<"bool", "false">:$forceable,
                                        CArg<"::mlir::SymbolRefAttr", "{}">:$layer)>
  ];

  let extraClassDeclaration = [{
    /// Return the recursive properties of the type.
    RecursiveTypeProperties getRecursiveTypeProperties() const;

    RefType removeLayer() const {
      if (getLayer() == nullptr)
        return *this;
      return get(getType(), getForceable());
    }
  }];
}

def LHSTypeImpl : FIRRTLImplType<"LHS", [], "::circt::firrtl::FIRRTLType"> {
  let summary = "A wrapper for LHS types.";
  let description = [{
    A LHS type is a type usable for the destination of a strict connect
    and for field indexing.  No other operations are valid.  Any passive,
    strict-connectable types are valid inside LHS types.
  }];
  let parameters = (ins "FIRRTLBaseType":$type);
  let genAccessors = true;
  let genStorageClass = true;
  let genVerifyDecl = true;

  let typeName = "firrtl.lhs";

  let builders = [
    TypeBuilderWithInferredContext<(ins "::circt::firrtl::FIRRTLBaseType":$type)>
  ];
}

def BaseTypeAliasImpl : FIRRTLImplType<"BaseTypeAlias", [DeclareTypeInterfaceMethods<FieldIDTypeInterface>],
                                        "::circt::firrtl::FIRRTLBaseType"> {
  let summary = "type alias for firrtl base types";
  let parameters =
    (ins  "StringAttr":$name, TypeParameter<"::circt::firrtl::FIRRTLBaseType",
                                            "An inner type">:$innerType);
  let storageClass = "BaseTypeAliasStorage";
  let typeName = "firrtl.basetypealias";
  let genAccessors = true;
  let skipDefaultBuilders = true;
  let extraClassDeclaration = [{
      // FIRRTLBaseType utils.
      FIRRTLBaseType getAnonymousType();
      FIRRTLBaseType getPassiveType();
      FIRRTLBaseType getConstType(bool isConst);
      FIRRTLBaseType getAllConstDroppedType();

      /// Return the recursive properties of the type.
      RecursiveTypeProperties getRecursiveTypeProperties() const;

      // If a given `newInnerType` is identical to innerType, return `*this`
      // because we can reuse the type alias. Otherwise return `newInnerType`.
      FIRRTLBaseType getModifiedType(FIRRTLBaseType newInnerType);
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
           "::circt::firrtl::FIRRTLBaseType":$innerType)>
  ];
}

//===----------------------------------------------------------------------===//
// Non-Hardware Type Definitions
//===----------------------------------------------------------------------===//

class PropImplType<string name,
                   list<Trait> traits = [],
                   string baseCppClass = "::circt::firrtl::FIRRTLType">
  : FIRRTLImplType<name, traits, baseCppClass>;

def ClassImpl : PropImplType<"Class", [
  DeclareTypeInterfaceMethods<FieldIDTypeInterface>]> {
  let summary = [{
    An instance of a class.

    Example:
    ```mlir
    !firrtl.class<@Module(in p0: !firrtl.uint<8>, out p1: !firrtl.uint<8>)>
    ```
  }];
  let parameters = (ins
    "FlatSymbolRefAttr":$name,
    ArrayRefParameter<"ClassElement">:$elements
  );
  let genStorageClass = false;
  let typeName = "firrtl.propClass";
  let genAccessors = false;
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "FlatSymbolRefAttr":$name,
      "ArrayRef<ClassElement>":$elements
    )>
  ];
  let extraClassDeclaration = [{
    /// Helper to parse this type as a module interface.
    static ParseResult parseInterface(
      AsmParser &parser, ClassType &result);

    StringRef getName() const;
    FlatSymbolRefAttr getNameAttr() const;

    ArrayRef<ClassElement> getElements() const;
    size_t getNumElements() const { return getElements().size(); }
    const ClassElement &getElement(IntegerAttr index) const;
    const ClassElement &getElement(size_t index) const;
    std::optional<uint64_t> getElementIndex(StringRef fieldName) const;

    /// Helper to print this type as a module interface.
    /// Example:
    /// ```
    /// @MyClass(in %port: !firrtl.uint<8>, out o: !firrtl.uint<8>)
    /// ```
    void printInterface(AsmPrinter &p) const;
  }];
}

def AnyRefImpl : PropImplType<"AnyRef"> {
  let summary = "A reference to an instance of any class.";
  let description = [{
    A reference of type AnyRef can be used in ports, property assignments, and
    any other Property "plumbing" ops. But it is opaque, and references to
    objects of AnyRef type cannot be "dereferenced". There is no information
    about the referred to object's fields, so subfield access, etc. is illegal.
  }];
  let genStorageClass = true;
  let typeName = "firrtl.propRef";
}

def StringImpl : PropImplType<"String"> {
  let summary = "An unlimited length string type. Not representable in hardware.";
  let parameters = (ins);
  let genStorageClass = true;
  let typeName = "firrtl.propString";
}

def IntegerImpl : PropImplType<"FInteger"> {
  let summary = "An unlimited length signed integer type.  Not representable in hardware.";
  let parameters = (ins);
  let genStorageClass = true;
  let typeName = "firrtl.propInt";
}

def ListImpl : PropImplType<"List"> {
  let summary = "A typed property list of any length.  Not representable in hardware.";
  let parameters = (ins TypeParameter<"circt::firrtl::PropertyType", "element type">:$elementType);
  let genStorageClass = true;
  let genAccessors = true;
  let typeName = "firrtl.propList";
}

def PathImpl : PropImplType<"Path"> {
  let summary = "A path to a hardware entity.  Not representable in hardware.";
  let parameters = (ins);
  let genStorageClass = true;
  let genAccessors = true;
  let typeName = "firrtl.propPath";
}

def BoolImpl : PropImplType<"Bool"> {
  let summary = "A boolean property.  Not representable in hardware.";
  let parameters = (ins);
  let genStorageClass = true;
  let typeName = "firrtl.propBool";
}

def DoubleImpl : PropImplType<"Double"> {
  let summary = "A double property.  Not representable in hardware.";
  let parameters = (ins);
  let genStorageClass = true;
  let typeName = "firrtl.propDouble";
}

def FStringImpl : FIRRTLImplType<"FString", [], "::circt::firrtl::FIRRTLType" > {
  let summary = "A format string type";
  let parameters = (ins);
  let genStorageClass = true;
  let typeName = "firrtl.fstring";
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLTYPESIMPL_TD
