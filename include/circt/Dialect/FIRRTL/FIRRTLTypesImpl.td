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
include "circt/Dialect/HW/HWTypeInterfaces.td"

// Base class for other typedefs. Provides dialact-specific defaults.
class FIRRTLImplType<string name,
                     list<Trait> traits = [],
                     string baseCppClass = "::circt::firrtl::FIRRTLBaseType">
    : TypeDef<FIRRTLDialect, name, traits, baseCppClass> {}

//===----------------------------------------------------------------------===//
// Type Traits
//===----------------------------------------------------------------------===//

def WidthQualifiedTypeTrait : NativeTypeTrait<"WidthQualifiedTypeTrait"> {
  let cppNamespace = "::circt::firrtl";
}

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

def SIntImpl : FIRRTLImplType<"SInt",
                              [WidthQualifiedTypeTrait, FieldIDTypeInterface],
                              "::circt::firrtl::IntType"> {
  let summary = "A signed integer type, whose width may not be known.";
  let parameters = (ins "int32_t":$widthOrSentinel);
  let builders = [
    TypeBuilder<(ins "std::optional<int32_t>":$width)>,
    TypeBuilder<(ins)>,
  ];
  let genVerifyDecl = true;
  let extraClassDeclaration = [{
    using WidthQualifiedTypeTrait<SIntType>::getWidth;
    using WidthQualifiedTypeTrait<SIntType>::hasWidth;
  }];
}

def UIntImpl : FIRRTLImplType<"UInt",
                              [WidthQualifiedTypeTrait, FieldIDTypeInterface],
                              "::circt::firrtl::IntType"> {
  let summary = "An unsigned integer type, whose width may not be known.";
  let parameters = (ins "int32_t":$widthOrSentinel);
  let builders = [
    TypeBuilder<(ins "std::optional<int32_t>":$width)>,
    TypeBuilder<(ins)>,
  ];
  let genVerifyDecl = true;
  let extraClassDeclaration = [{
    using WidthQualifiedTypeTrait<UIntType>::getWidth;
    using WidthQualifiedTypeTrait<UIntType>::hasWidth;
  }];
}

def ClockTypeImpl : FIRRTLImplType<"Clock", [FieldIDTypeInterface]> {
  let summary = "Clock signal";
}

def ResetTypeImpl : FIRRTLImplType<"Reset", [FieldIDTypeInterface]> {
  let summary = "Reset Signal";
}

def AsyncResetTypeImpl : FIRRTLImplType<"AsyncReset", [FieldIDTypeInterface]> {
  let summary = "AsyncReset signal";
}

def AnalogTypeImpl : FIRRTLImplType<"Analog",
  [WidthQualifiedTypeTrait, FieldIDTypeInterface]> {
  let summary = "Analog signal";
  let parameters = (ins "int32_t":$widthOrSentinel);
  let builders = [
    TypeBuilder<(ins "std::optional<int32_t>":$width)>,
    TypeBuilder<(ins)>,
  ];
  let genVerifyDecl = true;
}

def FVectorTypeImpl : FIRRTLImplType<"FVector", [FieldIDTypeInterface]> {
  let summary = "a fixed size collection of elements, like an array.";
  let genStorageClass = false;
  let parameters = (ins "FIRRTLBaseType":$elementType, "size_t":$numElements);
  let skipDefaultBuilders = true;
  let builders = [
    TypeBuilderWithInferredContext<(ins
      "FIRRTLBaseType":$elementType,
      "size_t":$numElements)
    >
  ];
  let extraClassDeclaration = [{
    /// Return the recursive properties of the type.
    RecursiveTypeProperties getRecursiveTypeProperties();

    /// Return this type with any flip types recursively removed from itself.
    FIRRTLBaseType getPassiveType();

    /// Get an integer ID for the field. Field IDs start at 1, and are assigned
    /// to each field in a vector in a recursive depth-first walk of all
    /// elements. A field ID of 0 is used to reference the vector itself.
    uint64_t getFieldID(uint64_t index);

    /// Find the element index corresponding to the desired fieldID.  If the
    /// fieldID corresponds to a field in nested under an element, it will
    /// return the index of the parent element.
    uint64_t getIndexForFieldID(uint64_t fieldID);

    /// Find the index of the element that contains the given fieldID.
    /// As well, rebase the fieldID to the element.
    std::pair<uint64_t, uint64_t> getIndexAndSubfieldID(uint64_t fieldID);

    /// Strip off a single layer of this type and return the sub-type and a
    /// field ID targeting the same field, but rebased on the sub-type.
    std::pair<FIRRTLBaseType, uint64_t> getSubTypeByFieldID(uint64_t fieldID);

    /// Get the maximum field ID in this vector.  This is helpful for
    /// constructing field IDs when this VectorType is nested in another
    /// aggregate type.
    uint64_t getMaxFieldID();

    /// Returns the effective field id when treating the index field as the root
    /// of the type.  Essentially maps a fieldID to a fieldID after a subfield
    /// op. Returns the new id and whether the id is in the given child.
    std::pair<uint64_t, bool> rootChildFieldID(uint64_t fieldID, uint64_t index);
  }];
}

def BundleImpl : FIRRTLImplType<"Bundle", [FieldIDTypeInterface]> {
  let summary = "an aggregate of named elements. This is effectively a struct.";
  let genStorageClass = false;
  let parameters = (ins "ArrayRef<BundleElement>":$elements);
  let extraClassDeclaration = [{
    /// Each element of a bundle, which is a name and type.
    struct BundleElement {
      StringAttr name;
      bool isFlip;
      FIRRTLBaseType type;

      BundleElement(StringAttr name, bool isFlip, FIRRTLBaseType type)
          : name(name), isFlip(isFlip), type(type) {}

      bool operator==(const BundleElement &rhs) const {
        return name == rhs.name && isFlip == rhs.isFlip && type == rhs.type;
      }
      bool operator!=(const BundleElement &rhs) const {
        return !operator==(rhs);
      }

      friend llvm::hash_code hash_value(const BundleType::BundleElement &arg) {
        return mlir::hash_value(arg.name) ^ mlir::hash_value(arg.type);
      }
    };

    size_t getNumElements() { return getElements().size(); }

    /// Look up an element's index by name.  This returns None on failure.
    std::optional<unsigned> getElementIndex(StringAttr name);
    std::optional<unsigned> getElementIndex(StringRef name);

    /// Look up an element's name by index. This asserts if index is invalid.
    StringRef getElementName(size_t index);

    /// Look up an element by name.  This returns None on failure.
    std::optional<BundleElement> getElement(StringAttr name);
    std::optional<BundleElement> getElement(StringRef name);

    /// Look up an element by index.  This asserts if index is invalid.
    BundleElement getElement(size_t index);

    /// Look up an element type by name.
    FIRRTLBaseType getElementType(StringAttr name);
    FIRRTLBaseType getElementType(StringRef name);

    /// Look up an element type by index.
    FIRRTLBaseType getElementType(size_t index);

    /// Return the recursive properties of the type.
    RecursiveTypeProperties getRecursiveTypeProperties();

    /// Return this type with any flip types recursively removed from itself.
    FIRRTLBaseType getPassiveType();

    /// Get an integer ID for the field. Field IDs start at 1, and are assigned
    /// to each field in a bundle in a recursive pre-order walk of all fields,
    /// visiting all nested bundle fields.  A field ID of 0 is used to reference
    /// the bundle itself. The ID can be used to uniquely identify any specific
    /// field in this bundle.
    uint64_t getFieldID(uint64_t index);

    /// Find the element index corresponding to the desired fieldID.  If the
    /// fieldID corresponds to a field in a nested bundle, it will return the
    /// index of the parent field.
    uint64_t getIndexForFieldID(uint64_t fieldID);

    /// Find the index of the element that contains the given fieldID.
    /// As well, rebase the fieldID to the element.
    std::pair<uint64_t, uint64_t> getIndexAndSubfieldID(uint64_t fieldID);

    /// Strip off a single layer of this type and return the sub-type and a
    /// field ID targeting the same field, but rebased on the sub-type.
    std::pair<FIRRTLBaseType, uint64_t> getSubTypeByFieldID(uint64_t fieldID);

    /// Get the maximum field ID in this bundle.  This is helpful for
    /// constructing field IDs when this BundleType is nested in another
    /// aggregate type.
    uint64_t getMaxFieldID();

    /// Returns the effective field id when treating the index field as the root
    /// of the type.  Essentially maps a fieldID to a fieldID after a subfield
    /// op. Returns the new id and whether the id is in the given child.
    std::pair<uint64_t, bool> rootChildFieldID(uint64_t fieldID,
                                               uint64_t index);

    using iterator = ArrayRef<BundleElement>::iterator;
    iterator begin() const { return getElements().begin(); }
    iterator end() const { return getElements().end(); }
  }];
}

def RefImpl : FIRRTLImplType<"Ref", [], "::circt::firrtl::FIRRTLType"> {
  let summary = [{
    A reference type, such as `firrtl.ref<uint<1>>`.

    Used for remote reads and writes of the wrapped base type.

    Parameterized over the referenced base type,
    which must be passive and for now must also be ground.

    Not a base type.

    Values of this type are used to capture dataflow paths,
    and do not represent a circuit element or entity.
  }];
  let parameters = (ins "FIRRTLBaseType":$type);
  let genVerifyDecl = true;

  let skipDefaultBuilders = true;
  let builders = [
    TypeBuilderWithInferredContext<(ins "FIRRTLBaseType":$type)>
  ];
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLTYPESIMPL_TD
