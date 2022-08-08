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

// Base class for other typedefs. Provides dialact-specific defaults.
class FIRRTLType<string name> 
  : TypeDef<FIRRTLDialect, name, [], "::circt::firrtl::FIRRTLType"> {}

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//

// A parameterized integer type. Declares the firrtl::UIntType in C++.
def UIntTypeImpl : FIRRTLType<"UInt"> {
  let summary = "parameterized-width integer";
  let description = [{
    Parameterized integer types are equivalent to the MLIR standard integer
    type: it is signless, and may be any width integer.  This type represents
    the case when the width is a parameter in the HW dialect sense.
  }];

  let mnemonic = "uint";
  let parameters = (ins "::mlir::Attribute":$width);
  let assemblyFormat = "`<` params `>`";
}

// A parameterized integer type. Declares the firrtl::SIntType in C++.
def SIntTypeImpl : FIRRTLType<"SInt"> {
  let summary = "parameterized-width integer";
  let description = [{
    Parameterized integer types are equivalent to the MLIR standard integer
    type: it is signless, and may be any width integer.  This type represents
    the case when the width is a parameter in the HW dialect sense.
  }];

  let mnemonic = "sint";
  let parameters = (ins "::mlir::Attribute":$width);
  let assemblyFormat = "`<` params `>`";
}

// A parameterized integer type. Declares the firrtl::UIntType in C++.
def AnalogTypeImpl : FIRRTLType<"Analog"> {
  let summary = "parameterized-width analog type";
  let description = [{
    Parameterized integer types are equivalent to the MLIR standard integer
    type: it is signless, and may be any width integer.  This type represents
    the case when the width is a parameter in the HW dialect sense.
  }];

  let mnemonic = "analog";
  let parameters = (ins "::mlir::Attribute":$width);
  let assemblyFormat = "`<` params `>`";
}

// A packed struct. Declares the hw::StructType in C++.
def BundleTypeImpl : FIRRTLType<"Bundle"> {
  let summary = "FIRRTL bundle type";
  let description = [{
    Represents a structure of name, value pairs.
    !hw.struct<fieldName1: Type1, fieldName2: Type2>
  }];
  let mnemonic = "bundle";

  let hasCustomAssemblyFormat = 1;

  let parameters = (
    ins ArrayRefParameter<
      "::circt::firrtl::detail::FieldInfo", "struct fields">: $elements
  );

  let extraClassDeclaration = [{
    using BundleElement = ::circt::firrtl::detail::FieldInfo;
    size_t getNumElements() { return getElements().size(); }

    /// Look up an element's index by name.  This returns None on failure.
    llvm::Optional<unsigned> getElementIndex(StringAttr name);
    llvm::Optional<unsigned> getElementIndex(StringRef name);

    /// Look up an element's name by index. This asserts if index is invalid.
    StringRef getElementName(size_t index);

    /// Look up an element by name.  This returns None on failure.
    llvm::Optional<BundleElement> getElement(StringAttr name);
    llvm::Optional<BundleElement> getElement(StringRef name);

    /// Look up an element by index.  This asserts if index is invalid.
    BundleElement getElement(size_t index);

    /// Look up an element type by name.
    FIRRTLType getElementType(StringAttr name);
    FIRRTLType getElementType(StringRef name);

    /// Look up an element type by index.
    FIRRTLType getElementType(size_t index);

    /// Get an integer ID for the field. Field IDs start at 1, and are assigned
    /// to each field in a bundle in a recursive pre-order walk of all fields,
    /// visiting all nested bundle fields.  A field ID of 0 is used to reference
    /// the bundle itself. The ID can be used to uniquely identify any specific
    /// field in this bundle.
    unsigned getFieldID(unsigned index);

    /// Find the element index corresponding to the desired fieldID.  If the
    /// fieldID corresponds to a field in a nested bundle, it will return the
    /// index of the parent field.
    unsigned getIndexForFieldID(unsigned fieldID);

    /// Strip off a single layer of this type and return the sub-type and a field
    /// ID targeting the same field, but rebased on the sub-type.
    std::pair<FIRRTLType, unsigned> getSubTypeByFieldID(unsigned fieldID);

    /// Get the maximum field ID in this bundle.  This is helpful for constructing
    /// field IDs when this BundleType is nested in another aggregate type.
    unsigned getMaxFieldID();

    /// Returns the effective field id when treating the index field as the root
    /// of the type.  Essentially maps a fieldID to a fieldID after a subfield op.
    /// Returns the new id and whether the id is in the given child.
    std::pair<unsigned, bool> rootChildFieldID(unsigned fieldID, unsigned index);

    using iterator = ArrayRef<BundleElement>::iterator;
    iterator begin() const { return getElements().begin(); }
    iterator end() const { return getElements().end(); }
  }];
}

// A simple fixed size array. Declares the hw::ArrayType in C++.
def FVectorTypeImpl : FIRRTLType<"FVector"> {
  let summary = "fixed-sized array";
  let description = [{
    Fixed sized HW arrays are roughly similar to C arrays. On the wire (vs.
    in a memory), arrays are always packed. Memory layout is not defined as
    it does not need to be since in silicon there is not implicit memory
    sharing.
  }];

  let mnemonic = "vector";
  let parameters = (ins "::mlir::Type":$elementType, "::mlir::Attribute":$sizeAttr);
//  let genVerifyDecl = 1;
  let assemblyFormat = "`<` params `>`";

  let extraClassDeclaration = [{
    FIRRTLType getElementType();
    size_t getNumElements();

    /// Get an integer ID for the field. Field IDs start at 1, and are assigned
    /// to each field in a vector in a recursive depth-first walk of all elements.
    /// A field ID of 0 is used to reference the vector itself.
    size_t getFieldID(size_t index);

    /// Find the element index corresponding to the desired fieldID.  If the
    /// fieldID corresponds to a field in nested under an element, it will return
    /// the index of the parent element.
    size_t getIndexForFieldID(size_t fieldID);

    /// Strip off a single layer of this type and return the sub-type and a field
    /// ID targeting the same field, but rebased on the sub-type.
    std::pair<FIRRTLType, size_t> getSubTypeByFieldID(size_t fieldID);

    /// Get the maximum field ID in this vector.  This is helpful for constructing
    /// field IDs when this VectorType is nested in another aggregate type.
    size_t getMaxFieldID();

    /// Returns the effective field id when treating the index field as the root
    /// of the type.  Essentially maps a fieldID to a fieldID after a subfield op.
    /// Returns the new id and whether the id is in the given child.
    std::pair<size_t, bool> rootChildFieldID(size_t fieldID, size_t index);
  }];
}


def ClockTypeImpl : FIRRTLType<"Clock"> {
  let summary = "Clock signal";
  let description = [{
  }];

  let mnemonic = "clock";
}

def AsyncResetTypeImpl : FIRRTLType<"AsyncReset"> {
  let summary = "Asyncronous reset signal";
  let description = [{
  }];

  let mnemonic = "asyncreset";
}

def ResetTypeImpl : FIRRTLType<"Reset"> {
  let summary = "Synchronous reset signal";
  let description = [{
  }];

  let mnemonic = "reset";
}
