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
class FIRRTLImplType<string name, list<Trait> traits = []>
  : TypeDef<FIRRTLDialect, name, traits, "::circt::firrtl::FIRRTLBaseType"> {}

def WidthQualifiedTrait : NativeTypeTrait<"WidthQualifiedTrait"> {
  let cppNamespace = "::circt::firrtl";
}

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//
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
  [WidthQualifiedTrait, FieldIDTypeInterface]> {
  let summary = "Analog signal";
  let parameters = (ins "int32_t":$baseWidth);
  let builders = [
    TypeBuilder<(ins "int32_t":$baseWidth)>,
    TypeBuilder<(ins)>,
  ];

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
    size_t getFieldID(size_t index);

    /// Find the element index corresponding to the desired fieldID.  If the
    /// fieldID corresponds to a field in nested under an element, it will
    /// return the index of the parent element.
    size_t getIndexForFieldID(size_t fieldID);

    /// Strip off a single layer of this type and return the sub-type and a
    /// field ID targeting the same field, but rebased on the sub-type.
    std::pair<FIRRTLBaseType, size_t> getSubTypeByFieldID(size_t fieldID);

    /// Get the maximum field ID in this vector.  This is helpful for
    /// constructing field IDs when this VectorType is nested in another
    /// aggregate type.
    size_t getMaxFieldID();

    /// Returns the effective field id when treating the index field as the root
    /// of the type.  Essentially maps a fieldID to a fieldID after a subfield
    /// op. Returns the new id and whether the id is in the given child.
    std::pair<size_t, bool> rootChildFieldID(size_t fieldID, size_t index);
  }];
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLTYPESIMPL_TD
