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

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLTYPESIMPL_TD
