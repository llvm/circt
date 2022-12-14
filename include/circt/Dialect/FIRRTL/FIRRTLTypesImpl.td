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
