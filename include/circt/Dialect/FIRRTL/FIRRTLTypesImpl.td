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
class FIRRTLImplType<string name, list<Trait> traits = []> 
  : TypeDef<FIRRTLDialect, name, traits, "::circt::firrtl::FIRRTLBaseType"> {}

def WidthQualifiedTrait : NativeTypeTrait<"WidthQualifiedTrait"> {
  let cppNamespace = "::circt::firrtl";
}

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//
def ClockTypeImpl : FIRRTLImplType<"Clock"> {
  let summary = "Clock signal";
  let mnemonic = "clock";
  let assemblyFormat = "";
}

def ResetTypeImpl : FIRRTLImplType<"Reset"> {
  let summary = "Reset Signal";
  let mnemonic = "reset";
  let assemblyFormat = "";
}

def AsyncResetTypeImpl : FIRRTLImplType<"AsyncReset"> {
  let summary = "AsyncReset signal";
  let mnemonic = "asyncreset";
  let assemblyFormat = "";
}

def AnalogTypeImpl : FIRRTLImplType<"Analog", [WidthQualifiedTrait]> {
  let summary = "Analog signal";
  let mnemonic = "analog";
  let parameters = (ins "int32_t":$baseWidth);
  let assemblyFormat = "`<` $baseWidth `>`";
  let builders = [
    TypeBuilder<(ins "int32_t":$baseWidth)>,
    TypeBuilder<(ins)>,
  ];

}
