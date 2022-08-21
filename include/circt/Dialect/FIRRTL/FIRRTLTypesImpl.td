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
  : TypeDef<FIRRTLDialect, name, [], "::circt::firrtl::FIRRTLBaseType"> {}

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//
def ClockTypeImpl : FIRRTLType<"Clock"> {
  let summary = "Clock signal";
  let mnemonic = "clock";
  let assemblyFormat = "";
}

def ResetTypeImpl : FIRRTLType<"Reset"> {
  let summary = "Reset Signal";
  let mnemonic = "clock";
  let assemblyFormat = "";
}

def AsyncResetTypeImpl : FIRRTLType<"AsyncReset"> {
  let summary = "AsyncReset signal";
  let mnemonic = "clock";
  let assemblyFormat = "";
}

