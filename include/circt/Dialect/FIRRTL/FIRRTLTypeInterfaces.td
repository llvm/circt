//===- FIRRTLTypesImpl.td - FIRRTL data type interfaces ----*- tablegen -*-===//
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

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLTYPEINTERFACES_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLTYPEINTERFACES_TD

include "FIRRTLDialect.td"

//===----------------------------------------------------------------------===//
// Type Traits
//===----------------------------------------------------------------------===//

def WidthQualifiedTypeTrait : NativeTypeTrait<"WidthQualifiedTypeTrait"> {
  let cppNamespace = "::circt::firrtl";
}

def PhysicalType : TypeInterface<"PhysicalType"> {
  let description = [{
    Interface for describing physical types.  Types implementing this trait must
    be purely physical.  If they are aggregates, all member types are also
    physical.
  }];

  let methods = [
    InterfaceMethod<[{
      Return the width of a type.  -1 indicates uninferred.
    }], "int32_t", "getWidthOrSentinel", (ins)>,
  ];

  let cppNamespace = "::circt::firrtl";
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLTYPEINTERFACES_TD
