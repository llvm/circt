//===- FIRRTLTypeInterfaces.td - FIRRTL Type Interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the type interfaces of the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLTYPEINTERFACES_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def WidthQualifiedTypeTrait : NativeTypeTrait<"WidthQualifiedTypeTrait"> {
  let cppNamespace = "::circt::firrtl";
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLTYPEINTERFACES_TD
