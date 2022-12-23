//===- HWTypeInterfaces.td - HW Type Interfaces ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the type interfaces of the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWTYPEINTERFACES_TD
#define CIRCT_DIALECT_HW_HWTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

def FieldIDTypeInterface : TypeInterface<"FieldIDTypeInterface"> {
  let cppNamespace = "circt::hw";
  let description = [{
    Common methods for types which can be indexed by a FieldID.
  }];

  let methods = [
    InterfaceMethod<"Get the maximum field ID for this type",
      "uint64_t", "getMaxFieldID">
  ];
}

#endif // CIRCT_DIALECT_HW_HWTYPEINTERFACES_TD
