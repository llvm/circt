//===- FIRRTLTypeInterfaces.td - FIRRTL Type Interfaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the type interfaces of the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLTYPEINTERFACES_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLTYPEINTERFACES_TD

include "mlir/IR/OpBase.td"

/// Trait for types that can implicitly belong to a phase.
def PhasedTypeInterface : TypeInterface<"PhasedTypeInterface"> {
  let cppNamespace = "::circt::firrtl";
  let methods = [
    InterfaceMethod<"Is this type compatible with the given phase",
      "bool", "isLegalInPhase", (ins "Phase":$phase)>,
    InterfaceMethod<"Is this type fully compatible with the hardware phase.",
      "bool", "isLegalHardware", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return static_cast<const ConcreteType *>(this)->isLegalInPhase(Phase::Hardware);
      }]
    >,
    InterfaceMethod<"Is this type fully compatible with the property phase",
      "bool", "isLegalProperty", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return static_cast<const ConcreteType *>(this)->isLegalInPhase(Phase::Property);
      }]
    >
  ];
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLTYPEINTERFACES_TD
