//===- MSFT.td - MSFT dialect definition -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the MSFT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MSFT_TD
#define MSFT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MSFTDialect : Dialect {
  let name = "msft";

  let summary = "Microsoft internal support dialect";
  let description = [{
    Umbrella dialect for everything needed to support Microsoft development
    but not thoroughly discussed. Most (if not everything) in this dialect is
    a candidate for generalization and re-homing.
  }];

  let cppNamespace = "::circt::msft";
}

// Base class for the operation in this dialect.
class MSFTOp<string mnemonic, list<OpTrait> traits = []> :
    Op<MSFTDialect, mnemonic, traits>;

def DeviceType : I32EnumAttr<"DeviceType",
  "Type of device at physical location", [
    I32EnumAttrCase<"M20K", 1, "">,
    I32EnumAttrCase<"DSP", 2, "">,
    I32EnumAttrCase<"EC", 3, "">,
  ]>;

def PhysLocation : StructAttr<"PhysLocation", MSFTDialect, [
  // Device type.
  StructFieldAttr<"Type", DeviceType>,

  // X/Y coordinate of device.
  StructFieldAttr<"X", UI64Attr>,
  StructFieldAttr<"Y", UI64Attr>,

  // The resource number at the specified coordinates.
  StructFieldAttr<"Num", UI64Attr>,

  // The name of the entity within the op (to which this attribute is attached)
  // which this location is specifying.
  StructFieldAttr<"Entity", StrAttr>
]>;

#endif // MSFT_TD
