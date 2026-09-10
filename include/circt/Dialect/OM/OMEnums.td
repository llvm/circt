//===- OMEnums.td - OM Enum Definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of OM enum attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMENUMS_TD
#define CIRCT_DIALECT_OM_OMENUMS_TD

include "circt/Dialect/OM/OMDialect.td"
include "mlir/IR/EnumAttr.td"

let cppNamespace = "::circt::om" in {

//===----------------------------------------------------------------------===//
// Target Kinds
//===----------------------------------------------------------------------===//

def DontTouch : I32EnumAttrCase<"DontTouch", 0, "dont_touch">;
def Instance : I32EnumAttrCase<"Instance", 1, "instance">;
def MemberInstance: I32EnumAttrCase<"MemberInstance", 2, "member_instance">;
def MemberReference: I32EnumAttrCase<"MemberReference", 3, "member_reference">;
def Reference: I32EnumAttrCase<"Reference", 4, "reference">;
def TargetKind : I32EnumAttr<"TargetKind", "object model target kind",
  [DontTouch, Instance, MemberInstance, MemberReference, Reference]>  {}

}

#endif // CIRCT_DIALECT_OM_OMENUMS_TD
