//===- FIRRTLEnums.td - FIRRTL Enum Definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of FIRRTL enum attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLENUMS_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLENUMS_TD

include "FIRRTLDialect.td"
include "mlir/IR/EnumAttr.td"

let cppNamespace = "::circt::firrtl" in {

//===----------------------------------------------------------------------===//
// Name Kinds
//===----------------------------------------------------------------------===//

/// An attribute describing a kind of a name which defines name preservation
/// semantics. We preserve names tagged with `InterestingName`.
def DroppableName: I32EnumAttrCase<"DroppableName", 0, "droppable_name">;
def InterestingName: I32EnumAttrCase<"InterestingName", 1, "interesting_name">;

def NameKindEnumImpl: I32EnumAttr<"NameKindEnum", "name kind",
              [DroppableName, InterestingName]> {
  let genSpecializedAttr = 0;
}

def NameKindAttr: EnumAttr<FIRRTLDialect, NameKindEnumImpl, "name_kind">;

//===----------------------------------------------------------------------===//
// Module Instantiation Conventions
//===----------------------------------------------------------------------===//

def Convention : I32EnumAttr<"Convention", "lowering convention", [
                           I32EnumAttrCase<"Internal", 0, "internal">,
                           I32EnumAttrCase<"Scalarized", 1, "scalarized">]> {
  let genSpecializedAttr = 0;
}

def ConventionAttr : EnumAttr<FIRRTLDialect, Convention, "convention">;

//===----------------------------------------------------------------------===//
// Layer Lowering Conventions
//===----------------------------------------------------------------------===//

def LayerConvention : I32EnumAttr<"LayerConvention", "layer convention", [
  I32EnumAttrCase<"Bind", 0, "bind">,
  I32EnumAttrCase<"Inline", 1, "inline">
]> {

  let genSpecializedAttr = 0;
}

def LayerConventionAttr : EnumAttr<FIRRTLDialect, LayerConvention, "layerconvention">;

//===----------------------------------------------------------------------===//
// Layer Specialization
//===----------------------------------------------------------------------===//

def LayerSpecialization :
    I32EnumAttr<"LayerSpecialization", "layer specialization", [
  I32EnumAttrCase<"Enable", 0, "enable">,
  I32EnumAttrCase<"Disable", 1, "disable">
]> {
  let genSpecializedAttr = 0;
}

def LayerSpecializationAttr :
    EnumAttr<FIRRTLDialect, LayerSpecialization, "layerspecialization">;

//===----------------------------------------------------------------------===//
// Read Under Write Behaviour
//===----------------------------------------------------------------------===//

def RUW_Undefined: I32EnumAttrCase<"Undefined", 0>;
def RUW_Old: I32EnumAttrCase<"Old", 1>;
def RUW_New: I32EnumAttrCase<"New", 2>;

def RUWAttr: I32EnumAttr<"RUWAttr", "Read Under Write Enum",
                         [RUW_Undefined, RUW_Old, RUW_New]>;

//===----------------------------------------------------------------------===//
// Memory Port Direction
//===----------------------------------------------------------------------===//

def MemDir_Infer    : I32EnumAttrCase<"Infer", 0>;
def MemDir_Read     : I32EnumAttrCase<"Read", 1>;
def MemDir_Write    : I32EnumAttrCase<"Write", 2>;
def MemDir_ReadWrite: I32EnumAttrCase<"ReadWrite", 3>;

def MemDirAttr: I32EnumAttr<"MemDirAttr", "Memory Direction Enum",
                            [MemDir_Infer, MemDir_Read, MemDir_Write,
                              MemDir_ReadWrite]>;

//===----------------------------------------------------------------------===//
// Event Control
//===----------------------------------------------------------------------===//

/// AtPosEdge triggers on a rise from 0 to 1/X/Z, or X/Z to 1.
def AtPosEdge: I32EnumAttrCase<"AtPosEdge", 0, "posedge">;
/// AtNegEdge triggers on a drop from 1 to 0/X/Z, or X/Z to 0.
def AtNegEdge: I32EnumAttrCase<"AtNegEdge", 1, "negedge">;
/// AtEdge(v) is syntactic sugar for "AtPosEdge(v) or AtNegEdge(v)".
def AtEdge   : I32EnumAttrCase<"AtEdge", 2, "edge">;

def EventControlAttr : I32EnumAttr<"EventControl", "edge control trigger",
                                   [AtPosEdge, AtNegEdge, AtEdge]>  {}

//===----------------------------------------------------------------------===//
// OM Targets
//===----------------------------------------------------------------------===//

def DontTouch : I32EnumAttrCase<"DontTouch", 0, "dont_touch">;
def Instance : I32EnumAttrCase<"Instance", 1, "instance">;
def MemberInstance: I32EnumAttrCase<"MemberInstance", 2, "member_instance">;
def MemberReference: I32EnumAttrCase<"MemberReference", 3, "member_reference">;
def Reference: I32EnumAttrCase<"Reference", 4, "reference">;
def TargetKind : I32EnumAttr<"TargetKind", "object model target kind",
  [DontTouch, Instance, MemberInstance, MemberReference, Reference]>  {}

//===----------------------------------------------------------------------===//
// Printf Substitutions Attributes
//===----------------------------------------------------------------------===//

def PrintfSubstitutionEnum : I32EnumAttr<
  "PrintfSubstitution",
  "A special substitution for a FIRRTL printf",
  [
    I32EnumAttrCase<"SimulationTime", 0, "SimulationTime">
  ]
> {

  let genSpecializedAttr = 0;

}

}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLENUMS_TD
