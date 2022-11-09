//===- FIRRTLTypes.td - FIRRTL Types Definition ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the FIRRTL types.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// FIRRTL Types Definitions
//===----------------------------------------------------------------------===//

class FIRRTLDialectType<Pred pred, string summary, string cpp, string desc = "">
  : DialectType<FIRRTLDialect,pred, summary, cpp> {
  let description = desc;
}

def FIRRTLType : FIRRTLDialectType<CPred<"$_self.isa<FIRRTLType>()">,
  "FIRRTLType", "::circt::firrtl::FIRRTLType", [{
    Any FIRRTL dialect type, represented by FIRRTLType.
  }]>;

def FIRRTLBaseType : FIRRTLDialectType<
  CPred<"$_self.isa<FIRRTLBaseType>()">,
  "a base type", "::circt::firrtl::FIRRTLBaseType", [{
    A base FIRRTL type, such as a clock, integer, or wire.

    Base types represent circuit elements and constructs,
    examples include ClockType, IntType, or BundleType.
    Nearly all FIRRTL types are base types.

    All base types are FIRRTLType's, and inherit from FIRRTLBaseType.
  }]>;

def ForeignType : FIRRTLDialectType<CPred<"!$_self.isa<FIRRTLType>()">,
                                    "foreign type", "::mlir::Type">;

def ClockType : FIRRTLDialectType<CPred<"$_self.isa<ClockType>()">,
    "clock", "::circt::firrtl::ClockType">,
  BuildableType<"ClockType::get($_builder.getContext())">;

def IntType : FIRRTLDialectType<CPred<"$_self.isa<IntType>()">,
 "sint or uint type", "::circt::firrtl::IntType">;

def SIntType : FIRRTLDialectType<CPred<"$_self.isa<SIntType>()">,
 "sint type", "::circt::firrtl::SIntType">;

def UIntType : FIRRTLDialectType<CPred<"$_self.isa<UIntType>()">,
 "uint type", "::circt::firrtl::UIntType">;

def AnalogType : FIRRTLDialectType<CPred<"$_self.isa<AnalogType>()">,
 "analog type", "::circt::firrtl::AnalogType">;

def BundleType : FIRRTLDialectType<CPred<"$_self.isa<BundleType>()">,
 "BundleType", "::circt::firrtl::BundleType">;

def FVectorType : FIRRTLDialectType<CPred<"$_self.isa<FVectorType>()">,
  "FVectorType", "::circt::firrtl::FVectorType">;

def AggregateType : FIRRTLDialectType<
  Or<[
    CPred<"$_self.isa<FVectorType>()">,
    CPred<"$_self.isa<BundleType>()">
  ]>,
  "a aggregate type", "::circt::firrtl::FIRRTLBaseType">;

def ConnectableType : FIRRTLDialectType<
  Or<[
    CPred<"$_self.isa<FIRRTLBaseType>()">,
    CPred<"$_self.isa<RefType>()">,
    ]>,
    "a connectable type (base or ref type)", "::circt::firrtl::FIRRTLType", [{
    Any type that is valid for use in connect statements.
    Currently this is any base type or ref type.
  }]>;

def SizedBaseOrRefType : FIRRTLDialectType<
  Or<[
    CPred<"$_self.isa<FIRRTLBaseType>() && "
          "!$_self.cast<FIRRTLBaseType>().hasUninferredWidth()">,
    CPred<"$_self.isa<RefType>() && "
          "!$_self.cast<RefType>().getType().hasUninferredWidth()">,
    ]>,
  "a sized base or ref type (contains no uninferred widths)", "::circt::firrtl::FIRRTLType">;

def SizedType : FIRRTLDialectType<CPred<"$_self.isa<FIRRTLBaseType>() && "
    "!$_self.cast<FIRRTLBaseType>().hasUninferredWidth()">,
    "a sized type (contains no uninferred widths)", "::circt::firrtl::FIRRTLType">;
def SizedOrForeignType : AnyTypeOf<[SizedType, ForeignType]>;
def SizedBaseOrRefTypeOrForeignType : AnyTypeOf<[SizedBaseOrRefType, ForeignType]>;

def UInt1Type : FIRRTLDialectType<
    CPred<"$_self.isa<UIntType>() && "
          "($_self.cast<UIntType>().getWidth() == 1 ||"
          " $_self.cast<UIntType>().getWidth() == None)">,
    "UInt<1> or UInt", "::circt::firrtl::UIntType">,
  BuildableType<"UIntType::get($_builder.getContext(), 1)">;

def AsyncResetType : FIRRTLDialectType<
    CPred<"$_self.isa<AsyncResetType>()">,
    "AsyncReset", "::circt::firrtl::AsyncResetType">,
  BuildableType<"AsyncResetType::get($_builder.getContext())">;

def ResetType : FIRRTLDialectType<
    CPred<"$_self.isa<ResetType>()">,
    "Reset", "::circt::firrtl::ResetType">,
  BuildableType<"ResetType::get($_builder.getContext())">;

def PassiveType : FIRRTLDialectType<
  CPred<"$_self.isa<FIRRTLBaseType>() && $_self.cast<FIRRTLBaseType>().isPassive()">,
  "a passive base type (contain no flips)", "::circt::firrtl::FIRRTLBaseType">;

def RefType : FIRRTLDialectType<
   CPred<"$_self.isa<RefType>()">,
  "reference type", "::circt::firrtl::RefType", [{
    A reference type, such as `firrtl.ref<uint<1>>`.

    Used for remote reads and writes of the wrapped base type.

    Parameterized over the referenced base type,
    which must be passive and for now must also be ground.

    Not a base type.

    Values of this type are used to capture dataflow paths,
    and do not represent a circuit element or entity.
  }]>;

//===----------------------------------------------------------------------===//
// FIRRTL Types Predicates
//===----------------------------------------------------------------------===//

def OneBitType : FIRRTLDialectType<
 CPred<"($_self.isa<IntType>() && $_self.cast<IntType>().getWidth() == 1) || "
   "($_self.isa<AnalogType>() && $_self.cast<AnalogType>().getWidth() == 1)">,
 "UInt<1>, SInt<1>, or Analog<1>", "::circt::firrtl::FIRRTLBaseType">;


def AnyResetType : FIRRTLDialectType<
    CPred<"$_self.isa<FIRRTLBaseType>() && $_self.cast<FIRRTLBaseType>().isResetType()">,
    "Reset", "::circt::firrtl::FIRRTLBaseType">;

def AnyRegisterType : FIRRTLDialectType<
    CPred<"$_self.isa<FIRRTLBaseType>() && "
          "$_self.cast<FIRRTLBaseType>().isRegisterType()">,
    "a passive base type that does not contain analog",
    "::circt::firrtl::FIRRTLBaseType">;

def UIntSIntClockType : AnyTypeOf<[SIntType, UIntType, ClockType],
                                  "sint, uint, or clock",
                                  "::circt::firrtl::FIRRTLBaseType">;

def OneBitCastableType : AnyTypeOf<
  [OneBitType, AnyResetType, AsyncResetType, ClockType],
  "1-bit uint/sint/analog, reset, asyncreset, or clock",
                                  "::circt::firrtl::FIRRTLBaseType">;

//===----------------------------------------------------------------------===//
// FIRRTL Enum Definitions
//===----------------------------------------------------------------------===//

/// RUWAttr: Undefined/Old/New.
def RUW_Undefined: I32EnumAttrCase<"Undefined", 0>;
def RUW_Old: I32EnumAttrCase<"Old", 1>;
def RUW_New: I32EnumAttrCase<"New", 2>;

def RUWAttr: I32EnumAttr<"RUWAttr", "Read Under Write Enum",
                         [RUW_Undefined, RUW_Old, RUW_New]>;

def MemDir_Infer    : I32EnumAttrCase<"Infer", 0>;
def MemDir_Read     : I32EnumAttrCase<"Read", 1>;
def MemDir_Write    : I32EnumAttrCase<"Write", 2>;
def MemDir_ReadWrite: I32EnumAttrCase<"ReadWrite", 3>;

def MemDirAttr: I32EnumAttr<"MemDirAttr", "Memory Direction Enum",
                            [MemDir_Infer, MemDir_Read, MemDir_Write,
                              MemDir_ReadWrite]>;
