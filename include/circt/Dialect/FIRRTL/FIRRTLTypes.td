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

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLTYPES_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLTYPES_TD

include "FIRRTLDialect.td"

//===----------------------------------------------------------------------===//
// FIRRTL Types Definitions
//===----------------------------------------------------------------------===//

class FIRRTLDialectType<Pred pred, string summary, string cpp, string desc = "">
  : DialectType<FIRRTLDialect, pred, summary, cpp> {
  let description = desc;
}

def FIRRTLType : FIRRTLDialectType<CPred<"$_self.isa<FIRRTLType>()">,
  "FIRRTLType", "::circt::firrtl::FIRRTLType", [{
    Any FIRRTL dialect type, represented by FIRRTLType.
  }]>;

def FIRRTLBaseType : FIRRTLDialectType<
  CPred<"isa<FIRRTLBaseType>($_self)">,
  "a base type", "::circt::firrtl::FIRRTLBaseType", [{
    A base FIRRTL type, such as a clock, integer, or wire.

    Base types represent circuit elements and constructs,
    examples include ClockType, IntType, or BundleType.
    Nearly all FIRRTL types are base types.

    All base types are FIRRTLType's, and inherit from FIRRTLBaseType.
  }]>;

def ForeignType : FIRRTLDialectType<CPred<"!$_self.isa<FIRRTLType>()">,
                                    "foreign type", "::mlir::Type">;

def ClockType :
  FIRRTLDialectType<CPred<"firrtl::type_isa<ClockType>($_self)">, "clock",
                    "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::ClockType>">;

def NonConstClockType :
  FIRRTLDialectType<CPred<"firrtl::type_isa<ClockType>($_self)">, "clock",
                    "::circt::firrtl::ClockType">,
  BuildableType<"::circt::firrtl::ClockType::get($_builder.getContext())">;

def IntType : FIRRTLDialectType<CPred<"firrtl::type_isa<IntType>($_self)">,
 "sint or uint type", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::IntType>">;

def SIntType : FIRRTLDialectType<CPred<"firrtl::type_isa<SIntType>($_self)">,
 "sint type", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::SIntType>">;

def UIntType : FIRRTLDialectType<CPred<"firrtl::type_isa<UIntType>($_self)">,
 "uint type", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::UIntType>">;

def AnalogType : FIRRTLDialectType<CPred<"firrtl::type_isa<AnalogType>($_self)">,
 "analog type", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::AnalogType>">;

def BundleType : FIRRTLDialectType<CPred<"firrtl::type_isa<BundleType>($_self)">,
 "BundleType", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::BundleType>">;
def OpenBundleType : FIRRTLDialectType<CPred<"firrtl::type_isa<OpenBundleType>($_self)">,
 "OpenBundleType", "::circt::firrtl::OpenBundleType">;

def FVectorType : FIRRTLDialectType<CPred<"firrtl::type_isa<FVectorType>($_self)">,
  "FVectorType", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::FVectorType>">;
def OpenVectorType : FIRRTLDialectType<CPred<"isa<OpenVectorType>($_self)">,
  "OpenVectorType", "::circt::firrtl::OpenVectorType">;

def FEnumType : FIRRTLDialectType<CPred<"firrtl::type_isa<FEnumType>($_self)">,
 "FEnumType", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::FEnumType>">;

def StringType : FIRRTLDialectType<CPred<"$_self.isa<StringType>()">,
  "StringType", "::circt::firrtl::StringType">;

def AggregateType : FIRRTLDialectType<
  CPred<"firrtl::type_isa<FVectorType, BundleType, FEnumType>($_self)">,
  "a aggregate type", "::circt::firrtl::FIRRTLBaseType">;

def SizedType : FIRRTLDialectType<CPred<"isa<FIRRTLBaseType>($_self) && "
    "!cast<FIRRTLBaseType>($_self).hasUninferredWidth()">,
    "a sized type (contains no uninferred widths)", "::circt::firrtl::FIRRTLType">;
def SizedOrForeignType : AnyTypeOf<[SizedType, ForeignType]>;

def AsyncResetType : FIRRTLDialectType<
    CPred<"firrtl::type_isa<AsyncResetType>($_self)">,
    "AsyncReset", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::AsyncResetType>">;

def ResetType : FIRRTLDialectType<
    CPred<"firrtl::type_isa<ResetType>($_self)">,
    "Reset", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::ResetType>">;

def PassiveType : FIRRTLDialectType<
  CPred<"isa<FIRRTLBaseType>($_self) && cast<FIRRTLBaseType>($_self).isPassive()">,
  "a passive base type (contain no flips)", "::circt::firrtl::FIRRTLBaseType">;

def RefType : FIRRTLDialectType<
  CPred<"isa<RefType>($_self)">,
   "reference type", "::circt::firrtl::RefType">;

def RWProbe : FIRRTLDialectType<
  CPred<"isa<RefType>($_self) && cast<RefType>($_self).getForceable()">,
   "rwprobe type", "::circt::firrtl::RefType">;

// TODO: When Refs can appear within Base, need to disallow that too.
def ConnectableType : AnyTypeOf<[FIRRTLBaseType, ForeignType, StringType]>;
def StrictConnectableType : AnyTypeOf<[PassiveType, ForeignType]>;

//===----------------------------------------------------------------------===//
// Sized and Unsized Integers
//===----------------------------------------------------------------------===//

def UnsizedUIntType :
  FIRRTLDialectType<
    CPred<"firrtl::type_isa<UIntType>($_self) && "
          "firrtl::type_cast<UIntType>($_self).getWidth() == std::nullopt">,
    "uint with uninferred width", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::UIntType>">;

class SizedUIntType<int width> : FIRRTLDialectType<
    CPred<"firrtl::type_isa<UIntType>($_self) && "
          "firrtl::type_cast<UIntType>($_self).getWidth() == " # width>,
    width # "-bit uint", "::circt::firrtl::BaseTypeAliasOr<::circt::firrtl::UIntType>">;

class NonConstSizedUIntType<int width> :
  SizedUIntType<width>,
  BuildableType<
    "::circt::firrtl::UIntType::get($_builder.getContext(), " # width # ")">;

def UInt1Type : SizedUIntType<1>;
def UInt32Type : SizedUIntType<32>;
def NonConstUInt1Type : NonConstSizedUIntType<1>;

def UInt1OrUnsizedType : AnyTypeOf<[UInt1Type, UnsizedUIntType]>;

//===----------------------------------------------------------------------===//
// FIRRTL Types Predicates
//===----------------------------------------------------------------------===//

def OneBitType : FIRRTLDialectType<
 CPred<"(firrtl::type_isa<IntType>($_self) && firrtl::type_cast<IntType>($_self).getWidth() == 1) || "
   "(firrtl::type_isa<AnalogType>($_self) && firrtl::type_cast<AnalogType>($_self).getWidth() == 1)">,
 "UInt<1>, SInt<1>, or Analog<1>", "::circt::firrtl::FIRRTLBaseType">;


def AnyResetType : FIRRTLDialectType<
    CPred<"isa<FIRRTLBaseType>($_self) && cast<FIRRTLBaseType>($_self).isResetType()">,
    "Reset", "::circt::firrtl::FIRRTLBaseType">;

def AnyRegisterType : FIRRTLDialectType<
    CPred<"isa<FIRRTLBaseType>($_self) && "
          "cast<FIRRTLBaseType>($_self).isRegisterType()">,
    "a passive non-'const' base type that does not contain analog",
    "::circt::firrtl::FIRRTLBaseType">;

def UIntSIntClockType : AnyTypeOf<[SIntType, UIntType, ClockType],
                                  "sint, uint, or clock",
                                  "::circt::firrtl::FIRRTLBaseType">;

def 1DVecUIntType : FIRRTLDialectType<
  CPred<"firrtl::type_isa<FVectorType>($_self) && isa<UIntType>(firrtl::type_cast<FVectorType>($_self).getElementType())">,
  "1d vector with UInt element type", "::circt::firrtl::FIRRTLBaseType">;

def 1DVecIntType : FIRRTLDialectType<
  CPred<"firrtl::type_isa<FVectorType>($_self) && isa<IntType>(firrtl::type_cast<FVectorType>($_self).getElementType())">,
  "1d vector with Int element type", "::circt::firrtl::FIRRTLBaseType">;

def OneBitCastableType : AnyTypeOf<
  [OneBitType, AnyResetType, AsyncResetType, ClockType],
  "1-bit uint/sint/analog, reset, asyncreset, or clock",
                                  "::circt::firrtl::FIRRTLBaseType">;

//===----------------------------------------------------------------------===//
// Constraints on RefOps
//===----------------------------------------------------------------------===//

class RefTypeConstraint<string ref, string base>
  : TypesMatchWith<"reference base type should match",
                   ref, base,
                   "cast<RefType>($_self).getType()">;

class RefResultTypeConstraint<string base, string ref>
  : TypesMatchWith<"reference base type should match",
                   base, ref,
                   "RefType::get(cast<FIRRTLBaseType>($_self).getPassiveType())">;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLTYPES_TD
