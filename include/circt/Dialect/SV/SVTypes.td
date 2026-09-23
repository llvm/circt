//===- SVTypes.td - SV data type definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data types for the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVTYPES
#define CIRCT_DIALECT_SV_SVTYPES

include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/SV/SVDialect.td"
include "mlir/IR/AttrTypeBase.td"


//===----------------------------------------------------------------------===//
// InOut type and related helpers.
//===----------------------------------------------------------------------===//

// Type constraint that indicates that an operand/result may only be a valid
// inout element type.
def InOutElementType : Type<CPred<"hw::isHWValueType($_self)">,
                            "a valid inout element">;

// This type constraint is used by operations whose asm strings have a type "T"
// in the asmstring, which constraints an input or output to type `T`, and then
// constraints another to `!hw.inout<T>`.  The "value" and "inoutValue" are the
// name of the operand or result for the type sides of this.
//
// Note that the inoutValue must be constrained to being an InOutElementType to
// ensure that functions and other things aren't passed into this.
class InOutTypeConstraint<string value, string inoutValue>
  : TypesMatchWith<"type should be element of inout type",
                   value, inoutValue,
                   "InOutType::get($_self.getContext(), $_self)">;

// This type constraint indicates that the specified value matches the element
// type of the specified inoutValue.  It is the dual to InOutTypeConstraint used
// when inference needs to "go the other way".
//
// Note that the inoutValue must be constrained to being an InOutElementType to
// ensure that functions and other things aren't passed into this.
class InOutElementConstraint<string value, string inoutValue>
  : TypesMatchWith<"type should be element of inout type",
                   inoutValue, value, "sv::getInOutElementType($_self)">;


//===----------------------------------------------------------------------===//
// SVNet type and related helpers.
//===----------------------------------------------------------------------===//

def SVNetType : DialectType<SVDialect,
    CPred<"::circt::hw::type_isa<circt::sv::NetType>($_self)">,
    "an SV net type", "::circt::sv::NetType">;

// Type constraint that indicates that an operand or result may only be a valid
// net element type.
def NetElementType : Type<CPred<"hw::isHWValueType($_self)">,
                          "a valid net element">;

// This type constraint is used by operations whose asm strings have a type
// "T" in the asmstring, which constrains an input or output to type `T`, and
// then constrains another to `!sv.net<T>`.  The "value" and "netValue" are
// the name of the operand or result for the type sides of this.
//
// Note that the netValue must be constrained to being a NetElementType to
// ensure that functions and other things aren't passed into this.
class NetTypeConstraint<string value, string netValue>
  : TypesMatchWith<"type should be element of net type",
                   value, netValue,
                   "NetType::get($_self.getContext(), $_self)">;

// This type constraint indicates that the specified value matches the
// element type of the specified netValue.
//
// Note that the netValue must be constrained to being a NetElementType to
// ensure that functions and other things aren't passed into this.
class NetElementConstraint<string value, string netValue>
  : TypesMatchWith<"type should be element of net type",
                   netValue, value, "sv::getNetElementType($_self)">;

//===----------------------------------------------------------------------===//
// SVVar type and related helpers.
//===----------------------------------------------------------------------===//

def SVVarType : DialectType<SVDialect,
    CPred<"::circt::hw::type_isa<circt::sv::VarType>($_self)">,
    "an SV variable type", "::circt::sv::VarType">;

// Type constraint that indicates that an operand/result may only be a valid
// var element type.
def VarElementType : Type<CPred<"hw::isHWValueType($_self)">,
                          "a valid var element">;

// This type constraint is used by operations whose asm strings have a type
// "T" in the asmstring, which constrains an input or output to type `T`, and
// then constrains another to `!sv.var<T>`.  The "value" and "varValue" are
// the name of the operand or result for the type sides of this.
//
// Note that the varValue must be constrained to being a VarElementType to
// ensure that functions and other things aren't passed into this.
class VarTypeConstraint<string value, string varValue>
  : TypesMatchWith<"type should be element of var type",
                   value, varValue,
                   "VarType::get($_self.getContext(), $_self)">;

// This type constraint indicates that the specified value matches the
// element type of the specified varValue.
//
// Note that the varValue must be constrained to being a VarElementType to
// ensure that functions and other things aren't passed into this.
class VarElementConstraint<string value, string varValue>
  : TypesMatchWith<"type should be element of var type",
                   varValue, value, "sv::getVarElementType($_self)">;

//===----------------------------------------------------------------------===//
// UnpackedOpenArrayType type 
//===----------------------------------------------------------------------===//

def UnpackedOpenArrayType : DialectType<SVDialect,
    CPred<"::circt::hw::type_isa<circt::sv::UnpackedOpenArrayType>($_self)">,
          "an Unpacked Open ArrayType",
          "::circt::hw::TypeAliasOr<circt::sv::UnpackedOpenArrayType>">;

#endif // CIRCT_DIALECT_SV_SVTYPES
