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

def UnpackedOpenArrayType : DialectType<SVDialect,
    CPred<"::circt::hw::type_isa<circt::sv::UnpackedOpenArrayType>($_self)">,
          "an Unpacked Open ArrayType",
          "::circt::hw::TypeAliasOr<circt::sv::UnpackedOpenArrayType>">;

def SVNetType : DialectType<SVDialect,
    CPred<"::circt::hw::type_isa<circt::sv::NetType>($_self)">,
    "an SV net type", "::circt::sv::NetType">;

def SVVarType : DialectType<SVDialect,
    CPred<"::circt::hw::type_isa<circt::sv::VarType>($_self)">,
    "an SV variable type", "::circt::sv::VarType">;

def SVLvalueType : AnyTypeOf<[SVNetType, SVVarType]>;

class SVLvalueElementConstraint<string value, string lvalue>
  : TypesMatchWith<"type should be element of SV lvalue type",
                   lvalue, value, "sv::getLvalueElementType($_self)">;

def SVForceDestType : AnyTypeOf<[SVNetType, SVVarType, InOutType]>;

class SVForceDestElementConstraint<string value, string lvalue>
  : TypesMatchWith<"type should be element of SV force destination type",
                   lvalue, value, "sv::getForceDestElementType($_self)">;

class HWInOutToSVNetConstraint<string input, string result>
  : TypesMatchWith<"result should be an SV net of the inout element type",
                   input, result,
                   "sv::NetType::get(sv::getInOutElementType($_self))">;

class SVNetToHWInOutConstraint<string input, string result>
  : TypesMatchWith<"result should be an InOut of an SV net element type",
                   input, result,
                   "hw::InOutType::get($_self.getContext(), "
                   "cast<sv::NetType>($_self).getElementType())">;

#endif // CIRCT_DIALECT_SV_SVTYPES
