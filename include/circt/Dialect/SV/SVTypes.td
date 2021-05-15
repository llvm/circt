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

class SVType<string name> : TypeDef<SVDialect, name> { }

//===----------------------------------------------------------------------===//
// Generic stuff shared with HW Dialect.
//===----------------------------------------------------------------------===//

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWIntegerType : Type<CPred<"rtl::isHWIntegerType($_self)">,
                          "an integer bitvector of one or more bits",
                          "::mlir::IntegerType">;

// Type constraint that indicates that an operand/result may only be a valid,
// known, non-directional type.
def HWValueType : Type<CPred<"rtl::isHWValueType($_self)">,
                        "a known primitive element">;

// Type constraint that indicates that an operand/result may only be a valid
// non-directional type.
def HWNonInOutType : Type<CPred<"!rtl::hasHWInOutType($_self)">,
                           "a type without inout">;

// FIXME: Shouldn't have to duplicate this from the HW dialect tblgen files.
def InOutType : Type<CPred<"$_self.isa<rtl::InOutType>()">, "InOutType",
                           "::circt::rtl::InOutType">;

//===----------------------------------------------------------------------===//
// InOut type and related helpers.
//===----------------------------------------------------------------------===//

// Type constraint that indicates that an operand/result may only be a valid
// inout element type.
def InOutElementType : Type<CPred<"rtl::isHWValueType($_self)">,
                            "a valid inout element">;

// This type constraint is used by operations whose asm strings have a type "T"
// in the asmstring, which constraints an input or output to type `T`, and then
// constraints another to `!rtl.inout<T>`.  The "value" and "inoutValue" are the
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
