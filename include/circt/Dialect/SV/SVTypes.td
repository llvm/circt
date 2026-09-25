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

class SVType<string name> : TypeDef<SVDialect, name> { }

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
// Net type and related helpers.
//===----------------------------------------------------------------------===//

def NetType : SVType<"Net"> {
  let summary = "SystemVerilog net handle";
  let description = [{
    Represents a SystemVerilog net data object handle (e.g. a structural wire).

    See SystemVerilog Spec 6.5, 6.6, and 6.7

  }];
  let mnemonic = "net";
  let parameters = (ins "::mlir::Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];

  let assemblyFormat = "`<` $elementType `>`";
}

// Type constraint that indicates that an operand or result may only be a valid
// net element type.
def NetElementType : Type<CPred<"hw::isHWValueType($_self)">,
                          "a valid net element">;

// Constrains `netValue` to `!sv.net<T>`, inferring the net handle from element type `T`.
class NetTypeConstraint<string value, string netValue>
  : TypesMatchWith<"type should be `!sv.net<T>`",
                   value, netValue,
                   "NetType::get($_self.getContext(), $_self)">;

// Constrains `value` to type `T`, inferring the element type from net handle `netValue` (`!sv.net<T>`).
class NetElementConstraint<string value, string netValue>
  : TypesMatchWith<"type should be net's element type",
                   netValue, value, "sv::getNetElementType($_self)">;

//===----------------------------------------------------------------------===//
// Var type and related helpers.
//===----------------------------------------------------------------------===//

def VarType : SVType<"Var"> {
  let summary = "SystemVerilog variable handle";
  let description = [{
    Represents a SystemVerilog variable data object handle (e.g., an assignable storage location).

    See SystemVerilog Spec 6.5 and 6.8
  }];
  let mnemonic = "var";
  let parameters = (ins "::mlir::Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];

  let assemblyFormat = "`<` $elementType `>`";
}

// Type constraint that indicates that an operand/result may only be a valid
// var element type.
def VarElementType : Type<CPred<"hw::isHWValueType($_self)">,
                          "a valid var element">;

// Constrains `varValue` to `!sv.var<T>`, inferring the var handle from element type `T`.
class VarTypeConstraint<string value, string varValue>
  : TypesMatchWith<"type should be `!sv.var<T>`",
                   value, varValue,
                   "VarType::get($_self.getContext(), $_self)">;

// Constrains `value` to type `T`, inferring the element type from var handle `varValue` (`!sv.var<T>`).
class VarElementConstraint<string value, string varValue>
  : TypesMatchWith<"type should be var's element type",
                   varValue, value, "sv::getVarElementType($_self)">;

//===----------------------------------------------------------------------===//
// UnpackedOpenArrayType type 
//===----------------------------------------------------------------------===//

def UnpackedOpenArrayTypeImpl : SVType<"UnpackedOpenArray"> {
  let summary = "SystemVerilog unpacked 'open' array";
  let description = [{
    This type represents unpacked 'open' array. Open array is a special type that can be
    used as formal arguments of DPI import statements.

    See SystemVerilog Spec 35.5.6.1.
  }];

  let mnemonic = "open_uarray";
  let parameters = (ins "::mlir::Type":$elementType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      auto *ctx = elementType.getContext();
      return $_get(ctx, elementType);
    }]>
  ];

  let assemblyFormat = "`<` $elementType `>`";
}

def UnpackedOpenArrayType : DialectType<SVDialect,
    CPred<"::circt::hw::type_isa<circt::sv::UnpackedOpenArrayType>($_self)">,
          "an Unpacked Open ArrayType",
          "::circt::hw::TypeAliasOr<circt::sv::UnpackedOpenArrayType>">;

#endif // CIRCT_DIALECT_SV_SVTYPES
