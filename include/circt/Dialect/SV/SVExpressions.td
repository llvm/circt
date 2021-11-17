//===- SVExpressions.td - SV expression ops ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemVerilog expressions.
//
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/InferTypeOpInterface.td"

def AnySignlessIntegerOrInOutType : AnyTypeOf<[AnySignlessInteger, InOutType]>;
def HasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface,
                         ["getAsmResultNames"]>;

def VerbatimExprOp : SVOp<"verbatim.expr", [NoSideEffect, HasCustomSSAName]> {
  let summary = "Expression that expands to a value given SystemVerilog text";
  let description = [{
    This operation produces a typed value expressed by a string of
    SystemVerilog.  This can be used to access macros and other values that are
    only sensible as Verilog text.

    The text string is expected to have the highest precedence, so you should
    include parentheses in the string if it isn't a single token.  This is also
    assumed to not have side effects (use sv.verbatim.expr.se) if you need them.

    sv.verbatim.expr allows operand substitutions with {{0}} syntax.
    }];

  let arguments = (ins StrAttr:$string, Variadic<AnyType>:$operands,
                  DefaultValuedAttr<NameRefArrayAttr,"{}">:$symbols);
  let results = (outs AnySignlessIntegerOrInOutType:$result);
  let assemblyFormat = [{
    $string (`(` $operands^ `)`)?
      `:` functional-type($operands, $result) attr-dict 
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Twine":$string,
                CArg<"ValueRange", "{}">:$operands),
               "build(odsBuilder, odsState, resultType, "
               "odsBuilder.getStringAttr(string), operands," 
               "odsBuilder.getArrayAttr({}));">,
    OpBuilder<(ins "Type":$resultType, "StringAttr":$string,
                CArg<"ValueRange", "{}">:$operands),
               "build(odsBuilder, odsState, resultType, "
               "string, operands," 
               "odsBuilder.getArrayAttr({}));">
  ];
}

def VerbatimExprSEOp : SVOp<"verbatim.expr.se", [HasCustomSSAName]> {
  let summary = "Expression that expands to a value given SystemVerilog text";
  let description = [{
    This operation produces a typed value expressed by a string of
    SystemVerilog.  This can be used to access macros and other values that are
    only sensible as Verilog text.

    The text string is expected to have the highest precedence, so you should
    include parentheses in the string if it isn't a single token.  This is
    allowed to have side effects.

    sv.verbatim.se.expr allows operand substitutions with {{0}} syntax.
    }];

  let arguments = (ins StrAttr:$string, Variadic<AnyType>:$operands,
                 DefaultValuedAttr<NameRefArrayAttr,"{}">:$symbols );
  let results = (outs AnySignlessIntegerOrInOutType:$result);
  let assemblyFormat = [{
    $string (`(` $operands^ `)`)?
      `:` functional-type($operands, $result) attr-dict 
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Twine":$string),
               "build(odsBuilder, odsState, resultType, "
               "odsBuilder.getStringAttr(string), ValueRange{},"
               "odsBuilder.getArrayAttr({}));">
  ];
}

def ConstantXOp : SVOp<"constantX", [NoSideEffect, HasCustomSSAName]> {
  let summary = "A constant of value 'x'";
  let description = [{
    This operation produces a constant value of 'x'.  This 'x' follows the
    System Verilog rules for 'x' propagation.
    }];

  let arguments = (ins);
  let results = (outs HWIntegerType:$result);
  let assemblyFormat = " attr-dict `:` type($result)";
}

def ConstantZOp : SVOp<"constantZ", [NoSideEffect, HasCustomSSAName]> {
  let summary = "A constant of value 'z'";
  let description = [{
    This operation produces a constant value of 'z'.  This 'z' follows the
    System Verilog rules for 'z' propagation.
    }];

  let arguments = (ins);
  let results = (outs HWIntegerType:$result);
  let assemblyFormat = " attr-dict `:` type($result)";
}

def LocalParamOp : SVOp<"localparam",
      [FirstAttrDerivedResultType, NoSideEffect, HasCustomSSAName]> {
  let summary = "Declare a localparam";
  let description = [{
    The localparam operation produces a `localparam` declaration. See SV spec
    6.20.4 p125.
    }];

  let arguments = (ins AnyAttr:$value, StrAttr:$name);
  let results = (outs HWValueType:$result);

  let assemblyFormat = [{
    `:` type($result) custom<ImplicitSSAName>(attr-dict)
  }];

  let verifier = "return ::verify$cppClass(*this);";
}
