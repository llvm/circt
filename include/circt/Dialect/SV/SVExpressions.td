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

def TextualValueOp : SVOp<"textual_value", [NoSideEffect]> {
  let summary = "Expression that expands to a value given SystemVerilog text";
  let description = [{
    This operation produces a typed value expressed by a string of
    SystemVerilog.  This can be used to access macros and other values that are
    only sensible as Verilog text.

    The text string is expected to have the highest precedence, so it should
    be parenthesized if it isn't a single token.
    }];

  let arguments = (ins StrAttr:$string);
  let results = (outs AnySignlessInteger:$result);
  let assemblyFormat = "$string attr-dict `:` type($result)";
}

def ConstantXOp : SVOp<"constantX", [NoSideEffect]> {
  let summary = "A constant of value 'x'";
  let description = [{
    This operation produces a constant value of 'x'.  This 'x' follows the
    System Verilog rules for 'x' propagation.
    }];

  let arguments = (ins);
  let results = (outs RTLIntegerType:$result);
  let assemblyFormat = " attr-dict `:` type($result)";
}

def ConstantZOp : SVOp<"constantZ", [NoSideEffect]> {
  let summary = "A constant of value 'z'";
  let description = [{
    This operation produces a constant value of 'z'.  This 'z' follows the
    System Verilog rules for 'z' propagation.
    }];

  let arguments = (ins);
  let results = (outs RTLIntegerType:$result);
  let assemblyFormat = " attr-dict `:` type($result)";
}
