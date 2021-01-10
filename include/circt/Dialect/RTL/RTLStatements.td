//===- RTLStatements.td - RTL statements ops ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for statements.
//
//===----------------------------------------------------------------------===//

def ConnectOp : RTLOp<"connect", [InOutTypeConstraint<"src", "dest">]> {
  let summary = "Connect two signals";
  let description = [{
    Connect Operation:
    ```
      rtl.connect %dest, %src : t1
    ```
    }];

  let arguments = (ins InOutType:$dest, RTLValueType:$src);
  let results = (outs);
    let assemblyFormat = "$dest `,` $src  attr-dict `:` type($src)";
}

def WireOp : RTLOp<"wire", [DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "Define a new wire";
  let description = [{
    Declare a new wire:
    ```
      %name = rtl.wire : !rtl.inout<t1>
    ```
    }];

  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs InOutType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "::mlir::Type":$elementType,
                      CArg<"StringAttr", "StringAttr()">:$name)>,
    OpBuilderDAG<(ins "::mlir::Type":$elementType,
                      CArg<"StringRef">:$name), [{
      return build($_builder, $_state, elementType,
                   $_builder.getStringAttr(name));
    }]>
  ];

  // We handle the name in a custom way, so we use a customer parser/printer.
  let printer = "printWireOp(p, *this);";
  let parser = "return parseWireOp(parser, result);";
}

// Expressions used for working with inout types.

def ReadInOutOp
 : RTLOp<"read_inout",
         [NoSideEffect, InOutElementConstraint<"result", "input">]> {
  let summary = "Get the value of from something of inout type (e.g. a wire or"
                " inout port) as the value itself.";
  let arguments = (ins InOutType:$input);
  let results = (outs RTLValueType:$result);

  let builders = [
    OpBuilderDAG<(ins "Value":$input)>
  ];

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

class InOutIndexConstraint<string value, string inoutValue>
  : TypesMatchWith<"type should be element of inout type",
                   inoutValue, value,
                   "InOutType::get(rtl::getAnyRTLArrayElementType("
                                            "getInOutElementType($_self)))">;

def ArrayIndexOp
 : RTLOp<"arrayindex",
         [NoSideEffect, InOutIndexConstraint<"result", "input">]> {
  let summary = "Index an inout memory to produce an inout element";
  let description = "See SV Spec 11.5.2.";
  let arguments = (ins InOutArrayType:$input, RTLIntegerType:$index);
  let results = (outs InOutType:$result);

  let builders = [
    OpBuilderDAG<(ins "Value":$input, "Value":$index)>
  ];

  let assemblyFormat =
    "$input`[`$index`]` attr-dict `:` type($input) `,` type($index)";
}

def StructExplodeOp : RTLOp<"struct_explode", [NoSideEffect]> {
  let summary = "Expand a struct into its constituent parts.";
  let arguments = (ins StructType:$input);
  let results = (outs Variadic<RTLNonInOutType>:$result);

  let parser = "return ::parse$cppClass(parser, result);";
  let printer = "return ::print(p, *this);";
}


def StructCreateOp : RTLOp<"struct_create", [NoSideEffect]> {
  let summary = "Create a struct from constituent parts.";
  let arguments = (ins Variadic<RTLNonInOutType>:$input);
  let results = (outs StructType:$result);

  let parser = "return ::parse$cppClass(parser, result);";
  let printer = "return ::print(p, *this);";
}
