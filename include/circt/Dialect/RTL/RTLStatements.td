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

  let arguments = (ins InOutType:$dest, InOutElementType:$src);
  let results = (outs);
  
  let assemblyFormat = [{
    $dest `,` $src  attr-dict `:` type($src)
  }];
}

def RegOp : RTLOp<"reg", [DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "Define a new reg";
  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs InOutType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "::mlir::Type":$elementType,
                      CArg<"StringAttr", "StringAttr()">:$name)>
  ];

  // We handle the name in a custom way, so we use a customer parser/printer.
  let printer = [{ printRegWireOp(p, *this); }];
  let parser = [{ return parseRegWireOp(parser, result); }];
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
                      CArg<"StringAttr", "StringAttr()">:$name)>
  ];

  // We handle the name in a custom way, so we use a customer parser/printer.
  let printer = [{ printRegWireOp(p, *this); }];
  let parser = [{ return parseRegWireOp(parser, result); }];
}

def ReadInOutOp
 : RTLOp<"read_inout",
         [NoSideEffect, InOutElementConstraint<"result", "input">]> {
  let summary = "Get the value of from something of inout type (e.g. a wire or"
                " inout port) as the value itself.";
  let arguments = (ins InOutType:$input);
  let results = (outs InOutElementType:$result);

  let builders = [
    OpBuilderDAG<(ins "Value":$input)>
  ];

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}
