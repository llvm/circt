//===- SVDeclarations.td - SV declaration ops --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for System Verilog declarations.
//
//===----------------------------------------------------------------------===//

def RegOp : SVOp<"reg", [DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "Define a new `reg` in SystemVerilog";
  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs InOutType:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "::mlir::Type":$elementType,
                      CArg<"StringAttr", "StringAttr()">:$name)>
  ];

  // We handle the name in a custom way, so we use a customer parser/printer.
  let printer = "printRegOp(p, *this);";
  let parser = "return parseRegOp(parser, result);";
}

