//===- CalyxStructure.td - Calyx Structure -------------*- tablegen -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the Calyx structures.
//
//===----------------------------------------------------------------------===//

def ComponentOp : CalyxOp<"component", []> {
    let summary = "Calyx Component";
    let description = [{
        The "calyx.component" operation represents an overall
        Calyx component containing cells, wires, and control.
        ```
    }];
    let arguments =
                (ins DictionaryAttr:$inPortToWidth, DictionaryAttr:$outPortToWidth);
    let printer = "return ::print$cppClass(p, *this);";
    let parser = "return ::parse$cppClass(parser, result);";
    let builders = [
      OpBuilder<(ins "StringAttr":$name,
                 CArg<"DictionaryAttr", "DictionaryAttr()">:$inPortToWidth,
                 CArg<"DictionaryAttr", "DictionaryAttr()">:$outPortToWidth)>
    ];
}

