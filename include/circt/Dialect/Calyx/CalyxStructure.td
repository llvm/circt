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
          component @MyComponent(in1: 32) -> (out1: 32) {
            ...
          }
        ```
    }];
    let arguments =
                (ins SymbolRefAttr:$name, DictionaryAttr:$inPortToWidth, DictionaryAttr:$outPortToWidth);
    let printer = "return ::print$cppClass(p, *this);";
    let parser = "return ::parse$cppClass(parser, result);";
}
