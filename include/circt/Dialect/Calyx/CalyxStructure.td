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

def ComponentOp : CalyxOp<"component", [NoTerminator]> {
  let summary = "Calyx Component";
  let description = [{
    The "calyx.component" operation represents an overall
    Calyx component containing:
    (1) In- and output port definitions
        that define the interface.
    (2) Cells, wires, and control.
    ```
      component @MyComponent(in1: 32) -> (out1: 32) {
        cells { ... }
        wires { ... }
        control { ... }
      }
    ```
  }];
  let arguments = (ins
    SymbolRefAttr:$name,
    I64Attr:$inputPortCount
  );
  let regions = (region SizedRegion<1>: $body);


  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}
