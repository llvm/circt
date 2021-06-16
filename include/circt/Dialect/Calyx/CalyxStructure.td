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
    let printer = "return ::print$cppClass(p, *this);";
    let parser = "return ::parse$cppClass(parser, result);";
}

def CellsOp : CalyxOp<"cells", [HasParent<"ComponentOp">]> {
    let summary = "Calyx Cells";
    let description = [{
        The "calyx.cells" operation represents the cells of a Calyx component.
    }];
    let printer = "return ::print$cppClass(p, *this);";
    let parser = "return ::parse$cppClass(parser, result);";
}

def WiresOp : CalyxOp<"wires", [HasParent<"ComponentOp">]> {
    let summary = "Calyx Wires";
    let description = [{
        The "calyx.wires" operation represents the wires of a Calyx component.
    }];
    let printer = "return ::print$cppClass(p, *this);";
    let parser = "return ::parse$cppClass(parser, result);";
}

def ControlOp : CalyxOp<"control", [HasParent<"ComponentOp">]> {
    let summary = "Calyx Control";
    let description = [{
        The "calyx.control" operation represents the control of a Calyx component.
    }];
    let printer = "return ::print$cppClass(p, *this);";
    let parser = "return ::parse$cppClass(parser, result);";
}
