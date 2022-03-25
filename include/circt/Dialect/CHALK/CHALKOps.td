//===- CHALKOps.td - Definition of CHALK dialect operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CHALK_CHALKOPS_TD
#define CIRCT_DIALECT_CHALK_CHALKOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def CellOp : CHALKOp<"cell", [
    SingleBlock,
    IsolatedFromAbove,
    NoTerminator
  ]> {
  let summary = "Define a 2D VLSI layout";
  let description = [{
    `chalk.cell` represents a 2D VLSI layout, where the region nests
    multiple shapes, paths, or placements.
  }];

  let arguments = (ins StrAttr:$cell_name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$cell_name attr-dict $body";
}

class CHALKCellShapeBase<string mnemonic, list<Trait> traits = []> :
    CHALKOp<mnemonic, traits> {
}

def RectangleOp : CHALKCellShapeBase<"rectangle", [
    HasParent<"CellOp">,
    SingleBlock,
    IsolatedFromAbove,
    NoTerminator
  ]> {
  let summary = "Define a state of a machine";
  let description = [{
    `chalk.rectangle` represents a rectangle in a 2D VLSI layout. It
    uses attributes to store the (x,y) coordinates and the height
    and width.
  }];
  let arguments = (ins
    StrAttr: $sym_name,
    I64Attr:$xCoord,
    I64Attr:$yCoord,
    UI32Attr:$width,
    UI32Attr:$height
  );
  let results = (outs);
  let regions = (region);

  let assemblyFormat = "$sym_name attr-dict";
}

#endif // CIRCT_DIALECT_CHALK_CHALKOPS_TD
