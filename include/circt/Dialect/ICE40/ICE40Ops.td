//===- ICE40Ops.td - ICE40 ops ============-----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines the ICE40 ops, matching the technology library.  Documentation
// is here: https://www.latticesemi.com/-/media/LatticeSemi/Documents/TechnicalBriefs/FPGA-TN-02026-3-2-iCE40-Technology-Library.ashx?document_id=52206
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ICE40_ICE40OPS_TD
#define CIRCT_DIALECT_ICE40_ICE40OPS_TD

include "circt/Dialect/ICE40/ICE40Dialect.td"

def I1Property : IntProperty<"bool">;

def SBLut4Op : ICE40Op<"sb_lut4", []> {
  let summary = "A basic lut";
  let description = [{
    The basic look up table.  f(i0, i1, i2, i3) = lut_init[concat(i3,i2,i1,i0)].
    }];

  let arguments = (ins I1:$i0, I1:$i1, I1:$i2, I1:$i3, ArrayProperty<I1Property>:$lut_init);
  let results = (outs I1:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $i0 `,` $i1 `,` $i2 `,` $i3 `,` $lut_init attr-dict
  }];
}

def SBCarryOp : ICE40Op<"sb_carry", []> {
  let summary = "Dedicated carry Lookahead logic";
  let description = [{
    The basic carry lookahead logic.  f(i0, i1, ci) = 1 iff popcount(concat(i0,i1,ci)) >= 2.
    There are constraints on the use, as i0 and i1 have to be the i0 and i1 input of a single lut.
    This op doesn't check this.  There is a combined op that handles that.
    }];

  let arguments = (ins I1:$i0, I1:$i1, I1:$ci);
  let results = (outs I1:$result);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $i0 `,` $i1 `,` $ci attr-dict
  }];
}

def SBLut4CarryOp : ICE40Op<"sb_lut4_carry", []> {
  let summary = "LUT which also utilizes the carry logic";
  let description = [{
      carry_out(i0, i1, ci) = 1 iff popcount(concat(i0,i1,ci)) >= 2.
      result(i0, i1, i2, i3) = lut_init[concat(i3,i2,i1,i0)].
    There are constraints on the use, as i0 and i1 have to be the i0 and i1 input of a single lut.
    This op doesn't check this.  There is a combined op that handles that.
    }];

  let arguments = (ins I1:$i0, I1:$i1, I1:$i2, I1:$i3, I1:$ci, ArrayProperty<I1Property>:$lut_init);
  let results = (outs I1:$result, I1:$carry_out);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $i0 `,` $i1 `,` $i2 `,` $i3 `,` $ci `,` $lut_init attr-dict
  }];
}

#endif // CIRCT_DIALECT_ICE40_ICE40OPS_TD
