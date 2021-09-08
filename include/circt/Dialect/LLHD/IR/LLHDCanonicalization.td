//===- LLHDCanonicalization.td - LLHD canonicalization -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines LLHD canonicalization patterns.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_CANONICALIZATION_TD
#define CIRCT_DIALECT_LLHD_IR_CANONICALIZATION_TD

include "circt/Dialect/HW/HW.td"
include "circt/Dialect/LLHD/IR/LLHD.td"

def IndexAttrFromSignlessInteger : NativeCodeCall<
  "$_builder.getIndexAttr($0.cast<IntegerAttr>().getValue().getZExtValue())">;

//===----------------------------------------------------------------------===//
// DynExtractSliceOp
//===----------------------------------------------------------------------===//

def DynExtractSliceWithConstantOpStart : Pat<
  (LLHD_DynExtractSliceOp $target, (ConstantOp $start)),
  (LLHD_ExtractSliceOp $target, (IndexAttrFromSignlessInteger $start))>;

//===----------------------------------------------------------------------===//
// DynExtractElementOp
//===----------------------------------------------------------------------===//

def DynExtractElementWithConstantOpIndex : Pat<
  (LLHD_DynExtractElementOp $target, (ConstantOp $index)),
  (LLHD_ExtractElementOp $target, (IndexAttrFromSignlessInteger $index))>;

#endif // CIRCT_DIALECT_LLHD_IR_CANONICALIZATION_TD
