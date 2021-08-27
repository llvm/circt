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

include "circt/Dialect/LLHD/IR/LLHD.td"
include "mlir/Dialect/StandardOps/IR/Ops.td"

def IndexAttrFromSignlessInteger : NativeCodeCall<
  "$_builder.getIndexAttr($0.cast<IntegerAttr>().getValue().getZExtValue())">;

//===----------------------------------------------------------------------===//
// DynExtractSliceOp
//===----------------------------------------------------------------------===//

def DynExtractSliceWithLLHDConstOpStart : Pat<
  (LLHD_DynExtractSliceOp $target, (LLHD_ConstOp $start)),
  (LLHD_ExtractSliceOp $target, (IndexAttrFromSignlessInteger $start))>;

def DynExtractSliceWithConstantOpStart : Pat<
  (LLHD_DynExtractSliceOp $target, (ConstantOp $start)),
  (LLHD_ExtractSliceOp $target, (IndexAttrFromSignlessInteger $start))>;

//===----------------------------------------------------------------------===//
// DynExtractElementOp
//===----------------------------------------------------------------------===//

def DynExtractElementWithLLHDConstOpIndex : Pat<
  (LLHD_DynExtractElementOp $target, (LLHD_ConstOp $index)),
  (LLHD_ExtractElementOp $target, (IndexAttrFromSignlessInteger $index))>;

def DynExtractElementWithConstantOpIndex : Pat<
  (LLHD_DynExtractElementOp $target, (ConstantOp $index)),
  (LLHD_ExtractElementOp $target, (IndexAttrFromSignlessInteger $index))>;

#endif // CIRCT_DIALECT_LLHD_IR_CANONICALIZATION_TD
