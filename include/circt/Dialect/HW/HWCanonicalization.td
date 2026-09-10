//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWCANONICALIZATION_TD
#define CIRCT_DIALECT_HW_HWCANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "circt/Dialect/HW/HWOps.td"

def NotEqual : Constraint<CPred<"$0 != $1">>;

//===----------------------------------------------------------------------===//
// ArrayInjectOp
//===----------------------------------------------------------------------===//

def ArrayInjectToSameIndex : Pat<
  (ArrayInjectOp:$op (ArrayInjectOp $array, $index, $_), $index, $element),
  (ArrayInjectOp $array, $index, $element),
  [(NotEqual $op, $array)]
>;

#endif // CIRCT_DIALECT_HW_HWCANONICALIZATION_TD
