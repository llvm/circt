//===- OMCanonicalization.td - OM canonicalization patterns ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMCANONICALIZATION_TD
#define CIRCT_DIALECT_OM_OMCANONICALIZATION_TD

include "mlir/IR/PatternBase.td"
include "circt/Dialect/OM/OMOps.td"

def NotEqual : Constraint<CPred<"$0 != $1">>;

// object.eq(x, x) -> true, even for unknown x.
// Increase the benefit to ensure that this takes high precedence.
def ObjectEqSame : Pat<
  (ObjectEqOp $ref, $ref),
  (ConstantOp ConstBoolAttrTrue),
  [], [], (addBenefit 1)
>;

// object.eq(a, b) -> false, if a and b are distinct elaborated objects.
def ObjectEqDistinct : Pat<
  (ObjectEqOp (ElaboratedObjectOp:$lhs $_, $_),
              (ElaboratedObjectOp:$rhs $_, $_)),
  (ConstantOp ConstBoolAttrFalse),
  [(NotEqual $lhs, $rhs)]
>;

// object.eq(any_cast(x), y) -> object.eq(x, y).
// Require object-compatible x and a non-self-referential cast.
def ObjectEqAnyCastLHS : Pat<
  (ObjectEqOp (AnyCastOp:$cast AnyTypeOf<[ClassType, OMAnyType]>:$lhs), $rhs),
  (ObjectEqOp $lhs, $rhs),
  [(NotEqual $lhs, $cast)]
>;

// object.eq(x, any_cast(y)) -> object.eq(x, y).
// Require object-compatible y and a non-self-referential cast.
def ObjectEqAnyCastRHS : Pat<
  (ObjectEqOp $lhs, (AnyCastOp:$cast AnyTypeOf<[ClassType, OMAnyType]>:$rhs)),
  (ObjectEqOp $lhs, $rhs),
  [(NotEqual $rhs, $cast)]
>;

#endif // CIRCT_DIALECT_OM_OMCANONICALIZATION_TD
