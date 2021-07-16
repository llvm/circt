//===- PassDetails.h - Calyx pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the stuff shared between the different Calyx passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the
// header guard on some systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_CALYX_TRANSFORMS_PASSDETAILS_H
#define DIALECT_CALYX_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace calyx {

#define GEN_PASS_CLASSES
#include "circt/Dialect/Calyx/CalyxPasses.h.inc"

// TODO: Update documentation
// ^
/// Adds the group's "go" signal to the guards of assignments within `group`,
/// with the exception of the "done" terminator. If the assignment already
/// has a guard, then the bitwise 'and' is taken of the current guard and the
/// "go" signal. For example:
///    ```mlir
///      %go = calyx.group_go %undef : i1
///
///      // Case 1: No Guard
///      %in = %out : i8
///      =>
///      %in = %out, %go ? : i8
///
///      // Case 2: Guard
///      %in = %out, %guard ? : i8
///      =>
///      %and = comb.and %guard, %go : i1
///      %in = %out, %and ? : i8
///    ```
template <typename Op>
static void updateGroupAssignmentGuards(OpBuilder &builder, GroupOp &group,
                                        Op &op) {
  group.walk([&](AssignOp assign) {
    if (assign.guard())
      // Take the bitwise & of the current guard and the group's go signal.
      assign->setOperand(
          2, builder.create<comb::AndOp>(group.getLoc(), assign.guard(), op));
    else
      assign->insertOperands(2, {op});
  });
}

} // namespace calyx
} // namespace circt

#endif // DIALECT_CALYX_TRANSFORMS_PASSDETAILS_H
