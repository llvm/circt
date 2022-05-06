//===- CalyxHelpers.cpp - Calyx helper methods -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Various helper methods for building Calyx programs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"

namespace circt {
namespace calyx {

calyx::RegisterOp createRegister(Location loc, OpBuilder &builder,
                                 ComponentOp component, size_t width,
                                 Twine prefix) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(component.getBody());
  return builder.create<RegisterOp>(loc, (prefix + "_reg").str(), width);
}

hw::ConstantOp createConstant(Location loc, OpBuilder &builder,
                              ComponentOp component, size_t width,
                              size_t value) {
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(component.getBody());
  return builder.create<hw::ConstantOp>(loc,
                                        APInt(width, value, /*unsigned=*/true));
}
} // namespace calyx
} // namespace circt
