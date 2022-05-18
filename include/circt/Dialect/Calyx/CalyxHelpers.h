//===- CalyxHelpers.h - Calyx helper methods --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines various helper methods for building Calyx programs.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXHELPERS_H
#define CIRCT_DIALECT_CALYX_CALYXHELPERS_H

#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"

#include <memory>

namespace circt {
namespace calyx {

/// Creates a RegisterOp, with input and output port bit widths defined by
/// `width`.
calyx::RegisterOp createRegister(Location loc, OpBuilder &builder,
                                 ComponentOp component, size_t width,
                                 Twine prefix);

/// A helper function to create constants in the HW dialect.
hw::ConstantOp createConstant(Location loc, OpBuilder &builder,
                              ComponentOp component, size_t width,
                              size_t value);
} // namespace calyx
} // namespace circt

#endif // CIRCT_DIALECT_CALYX_CALYXHELPERS_H
