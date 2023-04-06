//===- PassDetails.h ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_ARC_TRANSFORMS_PASSDETAILS_H
#define DIALECT_ARC_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Arc/Dialect.h"
#include "circt/Dialect/Arc/Ops.h"
#include "circt/Dialect/Arc/Passes.h"
#include "circt/Dialect/Arc/Types.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace arc {

using namespace circt;

#define GEN_PASS_CLASSES
#include "circt/Dialect/Arc/Passes.h.inc"

} // namespace arc
} // namespace circt

#endif // DIALECT_ARC_TRANSFORMS_PASSDETAILS_H
