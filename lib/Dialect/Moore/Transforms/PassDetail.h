//===- PassDetail.h - Morre pass class details ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different Moore passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_MOORE_TRANSFORMS_PASSDETAIL_H
#define DIALECT_MOORE_TRANSFORMS_PASSDETAIL_H

#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace moore {

#define GEN_PASS_CLASSES
#include "circt/Dialect/Moore/MoorePasses.h.inc"

} // namespace moore
} // namespace circt

#endif // DIALECT_MOORE_TRANSFORMS_PASSDETAIL_H
