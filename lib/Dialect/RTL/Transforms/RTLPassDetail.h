//===- PassDetail.h - RTL pass class details --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_RTL_TRANSFORMS_RTLPASSDETAIL_H
#define DIALECT_RTL_TRANSFORMS_RTLPASSDETAIL_H

#include "circt/Dialect/RTL/RTLOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace rtl {

#define GEN_PASS_CLASSES
#include "circt/Dialect/RTL/RTLPasses.h.inc"

} // namespace rtl
} // namespace circt

#endif // DIALECT_RTL_TRANSFORMS_RTLPASSDETAIL_H
