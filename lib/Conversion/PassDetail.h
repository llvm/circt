//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_PASSDETAIL_H
#define CONVERSION_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace circt {

namespace firrtl {
class FIRRTLDialect;
class FModuleOp;
} // namespace firrtl

namespace rtl {
class RTLDialect;
class RTLModuleOp;
} // namespace rtl

namespace sv {
class SVDialect;
} // namespace sv

// Generate the classes which represent the passes
#define GEN_PASS_CLASSES
#include "circt/Conversion/Passes.h.inc"

} // namespace circt

#endif // CONVERSION_PASSDETAIL_H
