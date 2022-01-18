//===- PassDetails.h - GAA pass class details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different GAA passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_GAA_TRANSFORMS_PASSDETAILS_H
#define DIALECT_GAA_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/GAA/GAAOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {

namespace hw {
class HWDialect;
}

namespace gaa {

#define GEN_PASS_CLASSES
#include "circt/Dialect/GAA/GAAPasses.h.inc"

} // namespace gaa
} // namespace circt

#endif // DIALECT_GAA_TRANSFORMS_PASSDETAILS_H
