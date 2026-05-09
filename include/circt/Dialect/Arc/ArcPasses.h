//===- ArcPasses.h - Arc dialect passes -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCPASSES_H
#define CIRCT_DIALECT_ARC_ARCPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <optional>

#include "circt/Dialect/HW/HWOps.h"

namespace mlir {
class Pass;
} // namespace mlir

#include "circt/Dialect/Arc/ArcPassesEnums.h.inc"

namespace circt {
namespace arc {

#define GEN_PASS_DECL
#include "circt/Dialect/Arc/ArcPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Arc/ArcPasses.h.inc"

} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_ARCPASSES_H
