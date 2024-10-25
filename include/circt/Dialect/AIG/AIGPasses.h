//===- AIGPasses.h - AIG dialect passes -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AIG_AIGPASSES_H
#define CIRCT_DIALECT_AIG_AIGPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <optional>

namespace mlir {
class Pass;
} // namespace mlir

#include "circt/Dialect/AIG/AIGPassesEnums.h.inc"

namespace circt {
namespace aig {

#define GEN_PASS_DECL
#include "circt/Dialect/AIG/AIGPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "circt/Dialect/AIG/AIGPasses.h.inc"

} // namespace aig
} // namespace circt

#endif // CIRCT_DIALECT_AIG_AIGPASSES_H
