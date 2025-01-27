//===- EmitPasses.h - Emit dialect passes ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_EMIT_EMITPASSES_H
#define CIRCT_DIALECT_EMIT_EMITPASSES_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
namespace emit {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Emit/EmitPasses.h.inc"

} // namespace emit
} // namespace circt

#endif // CIRCT_DIALECT_EMIT_EMITPASSES_H
