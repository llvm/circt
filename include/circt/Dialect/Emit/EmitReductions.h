//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_EMIT_EMITREDUCTIONS_H
#define CIRCT_DIALECT_EMIT_EMITREDUCTIONS_H

#include "circt/Reduce/Reduction.h"

namespace circt {
namespace emit {

/// Register the Emit Reduction pattern dialect interface to the given registry.
void registerReducePatternDialectInterface(mlir::DialectRegistry &registry);

} // namespace emit
} // namespace circt

#endif // CIRCT_DIALECT_EMIT_EMITREDUCTIONS_H
