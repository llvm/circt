//===- RegisterAllEmitters.h - Register all emitters to ExportSystemC -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This registers the all the emitters of various dialects to the
// ExportSystemC pass.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TARGET_EXPORTSYSTEMC_REGISTERALLEMITTERS_H
#define CIRCT_TARGET_EXPORTSYSTEMC_REGISTERALLEMITTERS_H

#include "Patterns/SystemCEmissionPatterns.h"

namespace circt {
namespace ExportSystemC {

/// Collects the operation emission patterns of all supported dialects.
inline void registerAllOpEmitters(OpEmissionPatternSet &patterns,
                                  mlir::MLIRContext *context) {
  populateSystemCOpEmitters(patterns, context);
}

/// Collects the type emission patterns of all supported dialects.
inline void registerAllTypeEmitters(TypeEmissionPatternSet &patterns) {
  populateSystemCTypeEmitters(patterns);
}

} // namespace ExportSystemC
} // namespace circt

#endif // CIRCT_TARGET_EXPORTSYSTEMC_REGISTERALLEMITTERS_H
