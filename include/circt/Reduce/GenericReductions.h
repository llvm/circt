//===- GenericReductions.h - Generic reduction patterns ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_REDUCE_GENERICREDUCTIONS_H
#define CIRCT_REDUCE_GENERICREDUCTIONS_H

#include "circt/Reduce/Reduction.h"
#include <optional>

namespace circt {

/// Populate reduction patterns that are not specific to certain operations or
/// dialects.
///
/// The optional `maxNumRewrites` parameter allows callers to override the
/// greedy rewrite budget used by reductions that rely on the canonicalizer
/// pass.
void populateGenericReducePatterns(
    MLIRContext *context, ReducePatternSet &patterns,
    std::optional<int64_t> maxNumRewrites = std::nullopt);

} // namespace circt

#endif // CIRCT_REDUCE_GENERICREDUCTIONS_H
