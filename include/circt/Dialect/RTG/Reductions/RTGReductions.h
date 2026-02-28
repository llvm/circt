//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_REDUCTIONS_RTGREDUCTIONS_H
#define CIRCT_DIALECT_RTG_REDUCTIONS_RTGREDUCTIONS_H

#include "circt/Reduce/Reduction.h"

namespace circt {
namespace rtg {

/// A dialect interface to provide reduction patterns to a reducer tool.
struct RTGReducePatternDialectInterface : public ReducePatternDialectInterface {
  using ReducePatternDialectInterface::ReducePatternDialectInterface;
  void populateReducePatterns(circt::ReducePatternSet &patterns) const override;
};

/// Register the RTG Reduction pattern dialect interface to the given registry.
void registerReducePatternDialectInterface(mlir::DialectRegistry &registry);

} // namespace rtg
} // namespace circt

#endif // CIRCT_DIALECT_RTG_REDUCTIONS_RTGREDUCTIONS_H
