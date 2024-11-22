//===- RTGPasses.h - RTG pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_RTGPASSES_H
#define CIRCT_DIALECT_RTG_RTGPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <optional>

namespace circt {
namespace rtg {

/// Options for the ElaborationPass
struct ElaborationOptions {
  /// The seed for any RNG constructs used in the pass. If `std::nullopt` is
  /// passed, no seed is used and thus the IR after each invocation of this pass
  /// will be different non-deterministically. However, if a nubmer is provided,
  /// the pass is guaranteed to produce the same IR every time.
  std::optional<unsigned> seed;

  /// When in debug mode the pass queries the values that would otherwise be
  /// provided by the RNG from an attribute attached to the operation called
  /// 'rtg.elaboration'.
  bool debugMode = false;
};

std::unique_ptr<mlir::Pass> createElaborationPass();
std::unique_ptr<mlir::Pass>
createElaborationPass(const ElaborationOptions &options);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
#undef GEN_PASS_REGISTRATION

} // namespace rtg
} // namespace circt

#endif // CIRCT_DIALECT_RTG_RTGPASSES_H
