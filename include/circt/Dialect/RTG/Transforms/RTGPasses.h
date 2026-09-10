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

#ifndef CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSES_H
#define CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt {
namespace rtg {

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
#undef GEN_PASS_REGISTRATION

} // namespace rtg
} // namespace circt

#endif // CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSES_H
