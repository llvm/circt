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

/// Generate the code for registering passes.
#define GEN_PASS_DECL
#define GEN_PASS_DECL_ELABORATION
#define GEN_PASS_DECL_RTGTOELF
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"

} // namespace rtg
} // namespace circt

#endif // CIRCT_DIALECT_RTG_RTGPASSES_H
