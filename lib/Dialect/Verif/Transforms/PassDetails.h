//===- PassDetails.h - Verif pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Stuff shared between the different Verif passes.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef DIALECT_VERIF_TRANSFORMS_PASSDETAILS_H
#define DIALECT_VERIF_TRANSFORMS_PASSDETAILS_H

#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace verif {

#define GEN_PASS_CLASSES
#include "circt/Dialect/Verif/Passes.h.inc"

} // namespace verif
} // namespace circt

#endif // DIALECT_FSM_TRANSFORMS_PASSDETAILS_H
