//===- StaticLogicPasses.h - StaticLogic pass definitions --------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the StaticLogic
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_STATICLOGIC_PASSES_H_
#define CIRCT_STATICLOGIC_PASSES_H_

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt {
namespace staticlogic {

std::unique_ptr<mlir::Pass> createSchedulePipelinePass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/StaticLogic/StaticLogicPasses.h.inc"

} // namespace staticlogic
} // namespace circt

#endif // CIRCT_STATICLOGIC_PASSES_H_
