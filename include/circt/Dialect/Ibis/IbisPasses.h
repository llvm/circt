//===- Passes.h - Ibis pass entry points -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_IBISPASSES_H
#define CIRCT_DIALECT_IBIS_IBISPASSES_H

#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>
#include <optional>

namespace circt {
namespace ibis {

#define GEN_PASS_DECL_IBISTUNNELING
#include "circt/Dialect/Ibis/IbisPasses.h.inc"

std::unique_ptr<Pass> createCallPrepPass();
std::unique_ptr<Pass> createContainerizePass();
std::unique_ptr<Pass> createTunnelingPass(const IbisTunnelingOptions & = {});
std::unique_ptr<Pass> createPortrefLoweringPass();
std::unique_ptr<Pass> createCleanSelfdriversPass();
std::unique_ptr<Pass> createContainersToHWPass();
std::unique_ptr<Pass> createArgifyBlocksPass();
std::unique_ptr<Pass> createReblockPass();
std::unique_ptr<Pass> createInlineSBlocksPass();
std::unique_ptr<Pass> createConvertCFToHandshakePass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Ibis/IbisPasses.h.inc"

} // namespace ibis
} // namespace circt

#endif // CIRCT_DIALECT_IBIS_IBISPASSES_H
