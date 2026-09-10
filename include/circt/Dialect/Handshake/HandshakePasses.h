//===- HandshakePasses.h - Handshake pass entry points ----------*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H
#define CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H

#include "circt/Support/LLVM.h"
#include <map>
#include <memory>
#include <optional>
#include <set>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace handshake {
class FuncOp;

#define GEN_PASS_DECL
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeDotPrintPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createHandshakeOpCountPass();
std::unique_ptr<mlir::Pass> createHandshakeMaterializeForksSinksPass();
std::unique_ptr<mlir::Pass> createHandshakeDematerializeForksSinksPass();
std::unique_ptr<mlir::Pass> createHandshakeRemoveBuffersPass();
std::unique_ptr<mlir::Pass> createHandshakeAddIDsPass();
std::unique_ptr<mlir::Pass>
createHandshakeLowerExtmemToHWPass(std::optional<bool> createESIWrapper = {});
std::unique_ptr<mlir::Pass> createHandshakeLegalizeMemrefsPass();
std::unique_ptr<mlir::OperationPass<handshake::FuncOp>>
createHandshakeInsertBuffersPass(const std::string &strategy = "all",
                                 unsigned bufferSize = 2);
std::unique_ptr<mlir::Pass> createHandshakeLockFunctionsPass();
std::unique_ptr<mlir::Pass> createHandshakeSplitMergesPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"

} // namespace handshake
} // namespace circt

#endif // CIRCT_DIALECT_HANDSHAKE_HANDSHAKEPASSES_H
