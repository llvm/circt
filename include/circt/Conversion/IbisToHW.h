//===- IbisToHW.h - Ibis to HW conversion pass ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the Ibis dialect to the
// HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_IBISTOHW_H
#define CIRCT_CONVERSION_IBISTOHW_H

#include <memory>

namespace mlir {
template <typename T>
class OperationPass;
class ModuleOp;
} // namespace mlir

namespace circt {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertIbisToHWPass();

} // namespace circt

#endif // CIRCT_CONVERSION_IBISTOHW_H
