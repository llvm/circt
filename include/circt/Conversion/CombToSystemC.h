//===- CombToSystemC.h - Comb to SystemC pass entry point -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose the CombToSystemC pass
// constructors.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_COMBTOSYSTEMC_COMBTOSYSTEMC_H_
#define CIRCT_CONVERSION_COMBTOSYSTEMC_COMBTOSYSTEMC_H_

#include <memory>

namespace mlir {
template <typename T>
class OperationPass;
class ModuleOp;
} // namespace mlir

namespace circt {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertCombToSystemCPass();
} // namespace circt

#endif // CIRCT_CONVERSION_COMBTOSYSTEMC_COMBTOSYSTEMC_H_
