//===- ESIServices.h - Code related to ESI services -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESISERVICES_H
#define CIRCT_DIALECT_ESI_ESISERVICES_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt {
namespace esi {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createESIConnectServicesPass();
} // namespace esi
} // namespace circt

#endif
