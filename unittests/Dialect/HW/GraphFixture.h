//===- GraphFixutre.h - A fixture for instance graph unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef UNITTEST_DIALECT_FIRRTL_HW_GRAPHFIXTURE_H
#define UNITTEST_DIALECT_FIRRTL_HW_GRAPHFIXTURE_H

#include "circt/Dialect/HW/HWOps.h"

namespace fixtures {

mlir::ModuleOp createModule(mlir::MLIRContext *context);

} // end namespace fixtures

#endif // UNITTEST_DIALECT_FIRRTL_HW_GRAPHFIXTURE_H