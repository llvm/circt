//===- Utilities.h - Library of scheduling utilities ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a library of scheduling utilities.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_UTILITIES_H
#define CIRCT_SCHEDULING_UTILITIES_H

#include "circt/Scheduling/Problems.h"

#include <functional>

namespace circt {
namespace scheduling {

using HandleOpFn = std::function<LogicalResult(Operation *)>;
/// Visit \p prob's operations in topological order, using an internal worklist.
///
/// \p fun is expected to report success if the given operation was handled
/// successfully, and failure if an unhandled predecessor was detected.
///
/// Fails if the dependence graph contains cycles.
LogicalResult handleOperationsInTopologicalOrder(Problem &prob, HandleOpFn fun);

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_UTILITIES_H
