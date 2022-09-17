//===- Passes.h - Helpers for pipeline instrumentation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PASSES_H
#define CIRCT_SUPPORT_PASSES_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/Support/Chrono.h"

namespace circt {

// This class prints logs before and after of pass executions. This
// insrumentation assumes that passes are not parallelized for firrtl::CircuitOp
// and mlir::ModuleOp.
class VerbosePassInstrumentation final : public mlir::PassInstrumentation {
  // This stores start time points of passes.
  using TimePoint = llvm::sys::TimePoint<>;
  llvm::SmallVector<TimePoint> timePoints;
  std::string toolName;
  int level = 0;

public:
  VerbosePassInstrumentation(StringRef toolName) : toolName(toolName.str()) {}

  void runBeforePass(mlir::Pass *pass, Operation *op) override;

  void runAfterPass(mlir::Pass *pass, Operation *op) override;
};

} // namespace circt

#endif // CIRCT_SUPPORT_PASSES_H
