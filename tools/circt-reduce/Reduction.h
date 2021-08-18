//===- Reduction.h - Reductions for circt-reduce --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines abstract reduction patterns for the 'circt-reduce' tool.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_REDUCE_REDUCTION_H
#define CIRCT_REDUCE_REDUCTION_H

#include <memory>
#include <string>

namespace llvm {
class StringRef;
template <typename T>
class function_ref;
} // namespace llvm

namespace mlir {
struct LogicalResult;
class MLIRContext;
class Operation;
class Pass;
class PassManager;
} // namespace mlir

namespace circt {

/// An abstract reduction pattern.
struct Reduction {
  virtual ~Reduction();

  /// Check if the reduction can apply to a specific operation.
  virtual bool match(mlir::Operation *op) const = 0;

  /// Apply the reduction to a specific operation. If the returned result
  /// indicates that the application failed, the resulting module is treated the
  /// same as if the tester marked it as uninteresting.
  virtual mlir::LogicalResult rewrite(mlir::Operation *op) const = 0;

  /// Return a human-readable name for this reduction pattern.
  virtual std::string getName() const = 0;

  /// Return true if the tool should accept the transformation this reduction
  /// performs on the module even if the overall size of the output increases.
  /// This can be handy for patterns that reduce the complexity of the IR at the
  /// cost of some verbosity.
  virtual bool acceptSizeIncrease() const { return false; }
};

/// A reduction pattern that applies an `mlir::Pass`.
struct PassReduction : public Reduction {
  PassReduction(mlir::MLIRContext *context, std::unique_ptr<mlir::Pass> pass,
                bool canIncreaseSize = false);
  bool match(mlir::Operation *op) const override;
  mlir::LogicalResult rewrite(mlir::Operation *op) const override;
  std::string getName() const override;
  bool acceptSizeIncrease() const override { return canIncreaseSize; }

protected:
  mlir::MLIRContext *const context;
  std::unique_ptr<mlir::PassManager> pm;
  llvm::StringRef passName;
  bool canIncreaseSize;
};

/// Calls the function `add` with each available reduction, in the order they
/// should be applied.
void createAllReductions(
    mlir::MLIRContext *context,
    llvm::function_ref<void(std::unique_ptr<Reduction>)> add);

} // namespace circt

#endif // CIRCT_REDUCE_REDUCTION_H
