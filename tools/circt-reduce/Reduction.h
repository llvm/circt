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

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
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

  /// Called before the reduction is applied to a new subset of operations.
  /// Reductions may use this callback to collect information such as symbol
  /// tables about the module upfront.
  virtual void beforeReduction(mlir::ModuleOp) {}

  /// Called after the reduction has been applied to a subset of operations.
  /// Reductions may use this callback to perform post-processing of the
  /// reductions before the resulting module is tried for interestingness.
  virtual void afterReduction(mlir::ModuleOp) {}

  /// Check if the reduction can apply to a specific operation.
  virtual bool match(mlir::Operation *op) = 0;

  /// Apply the reduction to a specific operation. If the returned result
  /// indicates that the application failed, the resulting module is treated the
  /// same as if the tester marked it as uninteresting.
  virtual mlir::LogicalResult rewrite(mlir::Operation *op) = 0;

  /// Return a human-readable name for this reduction pattern.
  virtual std::string getName() const = 0;

  /// Return true if the tool should accept the transformation this reduction
  /// performs on the module even if the overall size of the output increases.
  /// This can be handy for patterns that reduce the complexity of the IR at the
  /// cost of some verbosity.
  virtual bool acceptSizeIncrease() const { return false; }

  /// Return true if the tool should not try to reapply this reduction after it
  /// has been successful. This is useful for reductions whose `match()`
  /// function keeps returning true even after the reduction has reached a
  /// fixed-point and no longer performs any change. An example of this are
  /// reductions that apply a lowering pass which always applies but may leave
  /// the input unmodified.
  ///
  /// This is mainly useful in conjunction with returning true from
  /// `acceptSizeIncrease()`. For reductions that don't accept an increase, the
  /// module size has to decrease for them to be considered useful, which
  /// prevents the tool from getting stuck at a local point where the reduction
  /// applies but produces no change in the input. However, reductions that *do*
  /// accept a size increase can get stuck in this local fixed-point as they
  /// keep applying to the same operations and the tool keeps accepting the
  /// unmodified input as an improvement.
  virtual bool isOneShot() const { return false; }
};

/// A reduction pattern that applies an `mlir::Pass`.
struct PassReduction : public Reduction {
  PassReduction(mlir::MLIRContext *context, std::unique_ptr<mlir::Pass> pass,
                bool canIncreaseSize = false, bool oneShot = false);
  bool match(mlir::Operation *op) override;
  mlir::LogicalResult rewrite(mlir::Operation *op) override;
  std::string getName() const override;
  bool acceptSizeIncrease() const override { return canIncreaseSize; }
  bool isOneShot() const override { return oneShot; }

protected:
  mlir::MLIRContext *const context;
  std::unique_ptr<mlir::PassManager> pm;
  llvm::StringRef passName;
  bool canIncreaseSize;
  bool oneShot;
};

/// Calls the function `add` with each available reduction, in the order they
/// should be applied.
void createAllReductions(
    mlir::MLIRContext *context,
    llvm::function_ref<void(std::unique_ptr<Reduction>)> add);

} // namespace circt

#endif // CIRCT_REDUCE_REDUCTION_H
