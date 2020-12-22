//===- ImplicitLocOpBuilder.h - Convenience OpBuilder -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper class to create ops with a modally set location.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_IMPLICITLOCOPBUILDER_H
#define CIRCT_SUPPORT_IMPLICITLOCOPBUILDER_H

#include "mlir/IR/Builders.h"

namespace circt {

/// ImplictLocOpBuilder maintains a 'current location', allowing use of the
/// create<> method without specifying the location.  It is otherwise the same
/// as OpBuilder.
class ImplicitLocOpBuilder : public mlir::OpBuilder {
  using OpBuilder = mlir::OpBuilder;
  using Location = mlir::Location;
  using Block = mlir::Block;
  using Value = mlir::Value;

public:
  /// Create an ImplicitLocOpBuilder using the insertion point and listener from
  /// an existing OpBuilder.
  ImplicitLocOpBuilder(Location loc, const OpBuilder &builder)
      : OpBuilder(builder), curLoc(loc) {}

  /// OpBuilder has a bunch of convenience constructors - we support them all
  /// with the additional Location.
  template <typename T>
  ImplicitLocOpBuilder(Location loc, T &&operand, Listener *listener = nullptr)
      : OpBuilder(operand, listener), curLoc(loc) {}

  ImplicitLocOpBuilder(Location loc, Block *block, Block::iterator insertPoint,
                       Listener *listener = nullptr)
      : OpBuilder(block, insertPoint, listener), curLoc(loc) {}
  ImplicitLocOpBuilder(mlir::Operation *op)
      : OpBuilder(op), curLoc(op->getLoc()) {}

  /// Accessors for the implied location.
  Location getLoc() const { return curLoc; }
  void setLoc(Location loc) { curLoc = loc; }

  // We allow clients to use the explicit-loc version of create as well.
  using OpBuilder::create;
  using OpBuilder::createOrFold;

  /// Create an operation of specific op type at the current insertion point and
  /// location.
  template <typename OpTy, typename... Args>
  OpTy create(Args &&... args) {
    return OpBuilder::create<OpTy>(curLoc, std::forward<Args>(args)...);
  }

  /// Create an operation of specific op type at the current insertion point,
  /// and immediately try to fold it. This functions populates 'results' with
  /// the results after folding the operation.
  template <typename OpTy, typename... Args>
  void createOrFold(llvm::SmallVectorImpl<Value> &results, Args &&... args) {
    OpBuilder::createOrFold<OpTy>(results, curLoc, std::forward<Args>(args)...);
  }

  /// Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  typename std::enable_if<OpTy::template hasTrait<mlir::OpTrait::OneResult>(),
                          Value>::type
  createOrFold(Args &&... args) {
    return OpBuilder::createOrFold<OpTy>(curLoc, std::forward<Args>(args)...);
  }

  /// Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  typename std::enable_if<OpTy::template hasTrait<mlir::OpTrait::ZeroResult>(),
                          OpTy>::type
  createOrFold(Args &&... args) {
    return OpBuilder::createOrFold<OpTy>(curLoc, std::forward<Args>(args)...);
  }

  /// This builder can also be used to emit diagnostics to the current location.
  mlir::InFlightDiagnostic
  emitError(const llvm::Twine &message = llvm::Twine()) {
    return mlir::emitError(curLoc, message);
  }
  mlir::InFlightDiagnostic
  emitWarning(const llvm::Twine &message = llvm::Twine()) {
    return mlir::emitWarning(curLoc, message);
  }
  mlir::InFlightDiagnostic
  emitRemark(const llvm::Twine &message = llvm::Twine()) {
    return mlir::emitRemark(curLoc, message);
  }

private:
  Location curLoc;
};

} // namespace circt

#endif // CIRCT_SUPPORT_IMPLICITLOCOPBUILDER_H
