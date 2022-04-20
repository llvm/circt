//===- BackedgeBuilder.h - Support for building backedges -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Backedges are operations/values which have to exist as operands before
// they are produced in a result. Since it isn't clear how to build backedges
// in MLIR, these helper classes set up a canonical way to do so.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_BACKEDGEBUILDER_H
#define CIRCT_SUPPORT_BACKEDGEBUILDER_H

#include "circt/Dialect/HW/HWOps.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class OpBuilder;
class PatternRewriter;
class Operation;
} // namespace mlir

namespace circt {

/// `Backedge` is a wrapper class around a `Value`. When assigned another
/// `Value`, it replaces all uses of itself with the new `Value` then become a
/// wrapper around the new `Value`.
class Backedge {
public:
  Backedge(mlir::Operation *op) : value(op->getResult(0)){};

  operator mlir::Value() { return value; }
  void setValue(mlir::Value newValue) {
    assert(value.getType() == newValue.getType());
    assert(!set && "backedge already set to a value!");
    value.replaceAllUsesWith(newValue);
    set = true;
  }

private:
  mlir::Value value;
  bool set = false;
};

/// Instantiate one of these and use it to build typed backedges. Backedges
/// which get used as operands must be assigned to with the actual value before
/// this class is destructed, usually at the end of a scope. It will check that
/// invariant then erase all the backedge ops during destruction.
///
/// Example use:
/// ```
///   circt::BackedgeBuilder back(rewriter, loc);
///   circt::Backedge ready = back.get(rewriter.getI1Type());
///   // Use `ready` as a `Value`.
///   auto addOp = rewriter.create<addOp>(loc, ready);
///   // When the actual value is available,
///   ready.set(anotherOp.getResult(0));
/// ```
///
/// This template is to enable the use of an operation in any dialect which
/// looks like a backedge op. Backedge ops have no arguments and return any
/// type. There is a standard one in the 'HW' dialect, but presumably not every
/// piece of code wants to load the 'HW' dialect. For the common case, an alias
/// is provided below.
template <typename BackedgeOp>
class BackedgeBuilderImpl {
public:
  /// To build a backedge op and manipulate it, we need a `PatternRewriter` and
  /// a `Location`. Store them during construct of this instance and use them
  /// when building.
  BackedgeBuilderImpl(mlir::OpBuilder &builder, mlir::Location loc)
      : builder(builder), rewriter(nullptr), loc(loc) {}
  BackedgeBuilderImpl(mlir::PatternRewriter &rewriter, mlir::Location loc)
      : builder(rewriter), rewriter(&rewriter), loc(loc) {}
  ~BackedgeBuilderImpl() {
    for (Operation *op : edges) {
      assert(op->use_empty() && "Backedge still in use");
      if (rewriter)
        rewriter->eraseOp(op);
      else
        op->erase();
    }
  }
  /// Create a typed backedge.
  Backedge get(mlir::Type resultType) {
    auto be = builder.create<BackedgeOp>(loc, resultType);
    edges.push_back(be);
    return Backedge(be.getOperation());
  }

private:
  mlir::OpBuilder &builder;
  mlir::PatternRewriter *rewriter;
  mlir::Location loc;
  llvm::SmallVector<BackedgeOp, 16> edges;
};

using BackedgeBuilder = BackedgeBuilderImpl<hw::BackedgeOp>;

} // namespace circt

#endif // CIRCT_SUPPORT_BACKEDGEBUILDER_H
