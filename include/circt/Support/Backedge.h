//===- Backedge.h - Support classes for building backedges ------*- C++ -*-===//
//
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_BACKEDGE_H
#define CIRCT_SUPPORT_BACKEDGE_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace circt {

// class Backedge : public mlir::Op<Backedge, mlir::OpTrait::OneResult> {
//   friend class BackedgeBuilder;

// public:
//   using Op::Op;
//   static llvm::StringRef getOperationName();
//   static void build(mlir::OpBuilder &b, mlir::OperationState &state,
//                     mlir::Type resultType);
// };

class BackedgeBuilder;

class Backedge {
  friend class BackedgeBuilder;
  Backedge(BackedgeBuilder &parent, mlir::Operation *op);

public:
  operator mlir::Value();

private:
  BackedgeBuilder &parent;
  mlir::Operation *op;
};

class BackedgeBuilder {
  friend class Backedge;

public:
  BackedgeBuilder(mlir::PatternRewriter &innerBuilder, mlir::Location loc);
  ~BackedgeBuilder();
  Backedge operator()(mlir::Type resultType);

private:
  mlir::PatternRewriter &rewriter;
  mlir::Location loc;
  llvm::SmallVector<mlir::Operation *, 16> edges;
};

} // namespace circt

#endif // CIRCT_SUPPORT_BACKEDGE_H
