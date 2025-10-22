//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitReductions.h"
#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Reduce/ReductionUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "emit-reductions"

using namespace circt;
using namespace emit;

//===----------------------------------------------------------------------===//
// Reduction patterns
//===----------------------------------------------------------------------===//

namespace {

/// A reduction pattern that erases emit dialect operations.
struct EmitOpEraser : public Reduction {
  void beforeReduction(mlir::ModuleOp op) override {
    innerSymUses = reduce::InnerSymbolUses(op);
  }

  uint64_t match(Operation *op) override {
    if (!isa_and_nonnull<emit::EmitDialect>(op->getDialect()))
      return 0;
    if (innerSymUses.hasRef(op))
      return 0;
    return 1;
  }

  LogicalResult rewrite(Operation *op) override {
    op->erase();
    return success();
  }

  std::string getName() const override { return "emit-op-eraser"; }
  bool acceptSizeIncrease() const override { return true; }

  reduce::InnerSymbolUses innerSymUses;
};

} // namespace

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

namespace {
/// A dialect interface to provide reduction patterns to a reducer tool.
struct EmitReducePatternDialectInterface
    : public ReducePatternDialectInterface {
  using ReducePatternDialectInterface::ReducePatternDialectInterface;
  void populateReducePatterns(ReducePatternSet &patterns) const override {
    patterns.add<EmitOpEraser, 1000>();
  }
};
} // namespace

void emit::registerReducePatternDialectInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, EmitDialect *dialect) {
    dialect->addInterfaces<EmitReducePatternDialectInterface>();
  });
}
