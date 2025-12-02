//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/Reductions/RTGReductions.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"

using namespace mlir;
using namespace circt;
using namespace rtg;

//===----------------------------------------------------------------------===//
// Reduction patterns
//===----------------------------------------------------------------------===//

/// Replace virtual registers with a constant register.
struct VirtualRegisterConstantifier : public OpReduction<VirtualRegisterOp> {
  LogicalResult rewrite(VirtualRegisterOp op) override {
    if (op.getAllowedRegs().getAllowedRegs().empty())
      return failure();

    OpBuilder builder(op);
    auto constReg = ConstantOp::create(builder, op.getLoc(),
                                       op.getAllowedRegs().getAllowedRegs()[0]);
    op.getResult().replaceAllUsesWith(constReg);
    op.erase();
    return success();
  }

  std::string getName() const override {
    return "rtg-virtual-register-constantifier";
  }
};

//===----------------------------------------------------------------------===//
// Reduction Registration
//===----------------------------------------------------------------------===//

void RTGReducePatternDialectInterface::populateReducePatterns(
    circt::ReducePatternSet &patterns) const {
  // Gather a list of reduction patterns that we should try. Ideally these are
  // assigned reasonable benefit indicators (higher benefit patterns are
  // prioritized). For example, things that can knock out entire modules while
  // being cheap should be tried first (and thus have higher benefit), before
  // trying to tweak operands of individual arithmetic ops.
  patterns.add<VirtualRegisterConstantifier, 2>();
}

void rtg::registerReducePatternDialectInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, RTGDialect *dialect) {
    dialect->addInterfaces<RTGReducePatternDialectInterface>();
  });
}
