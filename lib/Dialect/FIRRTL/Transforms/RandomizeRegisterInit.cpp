//===- RandomizeRegisterInit.cpp - Randomize register initialization ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RandomizeRegisterInit pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace circt;
using namespace firrtl;

namespace {
struct RandomizeRegisterInitPass
    : public RandomizeRegisterInitBase<RandomizeRegisterInitPass> {
  void runOnOperation() override;
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createRandomizeRegisterInitPass() {
  return std::make_unique<RandomizeRegisterInitPass>();
}

void RandomizeRegisterInitPass::runOnOperation() {
  OpBuilder builder(getOperation());

  // Collect all registers.
  SmallVector<Operation *> regs;
  getOperation().walk([&](Operation *op) {
    if (isa<RegOp, RegResetOp>(op))
      regs.push_back(op);
  });

  // Compute the width of all registers, and remember which bits are assigned
  // to each register.
  uint64_t width = 0;
  auto ui64Type = builder.getIntegerType(64, false);
  for (Operation *reg : regs) {
    auto regType = reg->getResult(0).getType().cast<FIRRTLBaseType>();
    Optional<int64_t> regWidth = getBitWidth(regType);
    assert(regWidth.has_value() && "register must have a valid FIRRTL width");

    auto start = builder.getIntegerAttr(ui64Type, width);
    auto end = builder.getIntegerAttr(ui64Type, width + regWidth.value() - 1);
    reg->setAttr("firrtl.random_init_start", start);
    reg->setAttr("firrtl.random_init_end", end);

    width += regWidth.value();
  }

  // Remember the width of the random vector in the module's attributes so
  // LowerToHW can grab it to create the appropriate random register.
  if (width > 0)
    getOperation()->setAttr("firrtl.random_init_width",
                            builder.getIntegerAttr(ui64Type, width));
}
