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
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Parallel.h"

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

/// Create attributes indicating the required size of random initialization
/// values for each register in the module, and mark which range of these values
/// each register should consume. The goal is for registers to always read the
/// same random bits for the same seed, regardless of optimizations that might
/// remove registers.
static void createRandomizationAttributes(FModuleOp mod) {
  OpBuilder builder(mod);

  // Set a maximum width for any single register, based on Verilog 1364-2005.
  constexpr uint64_t maxRegisterWidth = 65536;

  // Walk all registers.
  uint64_t currentRegister = 0;
  SmallDenseMap<uint64_t, uint64_t> widths;
  auto ui64Type = builder.getIntegerType(64, false);
  mod.walk([&](Operation *op) {
    if (!isa<RegOp, RegResetOp>(op))
      return;

    // Compute the width of all registers, and remember which bits are assigned
    // to each register.
    auto regType = op->getResult(0).getType().cast<FIRRTLBaseType>();
    Optional<int64_t> regWidth = getBitWidth(regType);
    assert(regWidth.has_value() && "register must have a valid FIRRTL width");

    auto currentWidth = widths[currentRegister];

    // If the current register width is non-zero, and the current width plus the
    // size of the next register exceeds the limit, spill over to a new
    // register. Note that if a single register exceeds the limit, this will
    // still happily emit a single random initialization register that also
    // exceeds the limit.
    if (currentWidth > 0 &&
        currentWidth + regWidth.value() > maxRegisterWidth) {
      ++currentRegister;
      currentWidth = widths[currentRegister];
    }

    auto start = builder.getIntegerAttr(ui64Type, currentWidth);
    auto end =
        builder.getIntegerAttr(ui64Type, currentWidth + regWidth.value() - 1);
    op->setAttr("firrtl.random_init_register",
                builder.getStringAttr(Twine(currentRegister)));
    op->setAttr("firrtl.random_init_start", start);
    op->setAttr("firrtl.random_init_end", end);

    widths[currentRegister] += regWidth.value();
  });

  // Remember the width of the random vector in the module's attributes so
  // LowerSeqToSV can grab it to create the appropriate random register.
  if (!widths.empty()) {
    SmallVector<NamedAttribute> widthDictionary;
    for (auto [registerIndex, registerWidth] : widths)
      widthDictionary.emplace_back(
          builder.getStringAttr(Twine(registerIndex)),
          builder.getIntegerAttr(ui64Type, registerWidth));
    mod->setAttr("firrtl.random_init_width",
                 builder.getDictionaryAttr(widthDictionary));
  }
}

void RandomizeRegisterInitPass::runOnOperation() {
  createRandomizationAttributes(getOperation());
}
