//===- LowerToBMC.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_LOWERTOBMC
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// Convert Lower To BMC pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerToBMCPass : public circt::impl::LowerToBMCBase<LowerToBMCPass> {
  using LowerToBMCBase::LowerToBMCBase;
  void runOnOperation() override;
};
} // namespace

void LowerToBMCPass::runOnOperation() {
  Namespace names;
  // Fetch the 'hw.module' operation to model check.
  auto moduleOp = getOperation();
  auto hwModule = moduleOp.lookupSymbol<hw::HWModuleOp>(topModule);
  if (!hwModule) {
    moduleOp.emitError("hw.module named '") << topModule << "' not found";
    return signalPassFailure();
  }

  // TODO: Check whether instances contain properties to check
  if (hwModule.getOps<verif::AssertOp>().empty() &&
      hwModule.getOps<hw::InstanceOp>().empty()) {
    hwModule.emitError("no property provided to check in module");
    return signalPassFailure();
  }

  if (!sortTopologically(&hwModule.getBodyRegion().front())) {
    hwModule->emitError("could not resolve cycles in module");
    return signalPassFailure();
  }

  // Create necessary function declarations and globals
  auto *ctx = &getContext();
  OpBuilder builder(ctx);
  Location loc = moduleOp->getLoc();
  builder.setInsertionPointToEnd(moduleOp.getBody());
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = builder.getI32Type();

  // Lookup or declare printf function.
  auto printfFunc =
      LLVM::lookupOrCreateFn(moduleOp, "printf", ptrTy, i32Ty, true);
  if (failed(printfFunc)) {
    moduleOp->emitError("failed to lookup or create printf");
    return signalPassFailure();
  }

  // Replace the top-module with a function performing the BMC
  auto entryFunc = builder.create<func::FuncOp>(
      loc, topModule, builder.getFunctionType({}, {}));
  builder.createBlock(&entryFunc.getBody());

  {
    OpBuilder::InsertionGuard guard(builder);
    auto *terminator = hwModule.getBody().front().getTerminator();
    builder.setInsertionPoint(terminator);
    builder.create<verif::YieldOp>(loc, terminator->getOperands());
    terminator->erase();
  }

  // Double the bound given to the BMC op unless in rising clocks only mode, as
  // a clock cycle involves two negations
  verif::BoundedModelCheckingOp bmcOp;
  auto numRegs = hwModule->getAttrOfType<IntegerAttr>("num_regs");
  auto initialValues = hwModule->getAttrOfType<ArrayAttr>("initial_values");
  if (numRegs && initialValues) {
    for (auto value : initialValues) {
      if (!isa<IntegerAttr, UnitAttr>(value)) {
        hwModule->emitError("initial_values attribute must contain only "
                            "integer or unit attributes");
        return signalPassFailure();
      }
    }
    bmcOp = builder.create<verif::BoundedModelCheckingOp>(
        loc, risingClocksOnly ? bound : 2 * bound,
        cast<IntegerAttr>(numRegs).getValue().getZExtValue(), initialValues);
  } else {
    hwModule->emitOpError("no num_regs or initial_values attribute found - "
                          "please run externalize "
                          "registers pass first");
    return signalPassFailure();
  }

  // Check that there's only one clock input to the module
  // TODO: supporting multiple clocks isn't too hard, an interleaving of clock
  // toggles just needs to be generated
  bool hasClk = false;
  for (auto input : hwModule.getInputTypes()) {
    if (isa<seq::ClockType>(input)) {
      if (hasClk) {
        hwModule.emitError("designs with multiple clocks not yet supported");
        return signalPassFailure();
      }
      hasClk = true;
    }
    if (auto hwStruct = dyn_cast<hw::StructType>(input)) {
      for (auto field : hwStruct.getElements()) {
        if (isa<seq::ClockType>(field.type)) {
          if (hasClk) {
            hwModule.emitError(
                "designs with multiple clocks not yet supported");
            return signalPassFailure();
          }
          hasClk = true;
        }
      }
    }
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    // Initialize clock to 0 if it exists, otherwise just yield nothing
    // We initialize to 1 if we're in rising clocks only mode
    auto *initBlock = builder.createBlock(&bmcOp.getInit());
    builder.setInsertionPointToStart(initBlock);
    if (hasClk) {
      auto initVal = builder.create<hw::ConstantOp>(loc, builder.getI1Type(),
                                                    risingClocksOnly ? 1 : 0);
      auto toClk = builder.create<seq::ToClockOp>(loc, initVal);
      builder.create<verif::YieldOp>(loc, ValueRange{toClk});
    } else {
      builder.create<verif::YieldOp>(loc, ValueRange{});
    }

    // Toggle clock in loop region if it exists, otherwise just yield nothing
    auto *loopBlock = builder.createBlock(&bmcOp.getLoop());
    builder.setInsertionPointToStart(loopBlock);
    if (hasClk) {
      loopBlock->addArgument(seq::ClockType::get(ctx), loc);
      if (risingClocksOnly) {
        // In rising clocks only mode we don't need to toggle the clock
        builder.create<verif::YieldOp>(loc,
                                       ValueRange{loopBlock->getArgument(0)});
      } else {
        auto fromClk =
            builder.create<seq::FromClockOp>(loc, loopBlock->getArgument(0));
        auto cNeg1 =
            builder.create<hw::ConstantOp>(loc, builder.getI1Type(), -1);
        auto nClk = builder.create<comb::XorOp>(loc, fromClk, cNeg1);
        auto toClk = builder.create<seq::ToClockOp>(loc, nClk);
        // Only yield clock value
        builder.create<verif::YieldOp>(loc, ValueRange{toClk});
      }
    } else {
      builder.create<verif::YieldOp>(loc, ValueRange{});
    }
  }
  bmcOp.getCircuit().takeBody(hwModule.getBody());
  hwModule->erase();

  // Define global string constants to print on success/failure
  auto createUniqueStringGlobal = [&](StringRef str) -> FailureOr<Value> {
    Location loc = moduleOp.getLoc();

    OpBuilder b = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto arrayTy = LLVM::LLVMArrayType::get(b.getI8Type(), str.size() + 1);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/true, LLVM::linkage::Linkage::Private,
        "resultString",
        StringAttr::get(b.getContext(), Twine(str).concat(Twine('\00'))));
    SymbolTable symTable(moduleOp);
    if (failed(symTable.renameToUnique(global, {&symTable}))) {
      return mlir::failure();
    }

    return success(
        builder.create<LLVM::AddressOfOp>(loc, global)->getResult(0));
  };

  auto successStrAddr =
      createUniqueStringGlobal("Bound reached with no violations!\n");
  auto failureStrAddr =
      createUniqueStringGlobal("Assertion can be violated!\n");

  if (failed(successStrAddr) || failed(failureStrAddr)) {
    moduleOp->emitOpError("could not create result message strings");
    return signalPassFailure();
  }

  auto formatString = builder.create<LLVM::SelectOp>(
      loc, bmcOp.getResult(), successStrAddr.value(), failureStrAddr.value());
  builder.create<LLVM::CallOp>(loc, printfFunc.value(),
                               ValueRange{formatString});
  builder.create<func::ReturnOp>(loc);

  if (insertMainFunc) {
    builder.setInsertionPointToEnd(getOperation().getBody());
    Type i32Ty = builder.getI32Type();
    auto mainFunc = builder.create<func::FuncOp>(
        loc, "main", builder.getFunctionType({i32Ty, ptrTy}, {i32Ty}));
    builder.createBlock(&mainFunc.getBody(), {}, {i32Ty, ptrTy}, {loc, loc});
    builder.create<func::CallOp>(loc, entryFunc, ValueRange{});
    // TODO: don't use LLVM here
    Value constZero = builder.create<LLVM::ConstantOp>(loc, i32Ty, 0);
    builder.create<func::ReturnOp>(loc, constZero);
  }
}
