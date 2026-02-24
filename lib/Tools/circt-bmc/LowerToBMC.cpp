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

  if (!sortTopologically(&hwModule.getBodyRegion().front())) {
    hwModule->emitError("could not resolve cycles in module");
    return signalPassFailure();
  }

  if (bound < ignoreAssertionsUntil) {
    hwModule->emitError(
        "number of ignored cycles must be less than or equal to bound");
    return signalPassFailure();
  }

  // Create necessary function declarations and globals
  auto *ctx = &getContext();
  OpBuilder builder(ctx);
  Location loc = moduleOp->getLoc();
  builder.setInsertionPointToEnd(moduleOp.getBody());
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  // Lookup or declare printf function.
  auto printfFunc =
      LLVM::lookupOrCreateFn(builder, moduleOp, "printf", ptrTy, voidTy, true);
  if (failed(printfFunc)) {
    moduleOp->emitError("failed to lookup or create printf");
    return signalPassFailure();
  }

  // Replace the top-module with a function performing the BMC
  auto entryFunc = func::FuncOp::create(builder, loc, topModule,
                                        builder.getFunctionType({}, {}));
  builder.createBlock(&entryFunc.getBody());

  {
    OpBuilder::InsertionGuard guard(builder);
    auto *terminator = hwModule.getBody().front().getTerminator();
    builder.setInsertionPoint(terminator);
    verif::YieldOp::create(builder, loc, terminator->getOperands());
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
    bmcOp = verif::BoundedModelCheckingOp::create(
        builder, loc, risingClocksOnly ? bound : 2 * bound,
        cast<IntegerAttr>(numRegs).getValue().getZExtValue(), initialValues);
    // Annotate the op with how many cycles to ignore - again, we may need to
    // double this to account for rising and falling edges
    if (ignoreAssertionsUntil)
      bmcOp->setAttr("ignore_asserts_until",
                     builder.getI32IntegerAttr(
                         risingClocksOnly ? ignoreAssertionsUntil
                                          : 2 * ignoreAssertionsUntil));
  } else {
    hwModule->emitOpError("no num_regs or initial_values attribute found - "
                          "please run externalize "
                          "registers pass first");
    return signalPassFailure();
  }

  // Count top-level clock inputs. Each gets an independent toggle entry in the
  // init/loop regions so the BMC explores all possible asynchronous clock
  // interleavings. Struct-embedded clocks are not yet supported.
  unsigned numClocks = 0;
  for (auto input : hwModule.getInputTypes()) {
    if (isa<seq::ClockType>(input)) {
      ++numClocks;
    }
    if (auto hwStruct = dyn_cast<hw::StructType>(input)) {
      for (auto field : hwStruct.getElements()) {
        if (isa<seq::ClockType>(field.type)) {
          hwModule.emitError(
              "designs with struct-embedded clocks not yet supported");
          return signalPassFailure();
        }
      }
    }
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    // Initialize all clocks to 0 (or 1 in rising-clocks-only mode).
    auto *initBlock = builder.createBlock(&bmcOp.getInit());
    builder.setInsertionPointToStart(initBlock);
    SmallVector<Value> initClocks;
    for (unsigned i = 0; i < numClocks; ++i) {
      auto initVal = hw::ConstantOp::create(builder, loc, builder.getI1Type(),
                                            risingClocksOnly ? 1 : 0);
      initClocks.push_back(seq::ToClockOp::create(builder, loc, initVal));
    }
    verif::YieldOp::create(builder, loc, initClocks);

    // Update clocks each step. If syncClocks is false, each clock independently
    // decides whether to toggle using a symbolic value.
    auto *loopBlock = builder.createBlock(&bmcOp.getLoop());
    builder.setInsertionPointToStart(loopBlock);
    for (unsigned i = 0; i < numClocks; ++i)
      loopBlock->addArgument(seq::ClockType::get(ctx), loc);
    SmallVector<Value> nextClocks;
    if (risingClocksOnly) {
      // Rising-clocks-only mode: pass clocks through unchanged.
      for (auto arg : loopBlock->getArguments())
        nextClocks.push_back(arg);
    } else {
      if (syncClocks) {
        // Old behavior: toggle every clock each step.
        for (auto arg : loopBlock->getArguments()) {
          auto fromClk = seq::FromClockOp::create(builder, loc, arg);
          auto cNeg1 =
              hw::ConstantOp::create(builder, loc, builder.getI1Type(), -1);
          auto nClk = comb::XorOp::create(builder, loc, fromClk, cNeg1);
          nextClocks.push_back(seq::ToClockOp::create(builder, loc, nClk));
        }
      } else {
        // New default behavior: each clock independently decides whether to
        // toggle. This allows exploring all possible CDC interleavings.
        for (auto arg : loopBlock->getArguments()) {
          auto fromClk = seq::FromClockOp::create(builder, loc, arg);
          auto toggle =
              verif::SymbolicValueOp::create(builder, loc, builder.getI1Type());
          auto nClk = comb::XorOp::create(builder, loc, fromClk, toggle);
          nextClocks.push_back(seq::ToClockOp::create(builder, loc, nClk));
        }
      }
    }
    verif::YieldOp::create(builder, loc, nextClocks);
  }
  bmcOp.getCircuit().takeBody(hwModule.getBody());
  hwModule->erase();

  // Define global string constants to print on success/failure
  auto createUniqueStringGlobal = [&](StringRef str) -> FailureOr<Value> {
    Location loc = moduleOp.getLoc();

    OpBuilder b = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto arrayTy = LLVM::LLVMArrayType::get(b.getI8Type(), str.size() + 1);
    auto global = LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/true, LLVM::linkage::Linkage::Private,
        "resultString",
        StringAttr::get(b.getContext(), Twine(str).concat(Twine('\00'))));
    SymbolTable symTable(moduleOp);
    if (failed(symTable.renameToUnique(global, {&symTable}))) {
      return mlir::failure();
    }

    return success(
        LLVM::AddressOfOp::create(builder, loc, global)->getResult(0));
  };

  auto successStrAddr =
      createUniqueStringGlobal("Bound reached with no violations!\n");
  auto failureStrAddr =
      createUniqueStringGlobal("Assertion can be violated!\n");

  if (failed(successStrAddr) || failed(failureStrAddr)) {
    moduleOp->emitOpError("could not create result message strings");
    return signalPassFailure();
  }

  auto formatString =
      LLVM::SelectOp::create(builder, loc, bmcOp.getResult(),
                             successStrAddr.value(), failureStrAddr.value());

  LLVM::CallOp::create(builder, loc, printfFunc.value(),
                       ValueRange{formatString});
  func::ReturnOp::create(builder, loc);

  if (insertMainFunc) {
    builder.setInsertionPointToEnd(getOperation().getBody());
    Type i32Ty = builder.getI32Type();
    auto mainFunc = func::FuncOp::create(
        builder, loc, "main", builder.getFunctionType({i32Ty, ptrTy}, {i32Ty}));
    builder.createBlock(&mainFunc.getBody(), {}, {i32Ty, ptrTy}, {loc, loc});
    func::CallOp::create(builder, loc, entryFunc, ValueRange{});
    // TODO: don't use LLVM here
    Value constZero = LLVM::ConstantOp::create(builder, loc, i32Ty, 0);
    func::ReturnOp::create(builder, loc, constZero);
  }
}
