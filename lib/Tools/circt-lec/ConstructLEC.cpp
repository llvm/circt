//===- ConstructLEC.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Tools/circt-lec/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_CONSTRUCTLEC
#include "circt/Tools/circt-lec/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// ConstructLEC pass
//===----------------------------------------------------------------------===//

namespace {
struct ConstructLECPass
    : public circt::impl::ConstructLECBase<ConstructLECPass> {
  using circt::impl::ConstructLECBase<ConstructLECPass>::ConstructLECBase;
  void runOnOperation() override;
  hw::HWModuleOp lookupModule(StringRef name);
  Value constructMiter(OpBuilder builder, Location loc, hw::HWModuleOp moduleA,
                       hw::HWModuleOp moduleB, bool withResult);
};
} // namespace

static Value lookupOrCreateStringGlobal(OpBuilder &builder, ModuleOp moduleOp,
                                        StringRef str) {
  Location loc = moduleOp.getLoc();
  auto global = moduleOp.lookupSymbol<LLVM::GlobalOp>(str);
  if (!global) {
    OpBuilder b = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto arrayTy = LLVM::LLVMArrayType::get(b.getI8Type(), str.size() + 1);
    global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/true, LLVM::linkage::Linkage::Private, str,
        StringAttr::get(b.getContext(), Twine(str).concat(Twine('\00'))));
  }

  // FIXME: sanity check the fetched global: do all the attributes match what
  // we expect?

  return builder.create<LLVM::AddressOfOp>(loc, global);
}

hw::HWModuleOp ConstructLECPass::lookupModule(StringRef name) {
  Operation *expectedModule = SymbolTable::lookupNearestSymbolFrom(
      getOperation(), StringAttr::get(&getContext(), name));
  if (!expectedModule || !isa<hw::HWModuleOp>(expectedModule)) {
    getOperation().emitError("module named '") << name << "' not found";
    return {};
  }
  return cast<hw::HWModuleOp>(expectedModule);
}

Value ConstructLECPass::constructMiter(OpBuilder builder, Location loc,
                                       hw::HWModuleOp moduleA,
                                       hw::HWModuleOp moduleB,
                                       bool withResult) {

  // Create the miter circuit that return equivalence result.
  auto lecOp =
      builder.create<verif::LogicEquivalenceCheckingOp>(loc, withResult);

  builder.cloneRegionBefore(moduleA.getBody(), lecOp.getFirstCircuit(),
                            lecOp.getFirstCircuit().end());
  builder.cloneRegionBefore(moduleB.getBody(), lecOp.getSecondCircuit(),
                            lecOp.getSecondCircuit().end());

  moduleA->erase();
  if (moduleA != moduleB)
    moduleB->erase();

  {
    auto *term = lecOp.getFirstCircuit().front().getTerminator();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(term);
    builder.create<verif::YieldOp>(loc, term->getOperands());
    term->erase();
    term = lecOp.getSecondCircuit().front().getTerminator();
    builder.setInsertionPoint(term);
    builder.create<verif::YieldOp>(loc, term->getOperands());
    term->erase();
  }

  sortTopologically(&lecOp.getFirstCircuit().front());
  sortTopologically(&lecOp.getSecondCircuit().front());

  return withResult ? lecOp.getIsProven() : Value{};
}

void ConstructLECPass::runOnOperation() {
  // Create necessary function declarations and globals
  OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
  Location loc = getOperation()->getLoc();

  // Lookup the modules.
  auto moduleA = lookupModule(firstModule);
  if (!moduleA)
    return signalPassFailure();
  auto moduleB = lookupModule(secondModule);
  if (!moduleB)
    return signalPassFailure();

  if (moduleA.getModuleType() != moduleB.getModuleType()) {
    moduleA.emitError("module's IO types don't match second modules: ")
        << moduleA.getModuleType() << " vs " << moduleB.getModuleType();
    return signalPassFailure();
  }

  // Only construct the miter with no additional insertions.
  if (insertMode == lec::InsertAdditionalModeEnum::None) {
    constructMiter(builder, loc, moduleA, moduleB, /*withResult*/ false);
    return;
  }

  mlir::FailureOr<mlir::LLVM::LLVMFuncOp> printfFunc;
  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  auto voidTy = LLVM::LLVMVoidType::get(&getContext());
  // Lookup or declare printf function.
  printfFunc = LLVM::lookupOrCreateFn(builder, getOperation(), "printf", ptrTy,
                                      voidTy, true);
  if (failed(printfFunc)) {
    getOperation()->emitError("failed to lookup or create printf");
    return signalPassFailure();
  }

  // Reuse the name of the first module for the entry function, so we don't
  // have to do any uniquing and the LEC driver also already knows this name.
  FunctionType functionType = FunctionType::get(&getContext(), {}, {});
  func::FuncOp entryFunc =
      builder.create<func::FuncOp>(loc, firstModule, functionType);

  if (insertMode == lec::InsertAdditionalModeEnum::Main) {
    OpBuilder::InsertionGuard guard(builder);
    auto i32Ty = builder.getI32Type();
    auto mainFunc = builder.create<func::FuncOp>(
        loc, "main", builder.getFunctionType({i32Ty, ptrTy}, {i32Ty}));
    builder.createBlock(&mainFunc.getBody(), {}, {i32Ty, ptrTy}, {loc, loc});
    builder.create<func::CallOp>(loc, entryFunc, ValueRange{});
    // TODO: don't use LLVM here
    Value constZero = builder.create<LLVM::ConstantOp>(loc, i32Ty, 0);
    builder.create<func::ReturnOp>(loc, constZero);
  }

  builder.createBlock(&entryFunc.getBody());

  // Create the miter circuit that returns equivalence result.
  auto areEquivalent =
      constructMiter(builder, loc, moduleA, moduleB, /*withResult*/ true);
  assert(!!areEquivalent && "Expected LEC operation with result.");

  // TODO: we should find a more elegant way of reporting the result than
  // already inserting some LLVM here
  Value eqFormatString =
      lookupOrCreateStringGlobal(builder, getOperation(), "c1 == c2\n");
  Value neqFormatString =
      lookupOrCreateStringGlobal(builder, getOperation(), "c1 != c2\n");
  Value formatString = builder.create<LLVM::SelectOp>(
      loc, areEquivalent, eqFormatString, neqFormatString);
  builder.create<LLVM::CallOp>(loc, printfFunc.value(),
                               ValueRange{formatString});

  builder.create<func::ReturnOp>(loc, ValueRange{});
}
