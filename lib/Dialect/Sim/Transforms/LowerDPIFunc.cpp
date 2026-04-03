//===- LowerDPIFunc.cpp - Lower sim.dpi.func to func.func  ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===
//
// This pass lowers Sim DPI func ops to MLIR func and call.
//
// sim.dpi.func @foo(input %a: i32, output %b: i64)
// hw.module @top (..) {
//    %result = sim.dpi.call @foo(%a) clock %clock
// }
//
// ->
//
// func.func @foo(%a: i32, %b: !llvm.ptr) // Output is passed by a reference.
// func.func @foo_wrapper(%a: i32) -> (i64) {
//    %0 = llvm.alloca: !llvm.ptr
//    %v = func.call @foo (%a, %0)
//    func.return %v
// }
// hw.module @mod(..) {
//    %result = sim.dpi.call @foo_wrapper(%a) clock %clock
// }
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "sim-lower-dpi-func"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_LOWERDPIFUNC
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

struct LoweringState {
  DenseMap<StringAttr, func::FuncOp> dpiFuncDeclMapping;
  circt::Namespace nameSpace;
};

struct LowerDPIFuncPass : public sim::impl::LowerDPIFuncBase<LowerDPIFuncPass> {

  LogicalResult lowerDPI();
  LogicalResult lowerDPIFuncOp(sim::DPIFuncOp simFunc,
                               LoweringState &loweringState,
                               SymbolTable &symbolTable);
  void runOnOperation() override;
};

} // namespace

LogicalResult LowerDPIFuncPass::lowerDPIFuncOp(sim::DPIFuncOp simFunc,
                                               LoweringState &loweringState,
                                               SymbolTable &symbolTable) {
  ImplicitLocOpBuilder builder(simFunc.getLoc(), simFunc);
  auto dpiType = simFunc.getDpiFunctionType();
  auto dpiArgs = dpiType.getArguments();
  auto *returnArg = dpiType.getReturnArgument();
  Type explicitReturnType = returnArg ? returnArg->type : Type{};

  llvm::SmallVector<Type> dpiFunctionArgumentTypes;
  for (auto &arg : dpiArgs) {
    // TODO: Support additional non-integer types beyond the pointer-shaped
    // open-array ABI path.
    if (!arg.type.isInteger() && !(isa<LLVM::LLVMPointerType>(arg.type) &&
                                   (arg.dir == sim::DPIDirection::Input ||
                                    arg.dir == sim::DPIDirection::Ref)))
      return simFunc->emitError()
             << "non-integer type argument is unsupported now";

    switch (arg.dir) {
    case sim::DPIDirection::Input:
      dpiFunctionArgumentTypes.push_back(arg.type);
      break;
    case sim::DPIDirection::Output:
    case sim::DPIDirection::InOut:
      dpiFunctionArgumentTypes.push_back(
          LLVM::LLVMPointerType::get(arg.type.getContext()));
      break;
    case sim::DPIDirection::Return:
      break;
    case sim::DPIDirection::Ref:
      dpiFunctionArgumentTypes.push_back(arg.type);
      break;
    }
  }

  auto funcType = builder.getFunctionType(
      dpiFunctionArgumentTypes,
      explicitReturnType ? TypeRange{explicitReturnType} : TypeRange{});
  func::FuncOp func;

  // Look up func.func by verilog name since the function name is equal to the
  // symbol name in MLIR
  if (auto verilogName = simFunc.getVerilogName()) {
    func = symbolTable.lookup<func::FuncOp>(*verilogName);
    if (func && func.getFunctionType() != funcType)
      return simFunc.emitOpError()
             << "references existing func.func @" << *verilogName
             << " with incompatible type " << func.getFunctionType()
             << "; expected " << funcType;
  }

  // If a referred function is not in the same module, create an external
  // function declaration.
  if (!func) {
    func = func::FuncOp::create(builder,
                                simFunc.getVerilogName()
                                    ? *simFunc.getVerilogName()
                                    : simFunc.getSymName(),
                                funcType);
    // External function needs to be private.
    func.setPrivate();
  }

  // Create a wrapper module that calls a DPI function.
  auto wrapperFuncType = cast<FunctionType>(simFunc.getFunctionType());
  auto funcOp = func::FuncOp::create(
      builder,
      loweringState.nameSpace.newName(simFunc.getSymName() + "_wrapper"),
      wrapperFuncType);

  // Map old symbol to a new func op.
  loweringState.dpiFuncDeclMapping[simFunc.getSymNameAttr()] = funcOp;

  builder.setInsertionPointToStart(funcOp.addEntryBlock());
  SmallVector<Value> functionInputs;
  SmallVector<std::pair<Type, Value>> wrapperOutputAllocas;
  Value explicitReturnValue;

  size_t inputIndex = 0;
  for (auto &arg : dpiArgs) {
    switch (arg.dir) {
    case sim::DPIDirection::Input:
      functionInputs.push_back(funcOp.getArgument(inputIndex));
      ++inputIndex;
      break;
    case sim::DPIDirection::Return:
      break;
    case sim::DPIDirection::Ref:
      functionInputs.push_back(funcOp.getArgument(inputIndex));
      ++inputIndex;
      break;
    case sim::DPIDirection::Output: {
      auto one =
          LLVM::ConstantOp::create(builder, builder.getI64IntegerAttr(1));
      auto alloca = LLVM::AllocaOp::create(
          builder, builder.getType<LLVM::LLVMPointerType>(), arg.type, one);
      functionInputs.push_back(alloca);
      wrapperOutputAllocas.push_back({arg.type, alloca});
      break;
    }
    case sim::DPIDirection::InOut: {
      auto one =
          LLVM::ConstantOp::create(builder, builder.getI64IntegerAttr(1));
      auto alloca = LLVM::AllocaOp::create(
          builder, builder.getType<LLVM::LLVMPointerType>(), arg.type, one);
      LLVM::StoreOp::create(builder, funcOp.getArgument(inputIndex), alloca);
      ++inputIndex;
      functionInputs.push_back(alloca);
      wrapperOutputAllocas.push_back({arg.type, alloca});
      break;
    }
    }
  }

  auto call = func::CallOp::create(builder, func, functionInputs);
  if (explicitReturnType)
    explicitReturnValue = call.getResult(0);

  SmallVector<Value> results;
  results.reserve(wrapperOutputAllocas.size() + (explicitReturnValue ? 1 : 0));
  for (auto [type, alloca] : wrapperOutputAllocas)
    results.push_back(LLVM::LoadOp::create(builder, type, alloca));
  if (explicitReturnValue)
    results.push_back(explicitReturnValue);

  func::ReturnOp::create(builder, results);

  simFunc.erase();
  return success();
}

LogicalResult LowerDPIFuncPass::lowerDPI() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering sim DPI func to func.func\n");
  auto op = getOperation();
  LoweringState state;
  state.nameSpace.add(op);
  auto &symbolTable = getAnalysis<SymbolTable>();
  for (auto simFunc : llvm::make_early_inc_range(op.getOps<sim::DPIFuncOp>()))
    if (failed(lowerDPIFuncOp(simFunc, state, symbolTable)))
      return failure();

  auto result = op.walk([&](sim::DPICallOp callOp) -> WalkResult {
    auto it = state.dpiFuncDeclMapping.find(callOp.getCalleeAttr().getAttr());
    if (it == state.dpiFuncDeclMapping.end()) {
      callOp.emitOpError() << "references unknown DPI function";
      return WalkResult::interrupt();
    }
    callOp.setCallee(it->second.getSymNameAttr());
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return success();
}

void LowerDPIFuncPass::runOnOperation() {
  if (failed(lowerDPI()))
    return signalPassFailure();
}
