//===- LowerDPIToArcs.cpp - Lower DPI ops to Arc  -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===
//
// This pass lowers Sim DPI operations to Arc and an external function call.
//
// sim.dpi.func @foo(input %a: i32, output %b: i64)
// hw.module @top (..) {
//    %result = sim.dpi.call @foo(%a) clock %clock
// }
//
// ->
//
// func.func @foo(%a: i32, %b: !llvm.ptr) // Output is passed by a reference.
// arc.define @foo_arc_def(%a: i32) -> (i64) {
//    %0 = llvm.alloca: !llvm.ptr
//    %v = func.call @foo (%a, %0)
//    arc.return %v:
// }
// hw.module @mod(..) {
//   arc.state @foo_arc_def ()
// }
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Sim/SimOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-lower-dpi-to-arcs"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_LOWERDPITOARCS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct LowerDPIToArcsPass
    : public arc::impl::LowerDPIToArcsBase<LowerDPIToArcsPass> {

  LogicalResult lowerDPI();
  LogicalResult lowerDPIFuncOp(sim::DPIFuncOp simFunc,
                               DenseMap<StringAttr, arc::DefineOp> &symToArcDef,
                               SymbolTable &symbolTable);
  void runOnOperation() override;
};

struct DPICallOpLowering : public OpConversionPattern<sim::DPICallOp> {
  DPICallOpLowering(const DenseMap<StringAttr, arc::DefineOp> &symToArcDef,
                    const TypeConverter &typeConverter, MLIRContext *context)
      : OpConversionPattern(typeConverter, context), symToArcDef(symToArcDef) {}
  LogicalResult
  matchAndRewrite(sim::DPICallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto enable = adaptor.getEnable();
    auto clock = adaptor.getClock();
    if (clock) {
      // Replace DPI call with a state. Latency is 1 for clocked one.
      rewriter.replaceOpWithNewOp<arc::StateOp>(
          op, symToArcDef.at(op.getCalleeAttr().getAttr()), clock, enable,
          /*latency=*/1, adaptor.getInputs());
    } else {
      // Unclocked call.
      if (enable)
        // TODO: Lower it to scf.if
        return op->emitError("unclocked call with enable is unsupported now");

      // Replace DPI call with a call since latency is 0.
      rewriter.replaceOpWithNewOp<arc::CallOp>(
          op, symToArcDef.at(op.getCalleeAttr().getAttr()),
          adaptor.getInputs());
    }
    return success();
  }

private:
  const DenseMap<StringAttr, arc::DefineOp> &symToArcDef;
};

} // namespace

static void populateLegality(ConversionTarget &target) {
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<arc::ArcDialect>();

  target.addIllegalOp<sim::DPIFuncOp>();
  target.addIllegalOp<sim::DPICallOp>();
}

static void populateTypeConversion(TypeConverter &typeConverter) {
  typeConverter.addConversion([&](Type type) { return type; });
}

LogicalResult LowerDPIToArcsPass::lowerDPIFuncOp(
    sim::DPIFuncOp simFunc, DenseMap<StringAttr, arc::DefineOp> &symToArcDef,
    SymbolTable &symbolTable) {
  ImplicitLocOpBuilder builder(simFunc.getLoc(), simFunc);
  auto moduleType = simFunc.getModuleType();

  llvm::SmallVector<Type> dpiFunctionArgumentTypes;
  for (auto arg : moduleType.getPorts()) {
    // TODO: Support a non-integer type.
    if (!arg.type.isInteger())
      return simFunc->emitError()
             << "non-integer type argument is unsupported now";

    if (arg.dir == hw::ModulePort::Input)
      dpiFunctionArgumentTypes.push_back(arg.type);
    else
      // Output must be passed by a reference.
      dpiFunctionArgumentTypes.push_back(
          LLVM::LLVMPointerType::get(arg.type.getContext()));
  }

  auto funcType = builder.getFunctionType(dpiFunctionArgumentTypes, {});
  func::FuncOp func;

  // Look up func.func by verilog name since the function name is equal to the
  // symbol name in MLIR
  if (auto verilogName = simFunc.getVerilogName()) {
    func = dyn_cast_or_null<func::FuncOp>(symbolTable.lookup(*verilogName));
    // TODO: Check if function type matches.
  }

  // If a referred function is not in the same module, create an external
  // function declaration.
  if (!func) {
    func = builder.create<func::FuncOp>(simFunc.getVerilogName()
                                            ? *simFunc.getVerilogName()
                                            : simFunc.getSymName(),
                                        funcType);
    // External function needs to be private.
    func.setPrivate();
  }

  // Create an Arc.
  SmallString<8> arcDefName;
  arcDefName += simFunc.getSymName();
  arcDefName += "_dpi_arc";
  // FIXME: Unique symbol.
  auto arcOp =
      builder.create<arc::DefineOp>(arcDefName, moduleType.getFuncType());

  auto inserted = symToArcDef.insert({simFunc.getSymNameAttr(), arcOp}).second;
  (void)inserted;
  assert(inserted && "symbol must be unique");

  builder.setInsertionPointToStart(arcOp.addEntryBlock());
  SmallVector<Value> functionInputs;
  SmallVector<LLVM::AllocaOp> functionOutputAllocas;

  size_t inputIndex = 0;
  for (auto [idx, arg] : llvm::enumerate(moduleType.getPorts())) {
    if (arg.dir == hw::ModulePort::Input) {
      functionInputs.push_back(arcOp.getArgument(inputIndex));
      ++inputIndex;
    } else {
      // Allocate an output placeholder.
      auto one = builder.create<LLVM::ConstantOp>(builder.getI64IntegerAttr(1));
      auto alloca = builder.create<LLVM::AllocaOp>(
          builder.getType<LLVM::LLVMPointerType>(), arg.type, one);
      functionInputs.push_back(alloca);
      functionOutputAllocas.push_back(alloca);
    }
  }

  builder.create<func::CallOp>(func, functionInputs);

  // Construct outputs of this Arc(= alloca loads).
  SmallVector<Value> results;
  for (auto functionOutputAlloca : functionOutputAllocas)
    results.push_back(builder.create<LLVM::LoadOp>(
        functionOutputAlloca.getElemType(), functionOutputAlloca));

  builder.create<arc::OutputOp>(results);

  simFunc.erase();
  return success();
}

LogicalResult LowerDPIToArcsPass::lowerDPI() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering DPI to arc and func\n");
  auto op = getOperation();
  DenseMap<StringAttr, arc::DefineOp> symToArcDef;
  auto &symbolTable = getAnalysis<SymbolTable>();

  for (auto simFunc : llvm::make_early_inc_range(op.getOps<sim::DPIFuncOp>()))
    if (failed(lowerDPIFuncOp(simFunc, symToArcDef, symbolTable)))
      return failure();

  ConversionTarget target(getContext());
  TypeConverter converter;
  RewritePatternSet patterns(&getContext());
  populateLegality(target);
  populateTypeConversion(converter);
  patterns.add<DPICallOpLowering>(symToArcDef, converter, &getContext());
  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

void LowerDPIToArcsPass::runOnOperation() {
  if (failed(lowerDPI()))
    return signalPassFailure();
}

std::unique_ptr<Pass> arc::createLowerDPIToArcsPass() {
  return std::make_unique<LowerDPIToArcsPass>();
}
