//===- SystemCLowerInstanceInterop.cpp - Instance-side interop lowering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SystemC instance-side interp lowering pass implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Interop/InteropOps.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "circt/Dialect/SystemC/SystemCPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace systemc {
#define GEN_PASS_DEF_SYSTEMCLOWERINSTANCEINTEROP
#include "circt/Dialect/SystemC/Passes.h.inc"
} // namespace systemc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::systemc;

//===----------------------------------------------------------------------===//
// Interop lowering patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower the systemc::InteropVerilatedOp operation.
class InteropVerilatedOpConversion
    : public OpConversionPattern<InteropVerilatedOp> {
public:
  using OpConversionPattern<InteropVerilatedOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InteropVerilatedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: instead of hardcoding the verilated module's class name, it should
    // be derived from a configs attribute as this can be specified via the CLI
    // arguments of verilator
    // stateType ::= VModuleName*
    SmallString<128> verilatedModuleName("V");
    verilatedModuleName += op.getModuleName();
    auto stateType = emitc::PointerType::get(
        emitc::OpaqueType::get(op->getContext(), verilatedModuleName));
    Location loc = op.getLoc();

    // Include the C++ header produced by Verilator at the location of the HW
    // module.
    auto *hwModule =
        SymbolTable::lookupNearestSymbolFrom(op, op.getModuleNameAttr());
    OpBuilder includeBuilder(hwModule);
    emitc::IncludeOp::create(includeBuilder, loc,
                             (verilatedModuleName + ".h").str(), false);

    // Request a pointer to the verilated module as persistent state.
    Value state = interop::ProceduralAllocOp::create(rewriter, loc, stateType,
                                                     InteropMechanism::CPP)
                      .getStates()[0];

    insertStateInitialization(rewriter, loc, state);

    ValueRange results = insertUpdateLogic(
        rewriter, loc, state, adaptor.getInputs(), op.getResults(),
        adaptor.getInputNames(), adaptor.getResultNames());

    insertStateDeallocation(rewriter, loc, state);

    // Replace the return values of the instance with the result values of the
    // interop update operation.
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  /// Insert a interop init operation to allocate an instance of the verilated
  /// module on the heap and let the above requested pointer point to it.
  void insertStateInitialization(PatternRewriter &rewriter, Location loc,
                                 Value state) const {
    auto initOp = interop::ProceduralInitOp::create(rewriter, loc, state,
                                                    InteropMechanism::CPP);

    OpBuilder initBuilder = OpBuilder::atBlockBegin(initOp.getBody());
    Value newState =
        NewOp::create(initBuilder, loc, state.getType(), ValueRange());
    interop::ReturnOp::create(initBuilder, loc, newState);
  }

  /// Create an update interop operation to assign the input values to the input
  /// ports of the verilated module, call 'eval', and read the output ports of
  /// the verilated module.
  ValueRange insertUpdateLogic(PatternRewriter &rewriter, Location loc,
                               Value stateValue, ValueRange inputValues,
                               ValueRange resultValues, ArrayAttr inputNames,
                               ArrayAttr resultNames) const {
    auto updateOp = interop::ProceduralUpdateOp::create(
        rewriter, loc, resultValues.getTypes(), inputValues, stateValue,
        InteropMechanism::CPP);

    OpBuilder updateBuilder = OpBuilder::atBlockBegin(updateOp.getBody());

    // Write to the verilated module's input ports.
    Value state = updateOp.getBody()->getArguments().front();
    for (size_t i = 0; i < inputValues.size(); ++i) {
      Value member = MemberAccessOp::create(
          updateBuilder, loc, inputValues[i].getType(), state,
          cast<StringAttr>(inputNames[i]), MemberAccessKind::Arrow);
      AssignOp::create(updateBuilder, loc, member,
                       updateOp.getBody()->getArgument(i + 1));
    }

    // Call 'eval'.
    auto evalFunc = MemberAccessOp::create(
        updateBuilder, loc,
        FunctionType::get(updateBuilder.getContext(), {}, {}), state, "eval",
        MemberAccessKind::Arrow);

    // TODO: this has to be changed to a systemc::CallIndirectOp once the PR is
    // merged, also remove the dependency to the func dialect from the cmake,
    // header include, pass dependent dialects
    func::CallIndirectOp::create(updateBuilder, loc, evalFunc.getResult());

    // Read the verilated module's output ports.
    SmallVector<Value> results;
    for (size_t i = 0; i < resultValues.size(); ++i) {
      results.push_back(MemberAccessOp::create(
          updateBuilder, loc, resultValues[i].getType(), state,
          cast<StringAttr>(resultNames[i]).getValue(),
          MemberAccessKind::Arrow));
    }

    interop::ReturnOp::create(updateBuilder, loc, results);

    return updateOp->getResults();
  }

  /// Deallocate the memory allocated in the interop init operation.
  void insertStateDeallocation(PatternRewriter &rewriter, Location loc,
                               Value state) const {
    auto deallocOp = interop::ProceduralDeallocOp::create(
        rewriter, loc, state, InteropMechanism::CPP);

    OpBuilder deallocBuilder = OpBuilder::atBlockBegin(deallocOp.getBody());
    DeleteOp::create(deallocBuilder, loc, deallocOp.getBody()->getArgument(0));
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

namespace {
struct SystemCLowerInstanceInteropPass
    : circt::systemc::impl::SystemCLowerInstanceInteropBase<
          SystemCLowerInstanceInteropPass> {
  void runOnOperation() override;
};
} // namespace

void circt::systemc::populateSystemCLowerInstanceInteropPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<InteropVerilatedOpConversion>(ctx);
}

void SystemCLowerInstanceInteropPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<interop::InteropDialect>();
  target.addLegalDialect<emitc::EmitCDialect>();
  target.addLegalDialect<SystemCDialect>();
  target.addLegalOp<func::CallIndirectOp>();
  target.addIllegalOp<InteropVerilatedOp>();

  // Setup the conversion.
  populateSystemCLowerInstanceInteropPatterns(patterns, &getContext());

  // Apply the partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create the SystemC Lower Interop pass.
std::unique_ptr<Pass> circt::systemc::createSystemCLowerInstanceInteropPass() {
  return std::make_unique<SystemCLowerInstanceInteropPass>();
}
