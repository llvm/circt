//===- DCTestSCFToDC.cpp - SCF to DC test pass ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This pass tests the SCF to DC conversion patterns by defining a simple
//  func.func-based conversion pass. It should not be used for anything but
//  testing the conversion patterns, given its lack of handling anything but
//  SCF ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/DC/DCOps.h"
#include "circt/Dialect/DC/DCPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/ValueMapper.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace dc;
using namespace mlir;

namespace circt {
namespace dc {

class ControlFlowConverter;

// A ControlFlowConversionPattern represents the lowering of a control flow
// construct to DC.
class ControlFlowConversionPatternBase {
public:
  ControlFlowConversionPatternBase(ControlFlowConverter &converter);
  virtual ~ControlFlowConversionPatternBase() = default;

  // Return true if this pattern matches the given operation.
  virtual bool matches(Operation *op) const = 0;
  virtual FailureOr<Value> dispatch(Value control, Operation *op) = 0;

protected:
  ControlFlowConverter &converter;
  OpBuilder &b;
  ValueMapper &mapper;
};

template <typename TOp>
class ControlFlowConversionPattern : public ControlFlowConversionPatternBase {
public:
  using ControlFlowConversionPatternBase::ControlFlowConversionPatternBase;
  using OpTy = TOp;
  // Convert the registered operation to DC.
  // The 'control' represents the incoming !dc.token-typed control value that
  // hands off control to this control operator.
  // The conversion is expected to return a !dc.value-typed value that
  // represents the outgoing control value that hands off control away from
  // this operation after it has been executed.
  virtual FailureOr<Value> convert(Value control, TOp op) = 0;
  FailureOr<Value> dispatch(Value control, Operation *op) override {
    return convert(control, cast<TOp>(op));
  }
  bool matches(Operation *op) const override { return isa<TOp>(op); }
};

// The ControlFlowConverter is a container of ControlFlowConvertionPatterns and
// is what drives a conversion from control flow ops to DC.
// Assumes that the insertion point of the opbuilder is set to where the
// converted region ops should be inserted.
class ControlFlowConverter {
public:
  ControlFlowConverter(OpBuilder &b, ValueMapper &mapper)
      : b(b), mapper(mapper) {}
  virtual ~ControlFlowConverter() = default;

  OpBuilder &b;
  ValueMapper &mapper;

  template <typename TConverter>
  void add() {
    static_assert(
        std::is_base_of<ControlFlowConversionPatternBase, TConverter>::value,
        "TConverter must be a subclass of ControlFlowConversionPatternBase");
    auto &converter =
        converters.emplace_back(std::make_unique<TConverter>(*this));
    converterLookup[b.getStringAttr(TConverter::OpTy::getOperationName())] =
        converter.get();
  }

  // Converts a region to DC using the registered patterns.
  virtual FailureOr<Value> convert(mlir::Value control, Region &region) = 0;

protected:
  // An analysis which determines which SSA values used within a region that
  // are defined outside of said region (i.e. a value is referenced via.
  // dominance).
  // TODO: do we even need this? we should probably just expect that there is
  // always a ValueMapper'd value for every (non-control) SSA value. However...
  // when working on SCF, we do NOT have an SSA maximization pass, which means
  // that we have to deal with SSA value liveness somehow.
  // DominanceValueUsages valueUsage;

  llvm::SmallVector<std::unique_ptr<ControlFlowConversionPatternBase>>
      converters;
  llvm::DenseMap<mlir::StringAttr, ControlFlowConversionPatternBase *>
      converterLookup;
};

ControlFlowConversionPatternBase::ControlFlowConversionPatternBase(
    ControlFlowConverter &converter)
    : converter(converter), b(converter.b), mapper(converter.mapper) {}

class TestControlFlowConverter : public ControlFlowConverter {
public:
  using ControlFlowConverter::ControlFlowConverter;
  FailureOr<Value> convert(mlir::Value control, Region &region) override;
};

FailureOr<Value> TestControlFlowConverter::convert(mlir::Value control,
                                                   Region &region) {
  if (!region.hasOneBlock())
    assert(false && "Only single-block regions are supported");

  for (auto &op : llvm::make_early_inc_range(region.front())) {
    auto it = converterLookup.find(op.getName().getIdentifier());
    if (it == converterLookup.end()) {
      // If no converter was registered for the op, assume it was a
      // non-control-flow op. Just copy it to the new region and map the
      // operands and results.
      Operation *clonedOp = b.clone(op);
      for (auto [i, origOperand] : llvm::enumerate(op.getOperands()))
        clonedOp->setOperand(i, mapper.get(origOperand));

      for (auto [i, origResult] : llvm::enumerate(op.getResults()))
        mapper.set(origResult, clonedOp->getResult(i));
      continue;
    }

    auto &converter = it->second;
    auto res = converter->dispatch(control, &op);
    if (failed(res))
      return failure();

    // The result of the conversion is the new current control token.
    control = *res;
  }

  return {control};
}

// Shorthand for joining on a range of DC typed values; will automatically
// insert unpack ops if necessary.
static Value join(OpBuilder &b, ValueRange values) {
  llvm::SmallVector<Value> tokens;
  for (auto v : values) {
    auto convertedValue = v;
    if (convertedValue.getType().isa<dc::ValueType>())
      convertedValue = b.create<dc::UnpackOp>(v.getLoc(), v).getToken();
    tokens.push_back(convertedValue);
  }

  return b.create<dc::JoinOp>(values.front().getLoc(), tokens).getResult();
}

// Performs a dc.join on a !dc.token and !dc.value-typed value by unpacking,
// joining, and then repacking
static Value joinValue(OpBuilder &b, Value token, Value value) {
  auto unpack = b.create<dc::UnpackOp>(value.getLoc(), value);
  auto joinedToken = join(b, ValueRange{token, unpack.getToken()});
  return b.create<dc::PackOp>(value.getLoc(), joinedToken, unpack.getOutput());
}

class WhileOpConversionPattern
    : public ControlFlowConversionPattern<scf::WhileOp> {
public:
  using ControlFlowConversionPattern::ControlFlowConversionPattern;
  FailureOr<Value> convert(mlir::Value control, scf::WhileOp op) override {
    BackedgeBuilder bb(b, op.getLoc());
    // =========================================================================
    // Before region
    // =========================================================================

    // Loop priming buffer - this buffer maintains state about whether we're
    // taking an external loop entry activation edge, or a loop backedge.
    // 0 = external entry, 1 = loop backedge.
    auto lprNext = bb.get(b.getType<ValueType>(b.getI1Type()));
    auto lpr = b.create<dc::BufferOp>(
        op.getLoc(), lprNext, b.getI64IntegerAttr(1), b.getI32ArrayAttr({0}));

    auto lprUnpacked = b.create<dc::UnpackOp>(op.getLoc(), lpr);
    Value inLoop = lprUnpacked.getOutput();

    // Condition region - we need to select between the condition
    // region arguments coming from either the external world or the
    // backedge from the 'after'-body.
    llvm::SmallVector<Backedge> afterArgs;
    llvm::SmallVector<Value> beforeSelectedArgs;
    for (auto [beforeArgInit, beforeArg] :
         llvm::zip(op.getInits(), op.getBeforeArguments())) {
      auto afterArg = bb.get(beforeArgInit.getType());
      afterArgs.push_back(afterArg);
      auto beforeSelectedArg = b.create<arith::SelectOp>(
          op.getLoc(), inLoop, afterArg, mapper.get(beforeArgInit));
      beforeSelectedArgs.push_back(beforeSelectedArg);

      // And map the before arguments to the selected arguments.
      mapper.set(beforeArg, beforeSelectedArg);
    }

    // The control going in to the condition region will be an LPR-controlled
    // dc.select op which selects the control token from either the external
    // world or the loop backedge.
    auto loopBackedgeControl = bb.get(b.getType<dc::TokenType>());
    auto condCtrl =
        b.create<dc::SelectOp>(op.getLoc(), lpr, loopBackedgeControl, control);

    // Recurse into the condition region - we still use the incoming control
    // edge here.
    auto beforeRes = converter.convert(condCtrl, op.getBefore());
    if (failed(beforeRes))
      return failure();
    Value beforeControl = *beforeRes;

    // Grab the condition op and use the argument to determine where to forward
    // control to, using a dc.branch.
    scf::ConditionOp condOp = op.getConditionOp();
    Value dstCtrl = b.create<dc::PackOp>(op.getLoc(), beforeControl,
                                         mapper.get(condOp.getCondition()));
    auto branchOp = b.create<dc::BranchOp>(op.getLoc(), dstCtrl);
    Value toAfter = branchOp.getTrueToken();
    Value toExit = branchOp.getFalseToken();

    // The loop priming register follows the convention established earlier
    // (0 = external entry, 1 = loop backedge).
    lprNext.setValue(dstCtrl);

    // Map the return values of the loop to the arguments of the condition op.
    for (auto [ret, condArg] : llvm::zip(op.getResults(), condOp.getArgs()))
      mapper.set(ret, mapper.get(condArg));

    // =========================================================================
    // After region
    // =========================================================================
    // We now have a token that represents control flowing to the 'after'
    // region.
    // First, we replace the 'after' region arguments with the arguments from
    // the condition op.
    for (auto [afterArg, condArgs] :
         llvm::zip(op.getAfterArguments(), condOp.getArgs())) {
      mapper.set(afterArg, mapper.get(condArgs));
    }

    // Then we process the 'after' region
    auto afterRes = converter.convert(toAfter, op.getAfter());
    if (failed(afterRes))
      return failure();

    // And then we need to assign the backedges created earlier for the 'after'
    // region return values and control backedge.
    scf::YieldOp afterYield = op.getYieldOp();
    for (auto [afterArg, afterBackedge] :
         llvm::zip(afterYield.getOperands(), afterArgs)) {
      afterBackedge.setValue(mapper.get(afterArg));
    }
    loopBackedgeControl.setValue(*afterRes);

    // The output control token of the for loop conversion is the toExit token.
    return toExit;
  }
};

class IfOpConversionPattern : public ControlFlowConversionPattern<scf::IfOp> {
public:
  using ControlFlowConversionPattern::ControlFlowConversionPattern;
  FailureOr<Value> convert(mlir::Value control, scf::IfOp op) override {
    // =========================================================================
    // Control
    // =========================================================================
    // Pack the incoming control token with the condition value.
    Value mappedCond = mapper.get(op.getCondition());
    auto dcCond =
        b.create<dc::PackOp>(op.getLoc(), control, mappedCond).getResult();

    // Branch on the packed condition.
    auto branchOp = b.create<dc::BranchOp>(op.getLoc(), dcCond);

    auto thenBranchRes =
        converter.convert(branchOp.getTrueToken(), op.getThenRegion());
    if (!op.elseBlock()) {
      assert(op.getNumResults() == 0 &&
             "IfOp with no else must have no results");
      return thenBranchRes;
    }

    // Have to merge the true and false branches to generate the output control
    // token.
    std::optional<FailureOr<Value>> elseBranchRes;
    elseBranchRes =
        converter.convert(branchOp.getFalseToken(), op.getElseRegion());

    if (failed(thenBranchRes) || failed(*elseBranchRes))
      return failure();

    auto ctrlOut =
        b.create<dc::UnpackOp>(
             op.getLoc(),
             b.create<dc::MergeOp>(op.getLoc(), *thenBranchRes, **elseBranchRes)
                 .getResult())
            .getToken();

    // =========================================================================
    // Data
    // =========================================================================
    // In case the scf.if yielded a value, we need to select the value from
    // either of the yield statements. We do this through an arith.select with
    // the condition being the select index.
    if (op.getNumResults() != 0) {
      llvm::SmallVector<Value> trueOuts, falseOuts;
      llvm::copy(op.thenYield().getOperands(), std::back_inserter(trueOuts));
      llvm::copy(op.elseYield().getOperands(), std::back_inserter(falseOuts));

      // Create an arith.select for each of the results and remap.
      for (auto [trueOut, falseOut, origOut] :
           llvm::zip(trueOuts, falseOuts, op.getResults())) {
        auto mappedOut = b.create<arith::SelectOp>(
            op.getLoc(), mappedCond, mapper.get(trueOut), mapper.get(falseOut));
        mapper.set(origOut, mappedOut);
      }
    }
    return ctrlOut;
  }
};

// Ignores an operation. This implies that other patterns will handle the
// operation, instead of letting the converter perform its default operation
// handling.
template <typename TOp>
class IgnoreOpConversionPattern : public ControlFlowConversionPattern<TOp> {
public:
  using ControlFlowConversionPattern<TOp>::ControlFlowConversionPattern;
  FailureOr<Value> convert(mlir::Value control, TOp) override {
    return control;
  }
};

class ReturnOpConversionPattern
    : public ControlFlowConversionPattern<func::ReturnOp> {
public:
  using ControlFlowConversionPattern::ControlFlowConversionPattern;
  FailureOr<Value> convert(mlir::Value control, func::ReturnOp op) override {
    // Pack the operands of the return op with the incoming control.
    llvm::SmallVector<Value> packedReturns;
    for (auto operand : mapper.get(op.getOperands()))
      packedReturns.push_back(
          b.create<dc::PackOp>(op.getLoc(), control, operand).getResult());

    // Create a hw.output op at the current insertion point.
    b.create<hw::OutputOp>(op.getLoc(), packedReturns);

    // End of the line!
    return Value();
  }
};

static hw::HWModuleOp createConvertedFunc(OpBuilder &b, func::FuncOp src) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(src);

  llvm::SmallVector<hw::PortInfo> inPorts, outPorts;

  // Create the new function with the same signature as the old one but with
  // !dc.value-typed arguments.
  for (auto [i, argType] : llvm::enumerate(src.getArgumentTypes())) {
    inPorts.push_back(hw::PortInfo{b.getStringAttr("in" + Twine(i)),
                                   dc::ValueType::get(b.getContext(), argType),
                                   hw::ModulePort::Direction::Input});
  }

  for (auto [i, resType] : llvm::enumerate(src.getResultTypes())) {
    outPorts.push_back(hw::PortInfo{b.getStringAttr("out" + Twine(i)),
                                    dc::ValueType::get(b.getContext(), resType),
                                    hw::ModulePort::Direction::Output});
  }

  auto mod = b.create<hw::HWModuleOp>(src.getLoc(), src.getNameAttr(),
                                      hw::ModulePortInfo(inPorts, outPorts));
  mod.getBodyBlock()->getTerminator()->erase();
  return mod;
}

} // namespace dc
} // namespace circt

namespace {
struct DCTestSCFToDCPass : public DCTestSCFToDCBase<DCTestSCFToDCPass> {
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    OpBuilder b(mod.getContext());
    for (auto func : llvm::make_early_inc_range(mod.getOps<func::FuncOp>())) {
      // Create the arg-converted, hw-module converted top-level op.
      auto convFunc = createConvertedFunc(b, func);
      b.setInsertionPointToStart(convFunc.getBodyBlock());

      // Initialize the value mapper by unpacking the function arguments.
      ValueMapper valueMapper;
      llvm::SmallVector<Value> argTokens;
      for (auto [orig, conv] : llvm::zip(
               func.getArguments(), convFunc.getBodyBlock()->getArguments())) {
        auto argUnpack = b.create<dc::UnpackOp>(orig.getLoc(), conv);
        valueMapper.set(orig, argUnpack.getOutput());
        argTokens.push_back(argUnpack.getToken());
      }

      TestControlFlowConverter converter(b, valueMapper);
      converter.add<IfOpConversionPattern>();
      converter.add<IgnoreOpConversionPattern<scf::YieldOp>>();
      converter.add<IgnoreOpConversionPattern<scf::ConditionOp>>();
      converter.add<ReturnOpConversionPattern>();
      converter.add<WhileOpConversionPattern>();

      // To prime the conversion, we need the incoming control token. DEfine
      // this as the join of all incoming control tokens.
      Value controlIn = b.create<dc::JoinOp>(func.getLoc(), argTokens);

      if (failed(converter.convert(controlIn, func.getBody())))
        return signalPassFailure();

      // Remove the original function.
      func.erase();
    }
  };
};

} // namespace

std::unique_ptr<mlir::Pass> circt::dc::createDCTestSCFToDCPass() {
  return std::make_unique<DCTestSCFToDCPass>();
}
