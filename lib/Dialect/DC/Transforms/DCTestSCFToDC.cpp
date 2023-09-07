//===- DCTestCFToDC.cpp - SCF to DC test pass ------------------*- C++ -*-===//
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

#include "mlir/Dialect/SCF/IR/CF.h"
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
using SplitMergePairs = llvm::DenseMap<Block *, Block *>;

class DCBuilder {
public:
  DCBuilder(OpBuilder &b) : b(b) {}

  // Unpacks a dc.value-typed value into a dc.token-typed token and a
  std::pair<Value, Value> unpack(Value v) {
    auto unpackOp = b.create<dc::UnpackOp>(v.getLoc(), v);
    return {unpackOp.getToken(), unpackOp.getOutput()};
  }

  // Packs a dc.value-typed value with a dc.token-typed token.
  Value pack(Value token, Value value) {
    return b.create<dc::PackOp>(value.getLoc(), token, value);
  }

  // Selects between two dc.value-typed values based on a dc.value<i1>
  // and returns a dc.value-typed value.
  Value mux(Value select, Value first, Value second) {
    auto [firstToken, firstValue] = unpack(first);
    auto [secondToken, secondValue] = unpack(second);
    auto muxedToken = b.create<dc::SelectOp>(select.getLoc(), select,
                                             firstToken, secondToken);
    auto [selectToken, selectValue] = unpack(muxedToken);
    auto muxedValue = b.create<arith::SelectOp>(select.getLoc(), selectValue,
                                                firstValue, secondValue);
    return pack(muxedToken, muxedValue);
  }

  // Joins a range of DC typed values; will automatically insert unpack ops if
  // necessary.
  Value join(ValueRange values) {
    llvm::SmallVector<Value> tokens;
    for (auto v : values) {
      auto convertedValue = v;
      if (convertedValue.getType().isa<dc::ValueType>())
        convertedValue = b.create<dc::UnpackOp>(v.getLoc(), v).getToken();
      tokens.push_back(convertedValue);
    }

    return b.create<dc::JoinOp>(values.front().getLoc(), tokens).getResult();
  }

private:
  OpBuilder &b;
};

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
  DCBuilder dcb;
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
protected:
  Region &region;

  std::unique_ptr<DominanceInfo> domInfo;
  std::unique_ptr<CFGLoopInfo> loopInfo;

  // A mapping of blocks that form split/merge block pairs.
  SplitMergePairs splitMergeBlockPairs;

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

  // We maintain mappings of the intra-block control tokens and block arguments.
  // These may either be real values or backedges to values that eventually will
  // be defined. This allows us to convert blocks in an arbitrary order, as long
  // as we eventually touch all blocks.
  using ControlEdge = std::pair<Block *, Block *>;
  struct BlockArgs {
    Backedge control;
    llvm::SmallVector<Backedge> args;
    static BlockArgs init(mlir::MLIRContext *ctx, BackedgeBuilder &bb,
                          Block *b) {
      BlockArgs args;
      args.control = bb.get(dc::TokenType::get(ctx));
      for (Type argT : b->getArgumentTypes())
        args.args.push_back(bb.get(dc::ValueType::get(ctx, argT)));
      return args;
    }
  };
  llvm::DenseMap<ControlEdge, BlockArgs> blockArgs;

  // Maintain a set of backedges for merge block select values. These are the
  // values which eventually determine which input a merge block (i.e. a block
  // with 2 predecessors) will use to select between incoming control edges.
  llvm::DenseMap<Block *, Backedge> mergeBlockSelects;

  // The set of blocks which have been converted.
  llvm::DenseSet<Block *> convertedBlocks;

public:
  ControlFlowConverter(OpBuilder &b, ValueMapper &mapper, Region &region)
      : b(b), mapper(mapper) {

    domInfo = std::make_unique<DominanceInfo>(region);
    loopInfo = std::make_unique<CFGLoopInfo>(domInfo.get());
  }
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

  LogicalResult go() {
    if (failed(prime()))
      return failure();

    if (failed(findBlockPairs(splitMergeBlockPairs)))
      return failure();

    return convert(&region.front());
  }

  // Converts the 'to' block to DC, using the 'from' block as a source of
  // of control. The 'control' token represents the incoming control token
  // that hands off control to the 'to' block from the 'from' block.
  virtual FailureOr<Value> convert(Block *block) = 0;

  // Assigns the backedges of a blocks' block arguments to the given values.
  void setBlockArgs(ControlEdge edge, Value control, ValueRange values) {
    auto &args = blockArgs[edge];
    args.control.setValue(control);
    for (auto [arg, value] : llvm::zip(args.args, values))
      arg.set(value);
  }

  // Assigns the select value backedge of a merge block.
  void setMergeBlockSelect(Block *block, Value select) {
    getMergeBlockSelect(block).set(select);
  }

  // Returns a merge block's select value.
  Backedge getMergeBlockSelect(Block *block) {
    auto it = mergeBlockSelects.find(block);
    if (it != mergeBlockSelects.end())
      return it->second;

    auto selectType = b.get<dc::ValueType>(b.getI1Type);
    return mergeBlockSelects.try_emplace(block, bb.get(selectType)).first;
  }

protected:
  // Primes the conversion by creating backedges for all block arguments and
  // controls.
  void prime();

  LogicalResult findBlockPairs(SplitMergePairs &blockPairs);

  bool formsIrreducibleCF(Block *splitBlock, Block *mergeBlock);
};

void ControlFlowConverter::prime() {
  for (Block &block : region.getBlocks()) {
    for (Block &pred : block.getPredecessors()) {
      blockArgs[{&pred, &block}] = BlockArgs::init(b.getContext(), bb, &block);
    }
  }
}

static bool loopsHaveSingleExit(CFGLoopInfo &loopInfo) {
  for (CFGLoop *loop : loopInfo.getTopLevelLoops())
    if (!loop->getExitBlock())
      return false;
  return true;
}

bool ControlFlowConverter::formsIrreducibleCF(Block *splitBlock,
                                              Block *mergeBlock) {
  CFGLoop *loop = loopInfo->getLoopFor(mergeBlock);
  for (auto *mergePred : mergeBlock->getPredecessors()) {
    // Skip loop predecessors
    if (loop && loop->contains(mergePred))
      continue;

    // A DAG-CFG is irreducible, iff a merge block has a predecessor that can be
    // reached from both successors of a split node, e.g., neither is a
    // dominator.
    // => Their control flow can merge in other places, which makes this
    // irreducible.
    if (llvm::none_of(splitBlock->getSuccessors(), [&](Block *splitSucc) {
          if (splitSucc == mergeBlock || mergePred == splitBlock)
            return true;
          return domInfo.dominates(splitSucc, mergePred);
        }))
      return true;
  }
  return false;
}

static Operation *findBranchToBlock(Block *block) {
  Block *pred = *block->getPredecessors().begin();
  return pred->getTerminator();
}

LogicalResult
ControlFlowConverter::findBlockPairs(SplitMergePairs &blockPairs) {
  // assumes that merge block insertion happended beforehand
  // Thus, for each split block, there exists one merge block which is the
  // post dominator of the child nodes.
  Operation *parentOp = region.getParentOp();

  // Assumes that each loop has only one exit block. Such an error should
  // already be reported by the loop rewriting.
  assert(loopsHaveSingleExit(*loopInfo) &&
         "expected loop to only have one exit block.");

  for (Block &block : r) {
    if (block.getNumSuccessors() < 2)
      continue;

    // Loop headers cannot be merge blocks.
    if (loopInfo.getLoopFor(&b))
      continue;

    assert(block.getNumSuccessors() == 2);
    Block *succ0 = block.getSuccessor(0);
    Block *succ1 = block.getSuccessor(1);

    if (succ0 == succ1)
      continue;

    Block *mergeBlock = postDomInfo.findNearestCommonDominator(succ0, succ1);

    // Precondition checks
    if (formsIrreducibleCF(&b, mergeBlock)) {
      return parentOp->emitError("expected only reducible control flow.")
                 .attachNote(findBranchToBlock(mergeBlock)->getLoc())
             << "This branch is involved in the irreducible control flow";
    }

    unsigned nonLoopPreds = 0;
    CFGLoop *loop = loopInfo.getLoopFor(mergeBlock);
    for (auto *pred : mergeBlock->getPredecessors()) {
      if (loop && loop->contains(pred))
        continue;
      nonLoopPreds++;
    }
    if (nonLoopPreds > 2)
      return parentOp
                 ->emitError("expected a merge block to have two predecessors. "
                             "Did you run the merge block insertion pass?")
                 .attachNote(findBranchToBlock(mergeBlock)->getLoc())
             << "This branch jumps to the illegal block";

    blockPairs[&block] = mergeBlock;
  }

  return success();
}

ControlFlowConversionPatternBase::ControlFlowConversionPatternBase(
    ControlFlowConverter &converter)
    : converter(converter), b(converter.b), dcb(converter.b),
      mapper(converter.mapper) {}

class TestControlFlowConverter : public ControlFlowConverter {
public:
  using ControlFlowConverter::ControlFlowConverter;

  FailureOr<Value> convert(mlir::Block *from, Block *to,
                           Value control) override;

private:
  // Returns the inputs to a block, which represents the control flowing into
  // a block as well as the values that
  Value getBlockInputs(mlir::Block *block);
};

LogicalResult TestControlFlowConverter::convert(Block *block) {
  if (convertedBlocks.contains(block))
    return;

  DCBuilder dcb(b);

  auto parentOp = block->getParentOp();
  size_t nPredecessors = block->getNumPredecessors();
  BlockArgs blockArgs;
  if (nPredecessors == 0) {
    // Entry block conversion. All entry args have been converted to DC, so
    // join them, and register the value mappings on the unpacked ops.
    llvm::SmallVector<Value> argTokens;
    for (auto [orig, conv] : llvm::zip(
             block.getArguments(), dstRegion.getBodyBlock()->getArguments())) {
      auto [argToken, argValue] = dcb.unpack(conv);
      blockArgs.args.push_back(argValue);
      argTokens.push_back(argToken);
    }
    blockArgs.control = join(b, argTokens);
  } else if (nPredecessors == 1) {
    // Convert single predecessor blocks. Simple case, since we're not muxing
    // values from different predecessor blocks.
    blockArgs = blockArgs.lookup({block->getSinglePredecessor(), block});
  } else if (nPredecessors == 2) {
    // Convert dual predecessor blocks - this can either be loop headers or
    // merge blocks. We need to determine where to get data from.
    // We get the block arguments by dc.select'ing between the two predecessor
    // values using the block select token.
    Value selectionToken = getMergeBlockSelect(block);

    assert(false && "Build LPR network");

    auto firstBlockArgs = blockArgs.lookup({block->getPredecessor(0), block});
    auto secondBlockArgs = blockArgs.lookup({block->getPredecessor(1), block});

    Value selectedControl =
        b.create<dc::SelectOp>(block->getLoc(), selectionToken,
                               firstBlockArgs.control, secondBlockArgs.control);

    llvm::SmallVector<Value> argTokens;
    for (auto [first, second, orig] : llvm::zip(
             firstBlockArgs.args, secondBlockArgs.args, block.getArguments())) {
      auto [selectedArgToken, selectedArgValue] =
          dcb.unpack(dcb.mux(selectionToken, first, second));
      blockArgs.args.push_back(selectedArgValue);
      argTokens.push_back(selectedArgToken);
    }
    argTokens.push_back(selectedControl);
    blockArgs.control = join(b, argTokens);
  } else {
    // >2 predecessor blocks are not supported, and should not happen after
    // merge block insertion has been performed.
    return parentOp->emitOpError() << "Block has more than two predecessors";
  }

  // Map the converted block arguments.
  assert(block->getNumArguments() == blockArgs.args.size() &&
         "Block argument count mismatch");
  for (auto [orig, conv] : llvm::zip(block.getArguments(), blockArgs.args))
    mapper.set(orig, conv);

  for (auto &op : llvm::make_early_inc_range(*block)) {
    auto it = converterLookup.find(op.getName().getIdentifier());
    if (it == converterLookup.end()) {
      // If no converter was registered for the op, it is assumed to not be a
      // control-flow op. Copy it to the destination region and map the
      // operands and results.
      Operation *clonedOp = b.clone(op);
      for (auto [i, origOperand] : llvm::enumerate(op.getOperands()))
        clonedOp->setOperand(i, mapper.get(origOperand));

      for (auto [i, origResult] : llvm::enumerate(op.getResults()))
        mapper.set(origResult, clonedOp->getResult(i));
      continue;
    }

    // This was a control-flow op, so we need to dispatch to the appropriate
    // converter.
    auto &converter = it->second;
    auto res = converter->dispatch(control, &op);
    if (failed(res))
      return failure();

    // The result of the conversion is the new current control token.
    control = *res;
  }

  return success();
} // namespace dc

class CondBrConversionPattern
    : public ControlFlowConversionPattern<cf::CondBrOp> {
public:
  using ControlFlowConversionPattern::ControlFlowConversionPattern;
  FailureOr<Value> convert(mlir::Value control, cf::CondBrOp op) override {
    // Pack the incoming control token with the condition value.
    Value mappedCond = mapper.get(op.getCondition());
    auto dcCond = dbc.pack(control, mappedCond).getResult();

    // Branch on the packed condition.
    auto branchOp = b.create<dc::BranchOp>(op.getLoc(), dcCond);

    // Assign the block arguments to the true and false destinations.
    Block *thisBlock = op->getBlock();
    llvm::SmallVector<Value> trueArgs, falseArgs;
    for (auto trueOperand : op.getTrueDestOperands()) {
      trueArgs.push_back(
          dcb.pack(branchOp.getTrueToken(), mapper.get(trueOperand))
              .getResult());
    }
    for (auto falseOperand : op.getFalseDestOperands()) {
      falseArgs.push_back(
          dcb.pack(branchOp.getFalseToken(), mapper.get(falseOperand))
              .getResult());
    }
    converter.setBlockArgs({thisBlock, op.getTrueDest()}, trueArgs);
    converter.setBlockArgs({thisBlock, op.getFalseDest()}, falseArgs);

    // Recurse into the true and false blocks.
    auto trueBranchRes =
        converter.convert(branchOp.getTrueToken(), op.getTrueDest());
    auto falseBranchRes =
        converter.convert(branchOp.getFalseToken(), op.getFalseDest());

    // This is a conditional branch meaning that it resides in a split block. We
    // must assign the condition value to the merge block select backedge, based
    // on the block pair info.
    auto infoIt = converter.splitMergeBlockPairs.find(thisBlock);
    assert(infoIt != converter.splitMergeBlockPairs.end() &&
           "Conditional branch not in a split block");
    converter.setMergeBlockSelect(infoIt->second, mappedCond);

    // Nothing to feed back - control was sunk into the true and false blocks.
    return Value();
  }
};

class BrConversionPattern : public ControlFlowConversionPattern<cf::BrOp> {
public:
  using ControlFlowConversionPattern::ControlFlowConversionPattern;
  FailureOr<Value> convert(mlir::Value control, cf::BrOp op) override {
    // Assign the block arguments to the destination.
    Block *thisBlock = op->getBlock();
    Block *dstBlock = op.getDest();
    llvm::SmallVector<Value> args;
    for (auto operand : op.getDestOperands())
      args.push_back(dcb.pack(control, mapper.get(operand)).getResult());

    converter.setBlockArgs({thisBlock, dstBlock}, args);

    // Recurse into the true and false blocks.
    if (failed(converter.convert(control, dstBlock)))
      return failure();

    // Nothing to feed back - control was sunk into the true and false blocks.
    return Value();
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
      packedReturns.push_back(b.pack(control, operand));

    // Create a hw.output op at the current insertion point.
    b.create<hw::OutputOp>(op.getLoc(), packedReturns);

    // Nothing to return - control was sunk into the output op.
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
struct DCTestCFToDCPass : public DCTestCFToDCBase<DCTestCFToDCPass> {
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();

    OpBuilder b(mod.getContext());
    for (auto func : llvm::make_early_inc_range(mod.getOps<func::FuncOp>())) {
      // Create the arg-converted, hw-module converted top-level op.
      auto convFunc = createConvertedFunc(b, func);
      b.setInsertionPointToStart(convFunc.getBodyBlock());

      // To prime the conversion, we need the incoming control token. Define
      // this as the join of all incoming control tokens.
      Value controlIn = join(b, argTokens);

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
      converter.add<ReturnOpConversionPattern>();
      converter.add<CondBrConversionPattern>();
      converter.add<BrConversionPattern>();

      if (failed(
              converter.convert(controlIn, func.getBody().getBlocks().front())))
        return signalPassFailure();

      // Remove the original function.
      func.erase();
    }
  };
};

} // namespace

std::unique_ptr<mlir::Pass> circt::dc::createDCTestCFToDCPass() {
  return std::make_unique<DCTestCFToDCPass>();
}
