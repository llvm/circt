//===- SCFToCalyx.cpp - SCF to Calyx pass entry point -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SCF to Calyx conversion pass implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SCFToCalyx/SCFToCalyx.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include <variant>

using namespace llvm;
using namespace mlir;

namespace circt {

//===----------------------------------------------------------------------===//
// Utility types
//===----------------------------------------------------------------------===//

/// A mapping is maintained between a function operation and its corresponding
/// Calyx component. This facilitates translation when function and Calyx name
/// are not identical, such as when the top-level function is renamed to
/// 'main' to comply with Calyx conventions.
using FuncMapping = DenseMap<FuncOp, calyx::ComponentOp>;

struct WhileScheduleable {
  // While operation to schedule.
  scf::WhileOp whileOp;
  // The group to schedule before executing the while group, to set the
  // initial values of the init_args.
  calyx::GroupOp initGroup;
};

// A variant of types representing scheduleable operations.
using Scheduleable = std::variant<calyx::GroupOp, WhileScheduleable>;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Tries to match a constant value defined by op. If the match was
/// successful, returns true and binds the constant to 'value'.
static bool matchConstantOp(Operation *op, APInt &value) {
  return mlir::detail::constant_int_op_binder(&value).match(op);
}

/// Returns true if there exists only a single memref::LoadOp which loads from
/// the memory referenced by loadOp.
static bool singleLoadFromMemory(memref::LoadOp loadOp) {
  return llvm::count_if(loadOp.memref().getUses(), [](auto &user) {
           return dyn_cast<memref::LoadOp>(user.getOwner());
         }) <= 1;
}

/// Creates a DictionaryAttr containing a unit attribute 'name'. Used for
/// defining mandatory port attributes for calyx::ComponentOp's.
static DictionaryAttr getMandatoryPortAttr(MLIRContext *ctx, StringRef name) {
  return DictionaryAttr::get(
      ctx, {NamedAttribute(Identifier::get(name, ctx), UnitAttr::get(ctx))});
}

/// Adds the mandatory Calyx component I/O ports (->[clk, reset, go], [done]->)
/// to ports
static void addMandatoryComponentPorts(PatternRewriter &rewriter,
                                       SmallVector<calyx::PortInfo> &ports) {
  MLIRContext *ctx = rewriter.getContext();
  ports.push_back({.name = rewriter.getStringAttr("clk"),
                   .type = rewriter.getI1Type(),
                   .direction = calyx::Direction::Input,
                   .attributes = getMandatoryPortAttr(ctx, "clk")});
  ports.push_back({.name = rewriter.getStringAttr("reset"),
                   .type = rewriter.getI1Type(),
                   .direction = calyx::Direction::Input,
                   .attributes = getMandatoryPortAttr(ctx, "reset")});
  ports.push_back({.name = rewriter.getStringAttr("go"),
                   .type = rewriter.getI1Type(),
                   .direction = calyx::Direction::Input,
                   .attributes = getMandatoryPortAttr(ctx, "go")});
  ports.push_back({.name = rewriter.getStringAttr("done"),
                   .type = rewriter.getI1Type(),
                   .direction = calyx::Direction::Output,
                   .attributes = getMandatoryPortAttr(ctx, "done")});
}

/// Creates a new group within compOp.
template <typename TGroup, typename TRet = TGroup>
static TRet createGroup(PatternRewriter &rewriter, calyx::ComponentOp compOp,
                        Location loc, Twine uniqueName) {

  IRRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(compOp.getWiresOp().getBody());
  auto groupOp = rewriter.create<TGroup>(loc, uniqueName.str());
  rewriter.createBlock(&groupOp.getBodyRegion());
  return groupOp;
}

/// ComponentLoweringState handles the current state of lowering of a Calyx
/// component. It is mainly used as a key/value store for recording information
/// during partial lowering, which is required at later lowering passes.
class ProgramLoweringState;
class ComponentLoweringState {
public:
  ComponentLoweringState(ProgramLoweringState &pls, calyx::ComponentOp compOp)
      : programLoweringState(pls), compOp(compOp) {}

  ProgramLoweringState &getProgramState() { return programLoweringState; }

  /// Returns a unique name within compOp with the provided prefix.
  std::string getUniqueName(StringRef prefix) {
    std::string prefixStr = prefix.str();
    unsigned idx = prefixIdMap[prefixStr];
    prefixIdMap[prefixStr]++;
    return (prefix + "_" + std::to_string(idx)).str();
  }

  /// Returns a unique name associated with a specific operation.
  StringRef getUniqueName(Operation *op) {
    auto it = opNames.find(op);
    assert(it != opNames.end() && "A unique name should have been set for op");
    return it->second;
  }

  /// Registers a unique name for a given operation using a provided prefix.
  void setUniqueName(Operation *op, StringRef prefix) {
    auto it = opNames.find(op);
    assert(it == opNames.end() && "A unique name was already set for op");
    opNames[op] = getUniqueName(prefix);
  }

  /// Returns the calyx::ComponentOp associated with this lowering state.
  calyx::ComponentOp getCompOp() { return compOp; }

  /// Returns a unique name for instantiating a new instance of 'comp' within
  /// this component.
  std::string genNewInstanceName(calyx::ComponentOp comp) {
    int cnt =
        llvm::count_if(comp.getOps<calyx::InstanceOp>(), [&](auto instanceOp) {
          return instanceOp.getReferencedComponent() == comp;
        });
    return comp.getName().str() + "_inst_" + std::to_string(cnt);
  }

  /// Returns an ordered list of schedulables which registered themselves to be
  /// a result of lowering the block in the source program. The list order
  /// follows def-use chains between the scheduleables in the block.
  SmallVector<Scheduleable> getBlockScheduleables(mlir::Block *block) {
    auto it = blockScheduleables.find(block);
    if (it != blockScheduleables.end())
      return it->second;
    /// In cases of a block resulting in purely combinational logic, no
    /// scheduleables registered themselves with the block.
    return {};
  }

  /// Register 'scheduleable' as being generated through lowering 'block'.
  ///
  /// @todo: Add a post-insertion check to ensure that the use-def ordering
  /// invariant holds for the groups. When the control schedule is generated,
  /// scheduleables within a block are emitted sequentially based on the order
  /// that this function was called during conversion.
  ///
  /// Currently, we assume this to be always true since walking the IR implies
  /// sequentially iterate over operations within blocks - which translates to
  /// this function being called in the correct order.
  void addBlockScheduleable(mlir::Block *block,
                            const Scheduleable &scheduleable) {
    blockScheduleables[block].push_back(scheduleable);
  }

  /// Register value v as being evaluated when scheduling group.
  void registerEvaluatingGroup(Value v, calyx::GroupInterface group) {
    valueGroupAssigns[v] = group;
  }

  /// Return the group which evaluates the value v. Optionally, caller may
  /// specify the expected type of the group.
  template <typename TGroupOp = calyx::GroupInterface>
  TGroupOp getEvaluatingGroup(Value v) {
    auto it = valueGroupAssigns.find(v);
    assert(it != valueGroupAssigns.end() && "No group evaluating value!");
    if constexpr (std::is_same<TGroupOp, calyx::GroupInterface>::value)
      return it->second;
    else {
      auto group = dyn_cast<TGroupOp>(it->second.getOperation());
      assert(group && "Actual group type differed from expected group type");
      return group;
    }
  }

  /// Register 'grp' as a group which performs block argument
  /// register transfer when transitioning from basic block from to to.
  void addBlockArgGroup(Block *from, Block *to, calyx::GroupOp grp) {
    blockArgGroups[from][to].push_back(grp);
  }

  /// Returns a list of groups to be evaluated to perform the block argument
  /// register assignments when transitioning from basic block 'from' to 'to'.
  ArrayRef<calyx::GroupOp> getBlockArgGroups(Block *from, Block *to) {
    return blockArgGroups[from][to];
  }

  /// Register reg as being the idx'th argument register for block.
  void addBlockArgReg(Block *block, calyx::RegisterOp reg, unsigned idx) {
    assert(blockArgRegs[block].count(idx) == 0);
    assert(idx < block->getArguments().size());
    blockArgRegs[block][idx] = reg;
  }

  /// Return a mapping of block argument indices to block argument registers.
  const DenseMap<unsigned, calyx::RegisterOp> &getBlockArgRegs(Block *block) {
    return blockArgRegs[block];
  }

  /// Register reg as being the idx'th return value register.
  void addReturnReg(calyx::RegisterOp reg, unsigned idx) {
    assert(returnRegs.count(idx) == 0);
    returnRegs[idx] = reg;
  }

  /// Returns the idx'th return value registers.
  calyx::RegisterOp getReturnReg(unsigned idx) {
    assert(returnRegs.count(idx) && "No register registered for index!");
    return returnRegs[idx];
  }

  /// Register reg as being the idx'th iter_args register for 'whileOp'.
  void addWhileIterReg(scf::WhileOp whileOp, calyx::RegisterOp reg,
                       unsigned idx) {
    assert(whileIterRegs[whileOp].count(idx) == 0 &&
           "A register was already registered for the given while iter_arg "
           "index");
    assert(idx < whileOp.getAfterArguments().size());
    whileIterRegs[whileOp][idx] = reg;
  }

  /// Return a mapping of block argument indices to block argument registers.
  const DenseMap<unsigned, calyx::RegisterOp> &
  getWhileIterRegs(scf::WhileOp whileOp) {
    return whileIterRegs[whileOp];
  }

  void setWhileLatchGroup(scf::WhileOp whileOp, calyx::GroupOp grp) {
    assert(whileLatchGroups.count(whileOp) == 0);
    whileLatchGroups[whileOp] = grp;
  }

  calyx::GroupOp getWhileLatchGroup(scf::WhileOp whileOp) {
    return whileLatchGroups[whileOp];
  }
  /// Returns an SSA value for an arbitrary precision constant defined within
  /// compOp. A new constant is created if no constant is found.
  Value getConstant(PatternRewriter &rewriter, Location loc, int64_t value,
                    unsigned width) {
    IRRewriter::InsertionGuard guard(rewriter);
    Value v;
    auto it = constants.find(APInt(value, width));
    if (it == constants.end()) {
      rewriter.setInsertionPointToStart(compOp.getBody());
      return v = rewriter.create<hw::ConstantOp>(loc, APInt(value, width));
    }
    return it->second;
  }

  /// Registers a calyx::MemoryOp as being associated with a memory identified
  /// by 'memref'.
  void registerMemory(Value memref, calyx::MemoryOp memoryOp) {
    assert(memref.getType().isa<MemRefType>());
    assert(memories.find(memref) == memories.end() &&
           "Memory already registered for memref");
    memories[memref] = memoryOp;
  }

  /// Returns a calyx::MemoryOp registered for the given memref.
  calyx::MemoryOp getMemory(Value memref) {
    assert(memref.getType().isa<MemRefType>());
    auto it = memories.find(memref);
    assert(it != memories.end() && "No memory registered for memref");
    return it->second;
  }

  template <typename TLibraryOp>
  TLibraryOp getNewLibraryOpInstance(PatternRewriter &rewriter, Location loc,
                                     TypeRange resTypes) {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(compOp.getBody(), compOp.getBody()->begin());
    auto name = TLibraryOp::getOperationName().split(".").second;
    auto uniqueName = getUniqueName(name);
    return rewriter.create<TLibraryOp>(loc, rewriter.getStringAttr(uniqueName),
                                       resTypes);
  }

private:
  /// A reference to the parent program lowering state.
  ProgramLoweringState &programLoweringState;

  /// The component which this lowering state is associated to.
  calyx::ComponentOp compOp;

  /// A mapping from blocks to block argument registers.
  DenseMap<Block *, DenseMap<unsigned, calyx::RegisterOp>> blockArgRegs;

  /// A mapping from return value indexes to return value registers.
  DenseMap<unsigned, calyx::RegisterOp> returnRegs;

  /// A mapping from while ops to iteration argument registers.
  DenseMap<Operation *, DenseMap<unsigned, calyx::RegisterOp>> whileIterRegs;

  /// A mapping of string prefixes and the current uniqueness counter for that
  /// prefix. Used to generate unique names.
  std::map<std::string, unsigned> prefixIdMap;

  /// A mapping from Operations and previously assigned unique name of the op.
  std::map<Operation *, std::string> opNames;

  /// A mapping of currently available constants in this component.
  DenseMap<APInt, Value> constants;

  /// A mapping between SSA values and the groups which assign them.
  DenseMap<Value, calyx::GroupInterface> valueGroupAssigns;

  /// BlockScheduleables is a list of scheduleables that should be
  /// sequentially executed when executing the associated basic block.
  DenseMap<mlir::Block *, SmallVector<Scheduleable>> blockScheduleables;

  /// Block arg groups is a list of groups that should be sequentially
  /// executed when passing control from the source to destination block.
  /// Block arg groups are executed before blockScheduleables (akin to a
  /// phi-node).
  DenseMap<Block *, DenseMap<Block *, SmallVector<calyx::GroupOp>>>
      blockArgGroups;

  /// While latch groups is a group that should be sequentially executed when
  /// finishing a while loop body. The execution of this group will place the
  /// yield'ed loop body values in the iteration argument registers.
  DenseMap<Operation *, calyx::GroupOp> whileLatchGroups;

  /// A mapping from memref's to their corresponding calyx memory op.
  DenseMap<Value, calyx::MemoryOp> memories;
};

/// ProgramLoweringState handles the current state of lowering of a Calyx
/// program. It is mainly used as a key/value store for recording information
/// during partial lowering, which is required at later lowering passes.
class ProgramLoweringState {
public:
  explicit ProgramLoweringState(calyx::ProgramOp program,
                                StringRef topLevelFunction)
      : m_topLevelFunction(topLevelFunction), program(program) {
    getProgram();
  }

  /// Returns a meaningful name for a value within the program scope.
  template <typename ValueOrBlock>
  std::string irName(ValueOrBlock &v) {
    std::string s;
    llvm::raw_string_ostream os(s);
    AsmState asmState(program);
    v.printAsOperand(os, asmState);
    return s;
  }

  /// Returns a meaningful name for a block within the program scope (removes
  /// the ^ prefix from block names).
  std::string blockName(Block *b) {
    auto blockName = irName(*b);
    blockName.erase(std::remove(blockName.begin(), blockName.end(), '^'),
                    blockName.end());
    return blockName;
  }

  /// Returns the component lowering state associated with compOp.
  ComponentLoweringState &compLoweringState(calyx::ComponentOp compOp) {
    auto it = compStates.find(compOp);
    if (it != compStates.end())
      return it->second;

    /// Create a new ComponentLoweringState for the compOp.
    auto newCompStateIt = compStates.try_emplace(compOp, *this, compOp);
    return newCompStateIt.first->second;
  }

  /// Returns the current program.
  calyx::ProgramOp getProgram() {
    assert(program.getOperation() != nullptr);
    return program;
  }

  /// Returns the name of the top-level function in the source program.
  StringRef topLevelFunction() const { return m_topLevelFunction; }

private:
  StringRef m_topLevelFunction;
  calyx::ProgramOp program;
  DenseMap<Operation *, ComponentLoweringState> compStates;

  /// Map of the currently available combinational components in the
  /// design. First level represent the combinational op name (i.e.:
  /// comb.add), the second level represents the width of the component
  /// inputs, and the associated component (if any) to that input width.
  DenseMap<OperationName, DenseMap<unsigned, calyx::ComponentOp>> stdComps;
};

/// Creates a new register within the component associated to 'compState'.
static calyx::RegisterOp createReg(ComponentLoweringState &compState,
                                   PatternRewriter &rewriter, Location loc,
                                   Twine prefix, size_t width) {
  IRRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(compState.getCompOp().getBody());
  return rewriter.create<calyx::RegisterOp>(
      loc, rewriter.getStringAttr(prefix + "_reg"), width);
}

/// Creates register assignment operations within the provided groupOp.
static void buildAssignmentsForRegisterWrite(ComponentLoweringState &state,
                                             PatternRewriter &rewriter,
                                             calyx::GroupOp groupOp,
                                             calyx::RegisterOp &reg, Value v) {
  IRRewriter::InsertionGuard guard(rewriter);
  auto loc = v.getLoc();
  rewriter.setInsertionPointToEnd(groupOp.getBody());
  rewriter.create<calyx::AssignOp>(loc, reg.in(), v, Value());
  rewriter.create<calyx::AssignOp>(
      loc, reg.write_en(), state.getConstant(rewriter, loc, 1, 1), Value());
  rewriter.create<calyx::GroupDoneOp>(loc, reg.donePort(), Value());
}

/// Creates a SeqOp containing an inner body block.
static calyx::SeqOp createSeqOp(PatternRewriter &rewriter, Location loc) {
  auto seqOp = rewriter.create<calyx::SeqOp>(loc);
  rewriter.createBlock(&seqOp.getRegion());
  return seqOp;
}

/// Get the index'th output of compOp, which is associated with
/// funcOp.

static Value getComponentOutput(mlir::FuncOp funcOp, calyx::ComponentOp compOp,
                                unsigned index) {
  size_t resIdx = funcOp.getNumArguments() + 3 /*go, reset, clk*/ + index;
  assert(compOp.getNumArguments() > resIdx);
  return compOp.getArgument(resIdx);
}

/// Get the index'th input of compOp, which is associated with
/// funcOp.

static Value getComponentInput(calyx::ComponentOp compOp, unsigned index) {
  return compOp.getArgument(index);
}

static calyx::GroupOp buildWhileIterArgAssignments(
    PatternRewriter &rewriter, ComponentLoweringState &state, Location loc,
    scf::WhileOp whileOp, Twine uniqueSuffix, ValueRange ops) {
  assert(whileOp);
  /// Pass iteration arguments through registers. This follows closely
  /// to what is done for branch ops.
  auto groupName = "assign_" + uniqueSuffix;
  auto groupOp =
      createGroup<calyx::GroupOp>(rewriter, state.getCompOp(), loc, groupName);
  auto iterArgRegs = state.getWhileIterRegs(whileOp);
  // Create register assignment for each iter_arg. a calyx::GroupDone signal is
  // created for each register. This is later cleaned up in
  // GroupDoneCleanupPattern.
  for (auto arg : enumerate(ops)) {
    auto reg = iterArgRegs[arg.index()];
    buildAssignmentsForRegisterWrite(state, rewriter, groupOp, reg,
                                     arg.value());
  }
  return groupOp;
}

//===----------------------------------------------------------------------===//
// Partial lowering infrastructure
//===----------------------------------------------------------------------===//

/// Base class for partial lowering passes. A partial lowering pass
/// modifies the root operation in place, but does not replace the root
/// operation.
/// The RewritePatternType template parameter allows for using both
/// OpRewritePattern (default) or OpInterfaceRewritePattern.
template <class OpType,
          template <class> class RewritePatternType = OpRewritePattern>
class PartialLoweringPattern : public RewritePatternType<OpType> {
public:
  using RewritePatternType<OpType>::RewritePatternType;
  PartialLoweringPattern(MLIRContext *ctx, LogicalResult &resRef)
      : RewritePatternType<OpType>(ctx), partialPatternRes(resRef) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(
        op, [&] { partialPatternRes = partiallyLower(op, rewriter); });
    return partialPatternRes;
  }

  virtual LogicalResult partiallyLower(OpType op,
                                       PatternRewriter &rewriter) const = 0;

private:
  LogicalResult &partialPatternRes;
};

class FuncOpPartialLoweringPattern
    : public PartialLoweringPattern<mlir::FuncOp> {
public:
  FuncOpPartialLoweringPattern(MLIRContext *context, LogicalResult &resRef,
                               FuncMapping &_funcMap, ProgramLoweringState &pls)
      : PartialLoweringPattern(context, resRef), funcMap(_funcMap), pls(pls) {}

  LogicalResult partiallyLower(mlir::FuncOp funcOp,
                               PatternRewriter &rewriter) const override final {
    // Create local state assoaciated with the funcOp that is currently being
    // lowered (due to this pattern instance being reused for all matched
    // funcOps).
    auto it = funcMap.find(funcOp);
    if (it != funcMap.end()) {
      compOp = &it->second;
      compLoweringState = &pls.compLoweringState(*comp());
    } else {
      compOp = nullptr;
      compLoweringState = nullptr;
    }

    return PartiallyLowerFuncToComp(funcOp, rewriter);
  }

  // Returns the component operation associated with the currently executing
  // partial lowering.
  calyx::ComponentOp *comp() const {
    assert(compOp != nullptr);
    return compOp;
  }

  // Returns the component state associated with the currently executing
  // partial lowering.
  ComponentLoweringState &state() const {
    assert(compLoweringState != nullptr);
    return *compLoweringState;
  }

  ProgramLoweringState &progState() const { return pls; }

  /// Partial lowering implementation.
  virtual LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const = 0;

protected:
  FuncMapping &funcMap;

private:
  mutable calyx::ComponentOp *compOp = nullptr;
  mutable ComponentLoweringState *compLoweringState = nullptr;
  ProgramLoweringState &pls;
};

//===----------------------------------------------------------------------===//
// Partial lowering patterns
//===----------------------------------------------------------------------===//

/// Creates a new calyx program with the contents of the source module inlined
/// within.
///
/// A restriction of the current infrastructure is that a top-level 'module'
/// cannot be overwritten (even though this is essentially what is going on
/// when replacing standard::ModuleOp with calyx::ProgramOp). see:
/// https://llvm.discourse.group/t/de-privileging-moduleop-in-translation-apis/3733/26
///
struct ModuleOpConversion : public OpRewritePattern<mlir::ModuleOp> {
  ModuleOpConversion(MLIRContext *context, calyx::ProgramOp *programOpOutput)
      : OpRewritePattern<mlir::ModuleOp>(context),
        programOpOutput(programOpOutput) {
    assert(programOpOutput->getOperation() == nullptr &&
           "this function will set programOpOutput post module conversion");
  }

  LogicalResult matchAndRewrite(mlir::ModuleOp moduleOp,
                                PatternRewriter &rewriter) const override {
    if (programOpOutput->getOperation() != nullptr) {
      moduleOp.emitError() << "Multiple modules not supported";
      return failure();
    }

    rewriter.updateRootInPlace(moduleOp, [&] {
      // Create ProgramOp
      rewriter.setInsertionPointAfter(moduleOp);
      auto programOp = rewriter.create<calyx::ProgramOp>(moduleOp.getLoc());

      // Inline the entire body region inside.
      rewriter.inlineRegionBefore(moduleOp.getBodyRegion(),
                                  programOp.getBodyRegion(),
                                  programOp.getBodyRegion().end());

      // Inlining the body region also removes ^bb0 from the module body
      // region, so recreate that, before finally inserting the programOp
      auto moduleBlock = rewriter.createBlock(&moduleOp.getBodyRegion());
      rewriter.setInsertionPointToStart(moduleBlock);
      rewriter.insert(programOp);
      *programOpOutput = programOp;
    });
    return success();
  }

private:
  calyx::ProgramOp *programOpOutput = nullptr;
};

/// Inlines Calyx ExecuteRegionOp operations within their parent blocks.
/// An execution region op (ERO) is inlined by:
///  i  : add a sink basic block for all yield operations inside the
///       ERO to jump to
///  ii : Rewrite scf.yield calls inside the ERO to branch to the sink block
///  iii: inline the ERO region
/// The parent SCF op of the ERO is invalid after inlining the ERO, since
/// control-flow is explicitly not allowed in SCF ops (apart from ERO).
class InlineExecuteRegionOpPattern
    : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp execOp,
                                PatternRewriter &rewriter) const override {
    // Determine type of "yield" operations inside the ERO
    bool ok = false;
    TypeRange yieldTypes;
    execOp.walk([&](scf::YieldOp yieldOp) {
      yieldTypes = yieldOp.getOperandTypes();
      ok = true;
      return WalkResult::interrupt();
    });
    assert(ok && "ExecuteRegion must contain at least one scf.YieldOp");

    // Create sink basic block and rewrite uses of yield results to sink block
    // arguments.
    rewriter.setInsertionPointAfter(execOp);
    auto *sinkBlock = rewriter.splitBlock(
        execOp->getBlock(),
        execOp.getOperation()->getIterator()->getNextNode()->getIterator());
    sinkBlock->addArguments(yieldTypes);
    for (auto res : enumerate(execOp.getResults()))
      res.value().replaceAllUsesWith(sinkBlock->getArgument(res.index()));

    // Rewrite yield calls as branches
    for (auto yieldOp :
         make_early_inc_range(execOp.getRegion().getOps<scf::YieldOp>())) {
      rewriter.setInsertionPointAfter(yieldOp);
      rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, sinkBlock,
                                            yieldOp.getOperands());
    }

    // Inline the regionOp
    auto *preBlock = execOp->getBlock();
    auto *execOpEntryBlock = &execOp.getRegion().front();
    auto *postBlock = execOp->getBlock()->splitBlock(execOp);
    rewriter.inlineRegionBefore(execOp.getRegion(), postBlock);
    rewriter.mergeBlocks(postBlock, preBlock);
    rewriter.eraseOp(execOp);

    // Finally, erase the unused entry block of the execOp region
    rewriter.mergeBlocks(execOpEntryBlock, preBlock);

    return success();
  }
};

class InlineCombGroups
    : public PartialLoweringPattern<calyx::GroupInterface,
                                    OpInterfaceRewritePattern> {
public:
  InlineCombGroups(MLIRContext *context, LogicalResult &resRef,
                   ProgramLoweringState &pls)
      : PartialLoweringPattern(context, resRef), pls(pls) {}

  LogicalResult partiallyLower(calyx::GroupInterface groupIF,
                               PatternRewriter &rewriter) const override {
    auto &state =
        pls.compLoweringState(groupIF->getParentOfType<calyx::ComponentOp>());

    /// Maintain a set of the groups which we've inlined so far. The group
    /// itself is implicitly inlined.
    llvm::SmallSetVector<Operation *, 8> inlinedGroups;
    inlinedGroups.insert(groupIF);

    std::function<void(calyx::GroupInterface, bool)> recurseInline =
        [&](calyx::GroupInterface recGroupOp, bool init) {
          inlinedGroups.insert(recGroupOp);
          for (auto assignOp :
               recGroupOp.getBody()->getOps<calyx::AssignOp>()) {
            if (!init) {
              /// Inline the assignment into the stateful group.
              auto clonedAssignOp = rewriter.clone(*assignOp.getOperation());
              clonedAssignOp->moveBefore(groupIF.getBody(),
                                         groupIF.getBody()->end());
            }
            auto src = assignOp.src();
            auto srcDefOp = src.getDefiningOp();

            /// Things which stop recursive inlining (or in other words, what
            /// breaks combinational paths).
            /// - Component inputs
            /// - Register and memory reads
            /// - Constant ops
            /// - While return values (these are registers, however, while
            ///   return values have at the current point of conversion not yet
            ///   been rewritten to their register outputs, see comment in
            ///   LateSSAReplacement)
            if (src.isa<BlockArgument>() ||
                isa<calyx::RegisterOp, calyx::MemoryOp, hw::ConstantOp,
                    ConstantOp, scf::WhileOp>(srcDefOp))
              continue;

            auto srcCombGroup =
                state.getEvaluatingGroup<calyx::CombGroupOp>(src);
            assert(srcCombGroup && "expected combinational group");
            if (inlinedGroups.count(srcCombGroup))
              continue;

            recurseInline(srcCombGroup, false);
          }
        };
    recurseInline(groupIF, true);

    return success();
  }

private:
  ProgramLoweringState &pls;
};

/// Iterate through the operations of a source function and instantiate cell
/// instances based on the type of the operations.
class BuildOpGroups : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// We walk the operations instead of blocks to ensure that all def's have
    /// been visited before their uses.
    bool res = true;
    funcOp.walk([&](Operation *_op) {
      res &=
          TypeSwitch<mlir::Operation *, LogicalResult>(_op)
              .template Case<BranchOpInterface, ConstantOp, ReturnOp,
                             /// SCF
                             scf::YieldOp,
                             /// memref
                             memref::AllocOp, memref::LoadOp, memref::StoreOp,
                             /// standard arithmetic
                             AddIOp, SubIOp, CmpIOp, ShiftLeftOp,
                             UnsignedShiftRightOp, SignedShiftRightOp, AndOp,
                             XOrOp, OrOp, ZeroExtendIOp, TruncateIOp,
                             IndexCastOp>(
                  [&](auto op) { return buildOp(rewriter, op); })
              .template Case<scf::WhileOp, mlir::FuncOp, scf::ConditionOp>(
                  [&](auto) {
                    // Skip; these special cases are handled separately
                    return success();
                  })
              .Default([&](auto op) {
                // Using op->emitError instead of the assert to have a pretty
                // error message.
                op->emitError() << "Unhandled operation during BuildOpGroups()";
                assert(false);
                return failure();
              })
              .succeeded();

      if (res)
        return WalkResult::advance();
      return WalkResult::interrupt();
    });

    return success(res);
  }

private:
  /// Op builder specializations
  LogicalResult buildOp(PatternRewriter &rewriter, memref::LoadOp loadOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        memref::StoreOp storeOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        memref::AllocOp allocOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, scf::YieldOp yieldOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ConstantOp constOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        BranchOpInterface brOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AddIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SubIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        UnsignedShiftRightOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SignedShiftRightOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShiftLeftOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AndOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, OrOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, XOrOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CmpIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, TruncateIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ZeroExtendIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, IndexCastOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ReturnOp op) const;

  /// Helper functions
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibOp(PatternRewriter &rewriter, TSrcOp op,
                           TypeRange srcTypes, TypeRange dstTypes) const {
    SmallVector<Type> types;
    llvm::append_range(types, srcTypes);
    llvm::append_range(types, dstTypes);

    auto calyxOp = state().getNewLibraryOpInstance<TCalyxLibOp>(
        rewriter, op.getLoc(), types);

    auto directions = calyxOp.portDirections();
    SmallVector<Value, 4> opInputPorts;
    SmallVector<Value, 4> opOutputPorts;
    for (auto dir : enumerate(directions)) {
      if (dir.value() == calyx::Direction::Input)
        opInputPorts.push_back(calyxOp.getResult(dir.index()));
      else
        opOutputPorts.push_back(calyxOp.getResult(dir.index()));
    }
    /// @todo: modify the rest of this function when some library operators are
    /// not combinational (done, go, clk, reset ports)
    assert(opInputPorts.size() == op->getNumOperands() &&
           opOutputPorts.size() == op->getNumResults());

    /// Create assignments to each of the inputs of the operator
    auto group = createGroupForOp<TGroupOp>(rewriter, op);
    rewriter.setInsertionPointToEnd(group.getBody());
    for (auto dstOp : enumerate(opInputPorts))
      rewriter.create<calyx::AssignOp>(op.getLoc(), dstOp.value(),
                                       op->getOperand(dstOp.index()), Value());

    /// Replace SSA result values of the source operator with the new operator
    for (auto res : enumerate(opOutputPorts)) {
      state().registerEvaluatingGroup(res.value(), group);
      op->getResult(res.index()).replaceAllUsesWith(res.value());
    }
    return success();
  }

  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibOp(PatternRewriter &rewriter, TSrcOp op) const {
    return buildLibOp<TGroupOp, TCalyxLibOp, TSrcOp>(
        rewriter, op, op.getOperandTypes(), op->getResultTypes());
  }

  void assignAddressPorts(PatternRewriter &rewriter, Location loc,
                          calyx::GroupInterface group, calyx::MemoryOp memoryOp,
                          Operation::operand_range indices) const {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(group.getBody());
    auto addrPorts = memoryOp.addrPorts();
    for (auto &idx : enumerate(indices)) {
      rewriter.create<calyx::AssignOp>(loc, addrPorts[idx.index()], idx.value(),
                                       Value());
    }
  }

  template <typename TGroupOp>
  TGroupOp createGroupForOp(PatternRewriter &rewriter, Operation *op) const {
    Block *block = op->getBlock();
    auto groupName =
        state().getUniqueName(state().getProgramState().blockName(block));
    return createGroup<TGroupOp>(rewriter, state().getCompOp(),
                                 block->front().getLoc(), groupName);
  }
};

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::LoadOp loadOp) const {
  auto memoryOp = state().getMemory(loadOp.memref());
  if (singleLoadFromMemory(loadOp)) {
    // Single load from memory; Combinational case - we do not have to consider
    // adding registers in front of the memory.

    auto combGroup = createGroupForOp<calyx::CombGroupOp>(rewriter, loadOp);
    assignAddressPorts(rewriter, loadOp.getLoc(), combGroup, memoryOp,
                       loadOp.getIndices());

    /// We refrain from replacing the loadOp result with memoryOp.readData,
    /// since multiple loadOp's need to be converted to a single memory's
    /// ReadData. If this replacement is done now, we lose the link between
    /// which SSA memref::LoadOp values map to which groups for loading a value
    /// from the Calyx memory. At this point of lowering, we keep the
    /// memref::LoadOp SSA value, and do value replacement _after_ control has
    /// been generated (see LateSSAReplacement). This is *vital* for things such
    /// as InlineCombGroups to be able to properly track which memory assignment
    /// groups belong to which accesses.
    state().registerEvaluatingGroup(loadOp.getResult(), combGroup);
  } else {
    auto group = createGroupForOp<calyx::GroupOp>(rewriter, loadOp);
    assignAddressPorts(rewriter, loadOp.getLoc(), group, memoryOp,
                       loadOp.getIndices());

    /// Multiple loads from the same memory; In this case, we _may_ have a
    /// structural hazard in the design we generate. To get around this, we
    /// conservatively place a register in front of each load operation, and
    /// replace all uses of the loaded value with the register output. Proper
    /// handling of this requires the combinational group inliner/scheduler to
    /// be aware of when a combinational expression references multiple loaded
    /// values from the same memory, and then schedule assignments to temporary
    /// registers to get around the structural hazard.
    auto reg = createReg(state(), rewriter, loadOp.getLoc(),
                         state().getUniqueName("load"),
                         loadOp.getMemRefType().getElementTypeBitWidth());
    buildAssignmentsForRegisterWrite(state(), rewriter, group, reg,
                                     memoryOp.readData());
    loadOp.getResult().replaceAllUsesWith(reg.out());
    state().addBlockScheduleable(loadOp->getBlock(), group);
  }

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::StoreOp storeOp) const {
  auto memoryOp = state().getMemory(storeOp.memref());
  auto group = createGroupForOp<calyx::GroupOp>(rewriter, storeOp);

  // This is a sequential group, so register it as being scheduleable for the
  // block
  state().addBlockScheduleable(storeOp->getBlock(),
                               cast<calyx::GroupOp>(group));
  assignAddressPorts(rewriter, storeOp.getLoc(), group, memoryOp,
                     storeOp.getIndices());
  rewriter.setInsertionPointToEnd(group.getBody());
  rewriter.create<calyx::AssignOp>(storeOp.getLoc(), memoryOp.writeData(),
                                   storeOp.getValueToStore(), Value());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryOp.writeEn(),
      state().getConstant(rewriter, storeOp.getLoc(), 1, 1), Value());
  rewriter.create<calyx::GroupDoneOp>(storeOp.getLoc(), memoryOp.done(),
                                      Value());

  return success();
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocOp allocOp) const {
  rewriter.setInsertionPointToStart(comp()->getBody());
  MemRefType memtype = allocOp.getType();
  SmallVector<int64_t> addrSizes;
  SmallVector<int64_t> sizes;
  for (int64_t dim : memtype.getShape()) {
    sizes.push_back(dim);
    addrSizes.push_back(llvm::Log2_64_Ceil(dim));
  }

  auto memoryOp = rewriter.create<calyx::MemoryOp>(
      allocOp.getLoc(), state().getUniqueName("mem"),
      memtype.getElementType().getIntOrFloatBitWidth(), sizes, addrSizes);

  state().registerMemory(allocOp.getResult(), memoryOp);

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::YieldOp yieldOp) const {
  if (yieldOp.getOperands().size() == 0)
    return success();
  auto whileOp = dyn_cast<scf::WhileOp>(yieldOp->getParentOp());
  assert(whileOp);
  yieldOp.getOperands();
  auto assignGroup = buildWhileIterArgAssignments(
      rewriter, state(), yieldOp.getLoc(), whileOp,
      state().getUniqueName(whileOp) + "_latch", yieldOp.getOperands());
  state().setWhileLatchGroup(whileOp, assignGroup);
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ConstantOp constOp) const {
  /// Move constants directly to the compOp body.
  APInt value;
  matchConstantOp(constOp, value);
  auto hwConstOp = rewriter.replaceOpWithNewOp<hw::ConstantOp>(constOp, value);
  hwConstOp->moveAfter(comp()->getBody(), comp()->getBody()->begin());
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     BranchOpInterface brOp) const {
  /// Branch argument passing group creation
  /// Branch operands are passed through registers. In BuildBBRegs we
  /// created registers for all branch arguments of each block. We now
  /// create groups for assigning values to these registers.
  Block *srcBlock = brOp->getBlock();
  for (auto succBlock : enumerate(brOp->getSuccessors())) {
    auto succOperands = brOp.getSuccessorOperands(succBlock.index());
    if (!succOperands.hasValue() || succOperands.getValue().size() == 0)
      continue;
    // Create operand passing group
    std::string groupName = progState().blockName(srcBlock) + "_to_" +
                            progState().blockName(succBlock.value());
    auto groupOp = createGroup<calyx::GroupOp>(rewriter, *comp(), brOp.getLoc(),
                                               groupName);
    // Fetch block argument registers associated with the basic block
    auto dstBlockArgRegs = state().getBlockArgRegs(succBlock.value());
    // Create register assignment for each block argument
    for (auto arg : enumerate(succOperands.getValue())) {
      auto reg = dstBlockArgRegs[arg.index()];
      buildAssignmentsForRegisterWrite(state(), rewriter, groupOp, reg,
                                       arg.value());
    }
    /// Register the group as a block argument group, to be executed
    /// when entering the successor block from this block (srcBlock).
    state().addBlockArgGroup(srcBlock, succBlock.value(), groupOp);
  }
  return success();
}

/// For each return statement, we create a new group for assigning to the
/// previously created return value registers.
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ReturnOp retOp) const {
  if (retOp.getNumOperands() == 0)
    return success();

  std::string groupName = state().getUniqueName("ret_assign");
  Value anyRegDone;
  auto groupOp =
      createGroup<calyx::GroupOp>(rewriter, *comp(), retOp.getLoc(), groupName);
  for (auto op : enumerate(retOp.getOperands())) {
    auto reg = state().getReturnReg(op.index());
    buildAssignmentsForRegisterWrite(state(), rewriter, groupOp, reg,
                                     op.value());
  }
  // Schedule group for execution for when executing the return op block
  state().addBlockScheduleable(retOp->getBlock(), groupOp);
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AddIOp op) const {
  return buildLibOp<calyx::CombGroupOp, calyx::AddLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     SubIOp op) const {
  return buildLibOp<calyx::CombGroupOp, calyx::SubLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     UnsignedShiftRightOp op) const {
  return buildLibOp<calyx::CombGroupOp, calyx::RshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     SignedShiftRightOp op) const {
  return buildLibOp<calyx::CombGroupOp, calyx::SrshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShiftLeftOp op) const {
  return buildLibOp<calyx::CombGroupOp, calyx::LshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AndOp op) const {
  return buildLibOp<calyx::CombGroupOp, calyx::AndLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter, OrOp op) const {
  return buildLibOp<calyx::CombGroupOp, calyx::OrLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     XOrOp op) const {
  return buildLibOp<calyx::CombGroupOp, calyx::XorLibOp>(rewriter, op);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CmpIOp op) const {
  switch (op.predicate()) {
  case CmpIPredicate::eq:
    return buildLibOp<calyx::CombGroupOp, calyx::EqLibOp>(rewriter, op);
  case CmpIPredicate::ne:
    return buildLibOp<calyx::CombGroupOp, calyx::NeqLibOp>(rewriter, op);
  case CmpIPredicate::uge:
    return buildLibOp<calyx::CombGroupOp, calyx::GeLibOp>(rewriter, op);
  case CmpIPredicate::ult:
    return buildLibOp<calyx::CombGroupOp, calyx::LtLibOp>(rewriter, op);
  case CmpIPredicate::ugt:
    return buildLibOp<calyx::CombGroupOp, calyx::GtLibOp>(rewriter, op);
  case CmpIPredicate::ule:
    return buildLibOp<calyx::CombGroupOp, calyx::LeLibOp>(rewriter, op);
  case CmpIPredicate::sge:
    return buildLibOp<calyx::CombGroupOp, calyx::SgeLibOp>(rewriter, op);
  case CmpIPredicate::slt:
    return buildLibOp<calyx::CombGroupOp, calyx::SltLibOp>(rewriter, op);
  case CmpIPredicate::sgt:
    return buildLibOp<calyx::CombGroupOp, calyx::SgtLibOp>(rewriter, op);
  case CmpIPredicate::sle:
    return buildLibOp<calyx::CombGroupOp, calyx::SleLibOp>(rewriter, op);
  default:
    assert(false && "unsupported comparison predicate");
  }
  llvm_unreachable("unsupported comparison predicate");
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     TruncateIOp op) const {
  return buildLibOp<calyx::CombGroupOp, calyx::SliceLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ZeroExtendIOp op) const {
  return buildLibOp<calyx::CombGroupOp, calyx::PadLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     IndexCastOp op) const {
  Type sourceType = op.getOperand().getType();
  sourceType = sourceType.isIndex() ? rewriter.getI32Type() : sourceType;
  Type targetType = op.getResult().getType();
  targetType = targetType.isIndex() ? rewriter.getI32Type() : targetType;
  unsigned targetBits = targetType.getIntOrFloatBitWidth();
  unsigned sourceBits = sourceType.getIntOrFloatBitWidth();
  LogicalResult res = success();

  if (targetBits == sourceBits) {
    // Drop the index cast and replace uses of the target value with the source
    // value
    op.getResult().replaceAllUsesWith(op.getOperand());
  } else {
    // pad/slice the source operand
    if (sourceBits > targetBits)
      res = buildLibOp<calyx::CombGroupOp, calyx::SliceLibOp>(
          rewriter, op, {sourceType}, {targetType});
    else {
      res = buildLibOp<calyx::CombGroupOp, calyx::PadLibOp>(
          rewriter, op, {sourceType}, {targetType});
    }
  }
  rewriter.eraseOp(op);
  return res;
}

class RewriteMemoryAccesses : public PartialLoweringPattern<calyx::AssignOp> {
public:
  RewriteMemoryAccesses(MLIRContext *context, LogicalResult &resRef,
                        ProgramLoweringState &pls)
      : PartialLoweringPattern(context, resRef), pls(pls) {}

  LogicalResult partiallyLower(calyx::AssignOp assignOp,
                               PatternRewriter &rewriter) const override {
    auto dest = assignOp.dest();
    auto destDefOp = dest.getDefiningOp();
    //  Is this an assignment to a memory op?
    if (!destDefOp)
      return success();
    auto destDefMem = dyn_cast<calyx::MemoryOp>(destDefOp);
    if (!destDefMem)
      return success();

    // Is this an assignment to an address port of the memory op?
    bool isAssignToAddrPort = llvm::any_of(
        destDefMem.addrPorts(), [&](auto port) { return port == dest; });

    auto src = assignOp.src();
    auto &state =
        pls.compLoweringState(assignOp->getParentOfType<calyx::ComponentOp>());

    /// Since we don't have any width inference, `index` types are lowered to
    /// some fixed i## width. At the point of access (here), infer a slice
    /// operation to access the memory using the proper width.
    unsigned srcBits = src.getType().getIntOrFloatBitWidth();
    unsigned dstBits = dest.getType().getIntOrFloatBitWidth();
    if (srcBits == dstBits)
      return success();

    if (isAssignToAddrPort) {
      SmallVector<Type> types = {rewriter.getIntegerType(srcBits),
                                 rewriter.getIntegerType(dstBits)};
      auto sliceOp = state.getNewLibraryOpInstance<calyx::SliceLibOp>(
          rewriter, assignOp.getLoc(), types);
      rewriter.setInsertionPoint(assignOp->getBlock(),
                                 assignOp->getBlock()->begin());
      rewriter.create<calyx::AssignOp>(assignOp->getLoc(), sliceOp.getResult(0),
                                       src, Value());
      assignOp.setOperand(1, sliceOp.getResult(1));
    } else
      return assignOp.emitError()
             << "Will only infer slice operators for assign width mismatches "
                "to memory address ports.";

    return success();
  }

private:
  ProgramLoweringState &pls;
};

/// Connverts all index-typed operations and values to i32 values.
class ConvertIndexTypes : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    funcOp.walk([&](Block *block) {
      for (auto arg : block->getArguments())
        if (arg.getType().isIndex())
          arg.setType(rewriter.getI32Type());
    });

    funcOp.walk([&](Operation *op) {
      for (auto res : op->getResults()) {
        if (!res.getType().isIndex())
          continue;

        res.setType(rewriter.getI32Type());
        if (auto constOp = dyn_cast<ConstantOp>(op)) {
          APInt value;
          matchConstantOp(constOp, value);
          rewriter.setInsertionPoint(constOp);
          rewriter.replaceOpWithNewOp<ConstantOp>(
              constOp, rewriter.getI32IntegerAttr(value.getSExtValue()));
        }
      }
    });
    return success();
  }
};

/// Builds registers for each block argument in the program.
class BuildBBRegs : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    funcOp.walk([&](Block *block) {
      /// @todo: Do not register component input values, for now...
      if (block == &block->getParent()->front())
        return;

      for (auto arg : enumerate(block->getArguments())) {
        Type argType = arg.value().getType();
        assert(argType.isa<IntegerType>() && "unsupported block argument type");
        unsigned width = argType.getIntOrFloatBitWidth();
        std::string name =
            progState().blockName(block) + "_arg" + std::to_string(arg.index());
        auto reg =
            createReg(state(), rewriter, arg.value().getLoc(), name, width);
        state().addBlockArgReg(block, reg, arg.index());
        arg.value().replaceAllUsesWith(reg.out());
      }
    });
    return success();
  }
};

/// Builds registers for the return statement of the program and constant
/// assignments to the component return value.
class BuildReturnRegs : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {

    for (auto argType : enumerate(funcOp.getType().getResults())) {
      assert(argType.value().isa<IntegerType>() && "unsupported return type");
      unsigned width = argType.value().getIntOrFloatBitWidth();
      std::string name = "ret_arg" + std::to_string(argType.index());
      auto reg = createReg(state(), rewriter, funcOp.getLoc(), name, width);
      state().addReturnReg(reg, argType.index());

      rewriter.setInsertionPointToStart(
          state().getCompOp().getWiresOp().getBody());
      rewriter.create<calyx::AssignOp>(
          funcOp->getLoc(),
          getComponentOutput(funcOp, *comp(), argType.index()), reg.out(),
          Value());
    }
    return success();
  }
};

/// scf.WhileOp lowering
/// A register is created for each iteration argumenet of the while op. These
/// registers are then written to on the while op terminating yield operation
/// alongside before executing the whileOp in the schedule, to set the initial
/// values of the argument registers.
class BuildWhileGroups : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](scf::WhileOp whileOp) {
      state().setUniqueName(whileOp.getOperation(), "while");

      /// Check for do-while loops; @todo can we support these? for now, do
      /// not support loops where iterargs is changed in the 'before' region.
      /// scf.while also has support for differring iter_args and return args
      /// which we also remove here; iter_args and while return values are
      /// placed in the same registers.
      for (auto barg : enumerate(whileOp.before().front().getArguments())) {
        auto condOp = whileOp.getConditionOp().args()[barg.index()];
        if (barg.value() != condOp) {
          res = whileOp.emitError()
                << progState().irName(barg.value())
                << " != " << progState().irName(condOp)
                << "do-while loops not supported; expected iter-args to "
                   "remain untransformed in the 'before' region of the "
                   "scf.while op.";
          return WalkResult::interrupt();
        }
      }

      /// Create iteration argument registers.
      /// The iteration argument registers will be referenced:
      /// - In the "before" part of the while loop, calculating the conditional,
      /// - In the "after" part of the while loop,
      /// - Outside the while loop, rewriting the while loop return values.
      for (auto arg : enumerate(whileOp.getAfterArguments())) {
        std::string name = state().getUniqueName(whileOp).str() + "_arg" +
                           std::to_string(arg.index());
        auto reg = createReg(state(), rewriter, arg.value().getLoc(), name,
                             arg.value().getType().getIntOrFloatBitWidth());
        state().addWhileIterReg(whileOp, reg, arg.index());
        arg.value().replaceAllUsesWith(reg.out());

        /// Also replace uses in the "before" region of the while loop
        whileOp.before()
            .front()
            .getArgument(arg.index())
            .replaceAllUsesWith(reg.out());
      }

      /// Create iter args initial value assignment group
      auto initGroupOp = buildWhileIterArgAssignments(
          rewriter, state(), whileOp.getLoc(), whileOp,
          state().getUniqueName(whileOp) + "_init", whileOp.getOperands());

      /// Add the while op to the list of scheduleable things in the current
      /// block.
      state().addBlockScheduleable(whileOp->getBlock(),
                                   WhileScheduleable{whileOp, initGroupOp});
      return WalkResult::advance();
    });
    return res;
  }
};

/// buildControl
/// Builds a control schedule by traversing the DFG of the function and
/// associating this with the previously created groups.
/// For simplicity, the generated control flow is expanded for all possible
/// paths in the input DAG. This elaborated control flow is later reduced in
/// the runControlFlowSimplification passes.
class BuildControl : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    auto *entryBlock = &funcOp.getBlocks().front();

    // Build control graph
    rewriter.setInsertionPointToStart(comp()->getControlOp().getBody());
    auto topLevelSeqOp = createSeqOp(rewriter, funcOp.getLoc());
    DenseSet<Block *> path;
    return buildCFGControl(path, rewriter, topLevelSeqOp.getBody(), nullptr,
                           entryBlock);
  }

private:
  /// Sequentially schedules the groups that registered themselves with
  /// 'block'.
  LogicalResult scheduleBasicBlock(PatternRewriter &rewriter,
                                   const DenseSet<Block *> &path,
                                   mlir::Block *parentCtrlBlock,
                                   mlir::Block *block) const {
    auto compblockScheduleables = state().getBlockScheduleables(block);
    auto loc = block->front().getLoc();

    if (compblockScheduleables.size() > 1) {
      auto seqOp = createSeqOp(rewriter, loc);
      parentCtrlBlock = seqOp.getBody();
    }

    rewriter.setInsertionPointToEnd(parentCtrlBlock);

    for (auto &group : compblockScheduleables) {
      if (auto groupPtr = std::get_if<calyx::GroupOp>(&group); groupPtr) {
        rewriter.create<calyx::EnableOp>(loc, groupPtr->sym_name(),
                                         rewriter.getArrayAttr({}));
      } else if (auto whileSchedPtr = std::get_if<WhileScheduleable>(&group);
                 whileSchedPtr) {
        auto &whileOp = whileSchedPtr->whileOp;

        // Insert while iter arg initialization group
        rewriter.create<calyx::EnableOp>(
            loc, whileSchedPtr->initGroup.getName(), rewriter.getArrayAttr({}));

        auto cond = whileOp.getConditionOp().getOperand(0);
        auto condGroup = state().getEvaluatingGroup<calyx::CombGroupOp>(cond);
        auto symbolAttr = FlatSymbolRefAttr::get(
            StringAttr::get(getContext(), condGroup.sym_name()));
        auto whileCtrlOp =
            rewriter.create<calyx::WhileOp>(loc, cond, symbolAttr);
        auto *whileCtrlBlock = rewriter.createBlock(&whileCtrlOp.body(),
                                                    whileCtrlOp.body().begin());
        rewriter.setInsertionPointToEnd(whileCtrlBlock);
        auto whileSeqOp = createSeqOp(rewriter, whileOp.getLoc());

        /// Only schedule the after block. The 'before' block is
        /// implicitly scheduled when evaluating the while condition.
        LogicalResult res =
            buildCFGControl(path, rewriter, whileSeqOp.getBody(), block,
                            &whileOp.after().front());
        // Insert loop-latch at the end of the while group
        rewriter.setInsertionPointToEnd(whileSeqOp.getBody());
        rewriter.create<calyx::EnableOp>(
            loc, state().getWhileLatchGroup(whileOp).getName(),
            rewriter.getArrayAttr({}));
        if (res.failed())
          return res;
      }
    }
    return success();
  }

  /// Schedules a branch target, inserting branch argument assignment blocks
  /// before recursing into the scheduling of the block innards.
  /// Blocks 'from' and 'to' refer to blocks in the source program.
  /// parentCtrlBlock refers to the control block wherein control operations
  /// are to be inserted.
  LogicalResult scheduleBranchTarget(PatternRewriter &rewriter,
                                     const DenseSet<Block *> &path,
                                     Location loc, Block *from, Block *to,
                                     Block *parentCtrlBlock) const {
    // Schedule any registered block arguments to be executed before the body
    // of the branch.
    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    auto preSeqOp = createSeqOp(rewriter, loc);
    rewriter.setInsertionPointToEnd(preSeqOp.getBody());
    for (auto barg : state().getBlockArgGroups(from, to))
      rewriter.create<calyx::EnableOp>(loc, barg.sym_name(),
                                       rewriter.getArrayAttr({}));

    return buildCFGControl(path, rewriter, parentCtrlBlock, from, to);
  }

  LogicalResult buildCFGControl(DenseSet<Block *> path,
                                PatternRewriter &rewriter,
                                mlir::Block *parentCtrlBlock,
                                mlir::Block *preBlock,
                                mlir::Block *block) const {
    if (path.count(block) != 0)
      return preBlock->getTerminator()->emitError()
             << "CFG backedge detected. Loops must be raised to 'scf.while' or "
                "'scf.for' operations.";

    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    if (LogicalResult res =
            scheduleBasicBlock(rewriter, path, parentCtrlBlock, block);
        res.failed())
      return res;
    path.insert(block);

    auto successors = block->getSuccessors();
    auto nSuccessors = successors.size();
    if (nSuccessors > 0) {
      auto brOp = dyn_cast<BranchOpInterface>(block->getTerminator());
      assert(brOp);
      if (nSuccessors > 1) {
        /// @todo: we could choose to support ie. std.switch, but it would
        /// probably be easier to just require it to be lowered beforehand.
        assert(nSuccessors == 2 &&
               "only conditional branches supported for now...");
        // Wrap each branch inside an if/else
        auto cond = brOp->getOperand(0);
        auto condGroup = state().getEvaluatingGroup<calyx::CombGroupOp>(cond);
        auto symbolAttr = FlatSymbolRefAttr::get(
            StringAttr::get(getContext(), condGroup.sym_name()));
        auto ifOp =
            rewriter.create<calyx::IfOp>(brOp->getLoc(), cond, symbolAttr);
        auto *thenCtrlBlock =
            rewriter.createBlock(&ifOp.thenRegion(), ifOp.thenRegion().end());
        auto *elseCtrlBlock =
            rewriter.createBlock(&ifOp.elseRegion(), ifOp.elseRegion().end());
        rewriter.setInsertionPointToEnd(thenCtrlBlock);
        auto thenSeqOp = createSeqOp(rewriter, brOp.getLoc());
        rewriter.setInsertionPointToEnd(elseCtrlBlock);
        auto elseSeqOp = createSeqOp(rewriter, brOp.getLoc());
        LogicalResult tb =
            scheduleBranchTarget(rewriter, path, brOp.getLoc(), block,
                                 successors[0], thenSeqOp.getBody());
        LogicalResult fb =
            scheduleBranchTarget(rewriter, path, brOp.getLoc(), block,
                                 successors[1], elseSeqOp.getBody());
        return success(tb.succeeded() && fb.succeeded());
      } else {
        // Schedule sequentially within the current parent control block
        return scheduleBranchTarget(rewriter, path, brOp.getLoc(), block,
                                    successors.front(), parentCtrlBlock);
      }
    }
    return success();
  }
};

/// LateSSAReplacement contains various functions for replacing SSA values that
/// were not replaced during op construction.
class LateSSAReplacement : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                                         PatternRewriter &) const override {
    funcOp.walk([&](scf::WhileOp whileOp) {
      /// The yielded values returned from the while op will be present in the
      /// iterargs registers post execution of the loop.
      /// This is done now, as opposed to during BuildWhileGroups since if the
      /// results of the whileOp was replaced before
      /// BuildOpGroups/BuildControl, the whileOp would get dead-code
      /// eliminated.
      for (auto res : state().getWhileIterRegs(whileOp))
        whileOp.getResults()[res.first].replaceAllUsesWith(res.second.out());
    });

    funcOp.walk([&](memref::LoadOp loadOp) {
      if (singleLoadFromMemory(loadOp)) {
        /// In buildOpGroups we did not replace loadOp's results, to ensure a
        /// link between evaluating groups (which fix the input addresses of a
        /// memory op) and a readData result. Now, we may replace these SSA
        /// values with their memoryOp readData output.
        loadOp.getResult().replaceAllUsesWith(
            state().getMemory(loadOp.memref()).readData());
      }
    });
    return success();
  }
};

/// Erases the now converted FuncOp operations.
class CleanupFuncOps : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    rewriter.eraseOp(funcOp);
    return success();
  }
};

/// Creates a new Calyx component for each FuncOp in the program. The
/// top-level function (if specified) is rewritten to 'main').
struct FuncOpConversion : public FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    // Rewrite top-level component name to "main", or else, keep function name
    // as component name
    const auto compName = funcOp.sym_name() == progState().topLevelFunction()
                              ? "main"
                              : funcOp.sym_name();

    // Create I/O ports
    SmallVector<calyx::PortInfo> ports;
    FunctionType funcType = funcOp.getType();
    for (auto &arg : enumerate(funcOp.getArguments()))
      ports.push_back(calyx::PortInfo{
          .name = rewriter.getStringAttr("in" + std::to_string(arg.index())),
          .type = arg.value().getType(),
          .direction = calyx::Direction::Input,
          .attributes = DictionaryAttr::get(rewriter.getContext(), {})});

    for (auto &res : enumerate(funcType.getResults()))
      ports.push_back(calyx::PortInfo{
          .name = rewriter.getStringAttr("out" + std::to_string(res.index())),
          .type = res.value(),
          .direction = calyx::Direction::Output,
          .attributes = DictionaryAttr::get(rewriter.getContext(), {})});

    addMandatoryComponentPorts(rewriter, ports);

    // Create a calyx::ComponentOp corresponding to the to-be-lowered function.
    auto compOp = rewriter.create<calyx::ComponentOp>(
        funcOp.getLoc(), rewriter.getStringAttr(compName), ports);
    rewriter.createBlock(&compOp.getWiresOp().getBodyRegion());
    rewriter.createBlock(&compOp.getControlOp().getBodyRegion());

    // Rewrite the funcOp SSA argument values to the CompOp arguments
    for (auto &arg : enumerate(funcOp.getArguments())) {
      arg.value().replaceAllUsesWith(getComponentInput(compOp, arg.index()));
    }

    // Store function to component mapping for future reference
    funcMap[funcOp] = compOp;
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Simplification patterns
//===----------------------------------------------------------------------===//

/// Removes operations which have an empty body
template <typename TOp>
struct EliminateEmptyOpPattern : mlir::OpRewritePattern<TOp> {
  using mlir::OpRewritePattern<TOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getRegion().empty() || op.getRegion().front().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

template <>
struct EliminateEmptyOpPattern<calyx::IfOp>
    : mlir::OpRewritePattern<calyx::IfOp> {
  using mlir::OpRewritePattern<calyx::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::IfOp op,
                                PatternRewriter &rewriter) const override {
    if (op.thenRegion().empty() || op.thenRegion().front().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

/// Removes nested seq operations
/// seq {        seq {
///   seq {        ...
///     ...  ->  }
///   }
/// }
struct NestedSeqPattern : mlir::OpRewritePattern<calyx::SeqOp> {
  using mlir::OpRewritePattern<calyx::SeqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::SeqOp seqOp,
                                PatternRewriter &rewriter) const override {
    if (isa<calyx::SeqOp>(seqOp->getParentOp())) {
      if (auto *body = seqOp.getBody()) {
        for (auto &op : make_early_inc_range(*body))
          op.moveBefore(seqOp);
        rewriter.eraseOp(seqOp);
        return success();
      }
    }
    return failure();
  }
};

/// Removes calyx::CombGroupOps which are unused. These correspond to
/// combinational groups created during op building that, after conversion,
/// have either been inlined into calyx::GroupOps or are referenced by an
/// if/while with statement.
struct EliminateUnusedCombGroups : mlir::OpRewritePattern<calyx::CombGroupOp> {
  using mlir::OpRewritePattern<calyx::CombGroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::CombGroupOp combGroupOp,
                                PatternRewriter &rewriter) const override {
    auto control =
        combGroupOp->getParentOfType<calyx::ComponentOp>().getControlOp();
    if (!SymbolTable::symbolKnownUseEmpty(combGroupOp.sym_nameAttr(), control))
      return failure();

    rewriter.eraseOp(combGroupOp);
    return success();
  }
};

/// GroupDoneOp's are terminator operations and should therefore be the last
/// operator in a group. During group construction, we always append assignments
/// to the end of a group, resulting in group_done ops migrating away from the
/// terminator position. This pattern moves such ops to the end of their group.
struct NonTerminatingGroupDonePattern
    : mlir::OpRewritePattern<calyx::GroupDoneOp> {
  using mlir::OpRewritePattern<calyx::GroupDoneOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::GroupDoneOp groupDoneOp,
                                PatternRewriter & /*rewriter*/) const override {
    Block *block = groupDoneOp->getBlock();
    if (&block->back() == groupDoneOp)
      return failure();

    groupDoneOp->moveBefore(groupDoneOp->getBlock(),
                            groupDoneOp->getBlock()->end());
    return success();
  }
};

/// When building groups which contain accesses to multiple sequential
/// components, a group_done op is created for each of these. This pattern
/// emits the conjunction of each of the group_done values into a single
/// group_done.
struct MultipleGroupDonePattern : mlir::OpRewritePattern<calyx::GroupOp> {
  using mlir::OpRewritePattern<calyx::GroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::GroupOp groupOp,
                                PatternRewriter &rewriter) const override {
    auto groupDoneOps = SmallVector<calyx::GroupDoneOp>(
        groupOp.getBody()->getOps<calyx::GroupDoneOp>());

    if (groupDoneOps.size() <= 1) {
      // Nothing to do
      return failure();
    }

    // Create an and-tree of all calyx::GroupDoneOp's
    rewriter.setInsertionPointToEnd(groupDoneOps[0]->getBlock());
    Value acc = groupDoneOps[0].src();
    for (auto groupDoneOp : llvm::makeArrayRef(groupDoneOps).drop_front(1)) {
      auto newAndOp = rewriter.create<comb::AndOp>(groupDoneOp.getLoc(), acc,
                                                   groupDoneOp.src());
      acc = newAndOp.getResult();
    }

    // Create a group done op with the complex expression as a guard
    rewriter.create<calyx::GroupDoneOp>(
        groupOp.getLoc(),
        rewriter.create<hw::ConstantOp>(groupOp.getLoc(), APInt(1, 1)), acc);
    for (auto groupDoneOp : groupDoneOps)
      rewriter.eraseOp(groupDoneOp);

    return success();
  }
};

/// Returns true if 'child' is within the child tree of 'parent'
template <typename TParent>
static bool isChildOp(TParent &parent, Operation *child) {
  bool isChild = false;
  parent.walk([&](Operation *op) {
    if (child == op) {
      isChild = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return isChild;
}

/// Returns the last calyx::EnableOp within the child tree of 'parentSeqOp'.
/// If no EnableOp was found (for instance, if a "par" group is present),
/// returns nullptr.
static Operation *getLastSeqEnableOp(calyx::SeqOp parentSeqOp) {
  auto &lastOp = parentSeqOp.getBody()->back();
  if (auto enableOp = dyn_cast<calyx::EnableOp>(lastOp))
    return enableOp.getOperation();
  else if (auto seqOp = dyn_cast<calyx::SeqOp>(lastOp))
    return getLastSeqEnableOp(seqOp);
  return nullptr;
}

/// Removes common tail enable operations for sequential 'then'/'else'
/// branches inside an 'if' operation.
///
///   if %a with %A {           if %a with %A {
///     seq {                     seq {
///       ...                       ...
///       calyx.enable @B       } else {
///     }                         seq {
///   } else {              ->      ...
///     seq {                     }
///       ...                   }
///       calyx.enable @B       calyx.enable @B
///     }
///   }
///
struct CommonIfTailEnablePattern : mlir::OpRewritePattern<calyx::IfOp> {
  using mlir::OpRewritePattern<calyx::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(calyx::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    /// Check if there's anything in the branches; if not,
    /// EliminateEmptyOpPattern will eliminate a potentially
    /// empty/invalid if statement.
    if (ifOp.thenRegion().empty() || ifOp.thenRegion().front().empty())
      return failure();
    if (ifOp.elseRegion().empty() || ifOp.elseRegion().front().empty())
      return failure();

    auto thenSeqOp = dyn_cast<calyx::SeqOp>(ifOp.thenRegion().front().front());
    auto elseSeqOp = dyn_cast<calyx::SeqOp>(ifOp.elseRegion().front().front());
    assert(thenSeqOp && elseSeqOp &&
           "expected nested seq ops in both branches of a calyx.IfOp");

    auto lastThenEnableOp =
        dyn_cast<calyx::EnableOp>(getLastSeqEnableOp(thenSeqOp));
    auto lastElseEnableOp =
        dyn_cast<calyx::EnableOp>(getLastSeqEnableOp(elseSeqOp));

    if (!(lastThenEnableOp && lastElseEnableOp))
      return failure();

    if (lastThenEnableOp.groupName() != lastElseEnableOp.groupName())
      return failure();

    // Erase both enable operations and add group enable operation after the
    // shared IfOp parent.
    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<calyx::EnableOp>(
        ifOp.getLoc(), lastThenEnableOp.groupName(), rewriter.getArrayAttr({}));
    rewriter.eraseOp(lastThenEnableOp);
    rewriter.eraseOp(lastElseEnableOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//
class SCFToCalysPass : public SCFToCalyxBase<SCFToCalysPass> {
public:
  SCFToCalysPass()
      : SCFToCalyxBase<SCFToCalysPass>(), m_partialPatternRes(success()) {}
  void runOnOperation() override;

  LogicalResult mainFuncIsDefined(mlir::ModuleOp moduleOp,
                                  StringRef topLevelFunction) {
    if (SymbolTable::lookupSymbolIn(moduleOp, topLevelFunction) == nullptr) {
      moduleOp.emitError("Main function '" + topLevelFunction +
                         "' not found in module.");
      return failure();
    }
    return success();
  }

  //// Creates a new Calyx program with the contents of the source module
  /// inlined within.
  /// Furthermore, this function performs validation on the input function, to
  /// ensure that we've implemented the capabilities necessary to convert it.
  ///
  /// @todo: this seems unnecessarily complicated:
  /// A restriction of the current infrastructure is that a top-level 'module'
  /// cannot be overwritten (even though this is essentially what is going on
  /// when replacing standard::ModuleOp with calyx::ProgramOp). see:
  /// https://llvm.discourse.group/t/de-privileging-moduleop-in-translation-apis/3733/26
  LogicalResult createProgram(calyx::ProgramOp *programOpOut) {
    // Program conversion
    auto createModuleConvTarget = [&]() {
      ConversionTarget target(getContext());
      target.addLegalDialect<calyx::CalyxDialect>();
      target.addLegalDialect<scf::SCFDialect>();
      target.addIllegalDialect<hw::HWDialect>();
      target.addIllegalDialect<comb::CombDialect>();

      // For loops should have been lowered to while loops
      target.addIllegalOp<scf::ForOp>();

      // Only accept std operations which we've added lowerings for
      target.addIllegalDialect<StandardOpsDialect>();
      target.addLegalOp<AddIOp, SubIOp, CmpIOp, ShiftLeftOp,
                        UnsignedShiftRightOp, SignedShiftRightOp, AndOp, XOrOp,
                        OrOp, ZeroExtendIOp, TruncateIOp, CondBranchOp,
                        BranchOp, ReturnOp, ConstantOp, IndexCastOp>();

      target.addDynamicallyLegalOp<mlir::ModuleOp>([](mlir::ModuleOp moduleOp) {
        // A module is legalized after we've added a nested
        // calyx::ProgramOp within it.
        bool ok = false;
        moduleOp.walk([&](calyx::ProgramOp) {
          ok = true;
          return WalkResult::interrupt();
        });
        return ok;
      });
      return target;
    };

    RewritePatternSet patterns(&getContext());
    patterns.add<ModuleOpConversion>(&getContext(), programOpOut);
    auto target = createModuleConvTarget();
    return applyPartialConversion(getOperation(), target, std::move(patterns));
  }

  struct LoweringPattern {
    enum class Strategy { Once, Greedy };
    RewritePatternSet pattern;
    Strategy strategy;
  };

  /// 'Once' patterns are expected to take an additional LogicalResult&
  /// argument, to forward their result state (greedyPatternRewriteDriver
  /// results are skipped for Once patterns).
  template <typename TPattern, typename... PatternArgs>
  void addOncePattern(SmallVectorImpl<LoweringPattern> &patterns,
                      PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), m_partialPatternRes, args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Once});
  }

  template <typename TPattern, typename... PatternArgs>
  void addGreedyPattern(SmallVectorImpl<LoweringPattern> &patterns,
                        PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), args...);
    patterns.push_back(
        LoweringPattern{std::move(ps), LoweringPattern::Strategy::Greedy});
  }

  LogicalResult runPartialPattern(RewritePatternSet &pattern, bool runOnce) {
    auto &nativePatternSet = pattern.getNativePatterns();
    assert(nativePatternSet.size() == 1 &&
           "Should only apply 1 partial lowering pattern at once");

    // During component creation, the function body is inlined into the
    // component body for further processing. However, proper control flow
    // will only be established later in the conversion process, so ensure
    // that rewriter optimizations (especially DCE) are disabled.
    GreedyRewriteConfig config;
    config.enableRegionSimplification = false;
    if (runOnce)
      config.maxIterations = 1;

    /// can't return applyPatternsAndFoldGreedily. Root isn't
    /// necessarily erased so it will always return failed(). Instead, forward
    /// the 'succeeded' value from PartialLoweringPatternBase.
    (void)applyPatternsAndFoldGreedily(m_moduleOp, std::move(pattern), config);
    return m_partialPatternRes;
  }

private:
  LogicalResult m_partialPatternRes;
  mlir::ModuleOp m_moduleOp;
  calyx::ProgramOp m_programOp;
  std::shared_ptr<ProgramLoweringState> m_loweringState = nullptr;
};

void SCFToCalysPass::runOnOperation() {
  m_moduleOp = getOperation();
  std::string topLevelFunction;
  if (topLevelComponent.empty()) {
    topLevelFunction = "main";
  } else {
    topLevelFunction = topLevelComponent;
  }

  if (failed(mainFuncIsDefined(getOperation(), topLevelFunction))) {
    signalPassFailure();
    return;
  }

  /// Start conversion
  if (failed(createProgram(&m_programOp))) {
    signalPassFailure();
    return;
  }
  assert(m_programOp.getOperation() != nullptr &&
         "programOp should have been set during module "
         "conversion, if module conversion succeeded.");
  m_loweringState =
      std::make_shared<ProgramLoweringState>(m_programOp, topLevelFunction);

  /// --------------------------------------------------------------------------
  /// @note: If you are a developer, it may be helpful to add a
  /// 'm_module.dump()' operation after the execution of each stage to view the
  /// transformations that's going on.
  ///
  /// @todo: Some of the following passes could probably be merged to reduce the
  /// # of times the IR is walked, if we care about execution time. However, the
  /// passes are written to be as atomic as possible for readability and
  /// maintainability.
  /// --------------------------------------------------------------------------
  FuncMapping funcMap;
  SmallVector<LoweringPattern, 8> loweringPatterns;
  /// Wrap the top-level module into a calyx::ProgramOp. Additionally, this pass
  /// generates a ConversionTarget used to validate the operations within the
  /// program, and terminates if we identify any unsupported dialects and/or
  /// operations.
  addOncePattern<FuncOpConversion>(loweringPatterns, funcMap, *m_loweringState);

  /// This pass inlines scf.ExecuteRegionOp's by adding control-flow.
  addGreedyPattern<InlineExecuteRegionOpPattern>(loweringPatterns);

  /// This pattern converts all index types to a predefined width (currently
  /// i32). @todo In the future, we'd like to replace this with a proper
  /// width-inference pass.
  addOncePattern<ConvertIndexTypes>(loweringPatterns, funcMap,
                                    *m_loweringState);
  /// This pattern creates registers for all basic-block arguments.
  addOncePattern<BuildBBRegs>(loweringPatterns, funcMap, *m_loweringState);

  /// This pattern creates registers for the function return values.
  addOncePattern<BuildReturnRegs>(loweringPatterns, funcMap, *m_loweringState);

  /// This pattern creates registers for iteration arguments of scf.while
  /// operations. Additionally, creates a group for assigning the initial
  /// value of the iteration argument registers.
  addOncePattern<BuildWhileGroups>(loweringPatterns, funcMap, *m_loweringState);

  /// This pattern converts operations within basic blocks to Calyx library
  /// operators. Each operation is assigned in a distinct calyx::GroupOp. These
  /// groups are registered with the Block* of which the operation originated
  /// from. This is used during control schedule generation.
  /// By having a distinct group for each operation, groups are analogous to
  /// SSA values in the source program. This is fundamental for the
  /// MergeCombGroups pass.
  addOncePattern<BuildOpGroups>(loweringPatterns, funcMap, *m_loweringState);

  /// This pattern traverses the CFG of the program and generates a control
  /// schedule based on the calyx::GroupOp's which were registered for each
  /// basic block in the source function.
  addOncePattern<BuildControl>(loweringPatterns, funcMap, *m_loweringState);

  /// This pass recursively inlines use-def chains of combinational logic (from
  /// non-stateful groups) into stateful groups.
  addOncePattern<InlineCombGroups>(loweringPatterns, *m_loweringState);

  /// This pattern performs various SSA replacements that must be done
  /// after control generation.
  addOncePattern<LateSSAReplacement>(loweringPatterns, funcMap,
                                     *m_loweringState);

  /// This pattern rewrites accesses to memories which are too wide due to
  /// index types being converted to a fixed-width integer type.
  addOncePattern<RewriteMemoryAccesses>(loweringPatterns, *m_loweringState);

  /// This pattern removes the source FuncOp which has now been converted into
  /// a Calyx component.
  addOncePattern<CleanupFuncOps>(loweringPatterns, funcMap, *m_loweringState);

  /// Sequentially apply each lowering pattern.
  for (auto &pat : loweringPatterns) {
    auto res = runPartialPattern(pat.pattern,
                                 /*runOnce=*/pat.strategy ==
                                     LoweringPattern::Strategy::Once);
    if (failed(res)) {
      signalPassFailure();
      return;
    }
  }

  //===----------------------------------------------------------------------===//
  // Cleanup
  //===----------------------------------------------------------------------===//
  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns
      .add<EliminateEmptyOpPattern<calyx::CombGroupOp>,
           EliminateEmptyOpPattern<calyx::GroupOp>,
           EliminateEmptyOpPattern<calyx::SeqOp>,
           EliminateEmptyOpPattern<calyx::ParOp>,
           EliminateEmptyOpPattern<calyx::IfOp>,
           EliminateEmptyOpPattern<calyx::WhileOp>, NestedSeqPattern,
           CommonIfTailEnablePattern, MultipleGroupDonePattern,
           NonTerminatingGroupDonePattern, EliminateUnusedCombGroups>(
          &getContext());
  if (failed(applyPatternsAndFoldGreedily(m_moduleOp,
                                          std::move(cleanupPatterns)))) {
    signalPassFailure();
    return;
  }
}

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSCFToCalyxPass() {
  return std::make_unique<SCFToCalysPass>();
}

} // namespace circt
