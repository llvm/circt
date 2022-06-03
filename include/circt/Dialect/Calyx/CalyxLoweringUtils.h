//===- CalyxLoweringUtils.h - Calyx lowering utility methods ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines various lowering utility methods for converting to
// and from Calyx programs.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H
#define CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H

#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include <variant>

namespace circt {
namespace calyx {

// Walks the control of this component, and appends source information for leaf
// nodes. It also appends a position attribute that connects the source location
// metadata to the corresponding control operation.
WalkResult
getCiderSourceLocationMetadata(calyx::ComponentOp component,
                               SmallVectorImpl<Attribute> &sourceLocations);

// Tries to match a constant value defined by op. If the match was
// successful, returns true and binds the constant to 'value'. If unsuccessful,
// the value is unmodified.
bool matchConstantOp(Operation *op, APInt &value);

// Returns true if there exists only a single memref::LoadOp which loads from
// the memory referenced by loadOp.
bool singleLoadFromMemory(Value memoryReference);

// Returns true if there are no memref::StoreOp uses with the referenced
// memory.
bool noStoresToMemory(Value memoryReference);

// Get the index'th output port of compOp.
Value getComponentOutput(calyx::ComponentOp compOp, unsigned outPortIdx);

// If the provided type is an index type, converts it to i32, else, returns the
// unmodified type.
Type convIndexType(OpBuilder &builder, Type type);

// Creates a new calyx::CombGroupOp or calyx::GroupOp group within compOp.
template <typename TGroup>
TGroup createGroup(OpBuilder &builder, calyx::ComponentOp compOp, Location loc,
                   Twine uniqueName) {
  mlir::IRRewriter::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(compOp.getWiresOp().getBody());
  return builder.create<TGroup>(loc, uniqueName.str());
}

/// Creates register assignment operations within the provided groupOp.
/// The component operation will house the constants.
void buildAssignmentsForRegisterWrite(OpBuilder &builder,
                                      calyx::GroupOp groupOp,
                                      calyx::ComponentOp componentOp,
                                      calyx::RegisterOp &reg, Value inputValue);

// A structure representing a set of ports which act as a memory interface for
// external memories.
struct MemoryPortsImpl {
  Value readData;
  Value done;
  Value writeData;
  SmallVector<Value> addrPorts;
  Value writeEn;
};

// Represents the interface of memory in Calyx. The various lowering passes
// are agnostic wrt. whether working with a calyx::MemoryOp (internally
// allocated memory) or MemoryPortsImpl (external memory).
struct MemoryInterface {
  MemoryInterface();
  explicit MemoryInterface(const MemoryPortsImpl &ports);
  explicit MemoryInterface(calyx::MemoryOp memOp);

  // Getter methods for each memory interface port.
  Value readData();
  Value done();
  Value writeData();
  Value writeEn();
  ValueRange addrPorts();

private:
  std::variant<calyx::MemoryOp, MemoryPortsImpl> impl;
};

// A common interface for loop operations that need to be lowered to Calyx.
class LoopInterface {
public:
  virtual ~LoopInterface();

  // Returns the arguments to this loop operation.
  virtual Block::BlockArgListType getBodyArgs() = 0;

  // Returns body of this loop operation.
  virtual Block *getBodyBlock() = 0;

  // Returns the Block in which the condition exists.
  virtual Block *getConditionBlock() = 0;

  // Returns the condition as a Value.
  virtual Value getConditionValue() = 0;

  // Returns the number of iterations the loop will conduct if known.
  virtual Optional<uint64_t> getBound() = 0;
};

// Provides an interface for the control flow `while` operation across different
// dialects.
template <typename T>
class WhileOpInterface : LoopInterface {
  static_assert(std::is_convertible_v<T, Operation *>);

public:
  explicit WhileOpInterface(T op) : impl(op) {}
  explicit WhileOpInterface(Operation *op) : impl(dyn_cast_or_null<T>(op)) {}

  // Returns the operation.
  T getOperation() { return impl; }

  // Returns the source location of the operation.
  Location getLoc() { return impl->getLoc(); }

private:
  T impl;
};

//===----------------------------------------------------------------------===//
// Lowering state classes
//===----------------------------------------------------------------------===//

// Handles state during the lowering of a loop. It will be used for
// several lowering patterns.
template <typename Loop>
class LoopLoweringStateInterface {
  static_assert(std::is_base_of_v<LoopInterface, Loop>);

public:
  ~LoopLoweringStateInterface() = default;

  /// Register reg as being the idx'th iter_args register for 'op'.
  void addLoopIterReg(Loop op, calyx::RegisterOp reg, unsigned idx) {
    assert(loopIterRegs[op.getOperation()].count(idx) == 0 &&
           "A register was already registered for the given loop iter_arg "
           "index");
    assert(idx < op.getBodyArgs().size());
    loopIterRegs[op.getOperation()][idx] = reg;
  }

  /// Return a mapping of block argument indices to block argument.
  calyx::RegisterOp getLoopIterReg(Loop op, unsigned idx) {
    auto iterRegs = getLoopIterRegs(op);
    auto it = iterRegs.find(idx);
    assert(it != iterRegs.end() &&
           "No iter arg register set for the provided index");
    return it->second;
  }

  /// Return a mapping of block argument indices to block argument.
  const DenseMap<unsigned, calyx::RegisterOp> &getLoopIterRegs(Loop op) {
    return loopIterRegs[op.getOperation()];
  }

  /// Registers grp to be the loop latch group of `op`.
  void setLoopLatchGroup(Loop op, calyx::GroupOp group) {
    Operation *operation = op.getOperation();
    assert(loopLatchGroups.count(operation) == 0 &&
           "A latch group was already set for this loopOp");
    loopLatchGroups[operation] = group;
  }

  /// Retrieve the loop latch group registered for `op`.
  calyx::GroupOp getLoopLatchGroup(Loop op) {
    auto it = loopLatchGroups.find(op.getOperation());
    assert(it != loopLatchGroups.end() &&
           "No loop latch group was set for this loopOp");
    return it->second;
  }

  /// Creates a new group that assigns the 'ops' values to the iter arg
  /// registers of the loop operation.
  calyx::GroupOp buildLoopIterArgAssignments(OpBuilder &builder, Loop op,
                                             calyx::ComponentOp componentOp,
                                             Twine uniqueSuffix,
                                             MutableArrayRef<OpOperand> ops) {
    /// Pass iteration arguments through registers. This follows closely
    /// to what is done for branch ops.
    std::string groupName = "assign_" + uniqueSuffix.str();
    auto groupOp = calyx::createGroup<calyx::GroupOp>(builder, componentOp,
                                                      op.getLoc(), groupName);
    /// Create register assignment for each iter_arg. a calyx::GroupDone signal
    /// is created for each register. These will be &'ed together in
    /// MultipleGroupDonePattern.
    for (OpOperand &arg : ops) {
      auto reg = getLoopIterReg(op, arg.getOperandNumber());
      buildAssignmentsForRegisterWrite(builder, groupOp, componentOp, reg,
                                       arg.get());
    }
    return groupOp;
  }

private:
  /// A mapping from loop ops to iteration argument registers.
  DenseMap<Operation *, DenseMap<unsigned, calyx::RegisterOp>> loopIterRegs;

  /// A loop latch group is a group that should be sequentially executed when
  /// finishing a loop body. The execution of this group will write the
  /// yield'ed loop body values to the iteration argument registers.
  DenseMap<Operation *, calyx::GroupOp> loopLatchGroups;
};

// Handles state during the lowering of a Calyx component. This provides common
// tools for converting to the Calyx ComponentOp.
class ComponentLoweringStateInterface {
public:
  ComponentLoweringStateInterface(calyx::ComponentOp component);

  ~ComponentLoweringStateInterface();

  /// Returns the calyx::ComponentOp associated with this lowering state.
  calyx::ComponentOp getComponentOp();

  /// Register reg as being the idx'th argument register for block. This is
  /// necessary for the `BuildBBReg` pass.
  void addBlockArgReg(Block *block, calyx::RegisterOp reg, unsigned idx);

  /// Return a mapping of block argument indices to block argument registers.
  /// This is necessary for the `BuildBBReg` pass.
  const DenseMap<unsigned, calyx::RegisterOp> &getBlockArgRegs(Block *block);

  /// Register 'grp' as a group which performs block argument
  /// register transfer when transitioning from basic block from to to.
  void addBlockArgGroup(Block *from, Block *to, calyx::GroupOp grp);

  /// Returns a list of groups to be evaluated to perform the block argument
  /// register assignments when transitioning from basic block 'from' to 'to'.
  ArrayRef<calyx::GroupOp> getBlockArgGroups(Block *from, Block *to);

  /// Returns a unique name within compOp with the provided prefix.
  std::string getUniqueName(StringRef prefix);

  /// Returns a unique name associated with a specific operation.
  StringRef getUniqueName(Operation *op);

  /// Registers a unique name for a given operation using a provided prefix.
  void setUniqueName(Operation *op, StringRef prefix);

  /// Register value v as being evaluated when scheduling group.
  void registerEvaluatingGroup(Value v, calyx::GroupInterface group);

  /// Register reg as being the idx'th return value register.
  void addReturnReg(calyx::RegisterOp reg, unsigned idx);

  /// Returns the idx'th return value register.
  calyx::RegisterOp getReturnReg(unsigned idx);

  /// Registers a memory interface as being associated with a memory identified
  /// by 'memref'.
  void registerMemoryInterface(Value memref,
                               const calyx::MemoryInterface &memoryInterface);

  /// Returns the memory interface registered for the given memref.
  calyx::MemoryInterface getMemoryInterface(Value memref);

  /// If v is an input to any memory registered within this component, returns
  /// the memory. If not, returns null.
  Optional<calyx::MemoryInterface> isInputPortOfMemory(Value v);

  /// Assign a mapping between the source funcOp result indices and the
  /// corresponding output port indices of this componentOp.
  void setFuncOpResultMapping(const DenseMap<unsigned, unsigned> &mapping);

  /// Get the output port index of this component for which the funcReturnIdx of
  /// the original function maps to.
  unsigned getFuncOpResultMapping(unsigned funcReturnIdx);

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

  template <typename TLibraryOp>
  TLibraryOp getNewLibraryOpInstance(OpBuilder &builder, Location loc,
                                     TypeRange resTypes) {
    mlir::IRRewriter::InsertionGuard guard(builder);
    Block *body = component.getBody();
    builder.setInsertionPoint(body, body->begin());
    auto name = TLibraryOp::getOperationName().split(".").second;
    return builder.create<TLibraryOp>(loc, getUniqueName(name), resTypes);
  }

private:
  /// The component which this lowering state is associated to.
  calyx::ComponentOp component;

  /// A mapping from blocks to block argument registers.
  DenseMap<Block *, DenseMap<unsigned, calyx::RegisterOp>> blockArgRegs;

  /// Block arg groups is a list of groups that should be sequentially
  /// executed when passing control from the source to destination block.
  /// Block arg groups are executed before blockScheduleables (akin to a
  /// phi-node).
  DenseMap<Block *, DenseMap<Block *, SmallVector<calyx::GroupOp>>>
      blockArgGroups;

  /// A mapping of string prefixes and the current uniqueness counter for that
  /// prefix. Used to generate unique names.
  std::map<std::string, unsigned> prefixIdMap;

  /// A mapping from Operations and previously assigned unique name of the op.
  std::map<Operation *, std::string> opNames;

  /// A mapping between SSA values and the groups which assign them.
  DenseMap<Value, calyx::GroupInterface> valueGroupAssigns;

  /// A mapping from return value indexes to return value registers.
  DenseMap<unsigned, calyx::RegisterOp> returnRegs;

  /// A mapping from memref's to their corresponding Calyx memory interface.
  DenseMap<Value, calyx::MemoryInterface> memories;

  /// A mapping between the source funcOp result indices and the corresponding
  /// output port indices of this componentOp.
  DenseMap<unsigned, unsigned> funcOpResultMapping;
};

/// An interface for conversion passes that lower Calyx programs. This handles
/// state during the lowering of a Calyx program.
template <typename TComponentLoweringState>
class ProgramLoweringState {
public:
  explicit ProgramLoweringState(calyx::ProgramOp program,
                                StringRef topLevelFunction)
      : topLevelFunction(topLevelFunction), program(program) {}

  /// Returns the current program.
  calyx::ProgramOp getProgram() {
    assert(program.getOperation() != nullptr);
    return program;
  }

  /// Returns the name of the top-level function in the source program.
  StringRef getTopLevelFunction() const { return topLevelFunction; }

  /// Returns the component lowering state associated with compOp.
  TComponentLoweringState &compLoweringState(calyx::ComponentOp compOp) {
    auto it = compStates.find(compOp);
    if (it != compStates.end())
      return it->second;

    /// Create a new ComponentLoweringState for the compOp.
    auto newCompStateIt = compStates.try_emplace(compOp, compOp);
    return newCompStateIt.first->second;
  }

  /// Returns a meaningful name for a value within the program scope.
  template <typename ValueOrBlock>
  std::string irName(ValueOrBlock &v) {
    std::string s;
    llvm::raw_string_ostream os(s);
    mlir::AsmState asmState(program);
    v.printAsOperand(os, asmState);
    return s;
  }

  /// Returns a meaningful name for a block within the program scope (removes
  /// the ^ prefix from block names).
  std::string blockName(Block *b) {
    std::string blockName = irName(*b);
    blockName.erase(std::remove(blockName.begin(), blockName.end(), '^'),
                    blockName.end());
    return blockName;
  }

private:
  /// The name of this top-level function.
  StringRef topLevelFunction;
  /// The program associated with this state.
  calyx::ProgramOp program;
  /// Mapping from ComponentOp to component lowering state.
  DenseMap<Operation *, TComponentLoweringState> compStates;
};

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

struct ModuleOpConversion : public OpRewritePattern<mlir::ModuleOp> {
  ModuleOpConversion(MLIRContext *context, StringRef topLevelFunction,
                     calyx::ProgramOp *programOpOutput);

  LogicalResult matchAndRewrite(mlir::ModuleOp moduleOp,
                                PatternRewriter &rewriter) const override;

private:
  calyx::ProgramOp *programOpOutput;
  StringRef topLevelFunction;
};

/// FuncOpPartialLoweringPatterns are patterns which intend to match on FuncOps
/// and then perform their own walking of the IR. FuncOpPartialLoweringPatterns
/// have direct access to the TComponentLoweringState for the corresponding
/// component of the matched FuncOp.
template <typename TComponentLoweringState>
class FuncOpPartialLoweringPattern
    : public calyx::PartialLoweringPattern<mlir::func::FuncOp> {

public:
  FuncOpPartialLoweringPattern(
      MLIRContext *context, LogicalResult &resRef,
      DenseMap<mlir::func::FuncOp, calyx::ComponentOp> &map,
      calyx::ProgramLoweringState<TComponentLoweringState> &pls)
      : PartialLoweringPattern(context, resRef), functionMapping(map),
        programLoweringState(pls) {}

  LogicalResult partiallyLower(mlir::func::FuncOp funcOp,
                               PatternRewriter &rewriter) const override final {
    // Initialize the component op references if a calyx::ComponentOp has been
    // created for the matched funcOp.
    if (auto it = functionMapping.find(funcOp); it != functionMapping.end()) {
      componentOp = &it->second;
      componentLoweringState =
          &programLoweringState.compLoweringState(*getComponent());
    }

    return PartiallyLowerFuncToComp(funcOp, rewriter);
  }

  /// Returns the component operation associated with the currently executing
  /// partial lowering.
  calyx::ComponentOp *getComponent() const {
    assert(componentOp != nullptr &&
           "Component operation should be set during pattern construction");
    return componentOp;
  }

  /// Returns the component state associated with the currently executing
  /// partial lowering.
  TComponentLoweringState &getComponentState() const {
    assert(
        componentLoweringState != nullptr &&
        "Component lowering state should be set during pattern construction");
    return *componentLoweringState;
  }

  /// Return the program lowering state for this pattern.
  calyx::ProgramLoweringState<TComponentLoweringState> &programState() const {
    return programLoweringState;
  }

  /// Partial lowering implementation.
  virtual LogicalResult
  PartiallyLowerFuncToComp(mlir::func::FuncOp funcOp,
                           PatternRewriter &rewriter) const = 0;

protected:
  // A map from FuncOp to it's respective ComponentOp lowering.
  DenseMap<mlir::func::FuncOp, calyx::ComponentOp> &functionMapping;

private:
  mutable calyx::ComponentOp *componentOp = nullptr;
  mutable TComponentLoweringState *componentLoweringState = nullptr;
  calyx::ProgramLoweringState<TComponentLoweringState> &programLoweringState;
};

/// Iterate through the operations of a source function and instantiate
/// components or primitives based on the type of the operations.
template <typename TComponentLoweringState>
class BuildOpGroups
    : public calyx::FuncOpPartialLoweringPattern<TComponentLoweringState> {
  using FuncOpPartialLoweringPattern<
      TComponentLoweringState>::FuncOpPartialLoweringPattern;

  LogicalResult
  PartiallyLowerFuncToComp(mlir::func::FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// We walk the operations of the funcOp to ensure that all def's have
    /// been visited before their uses.
    bool opBuiltSuccessfully = true;
    funcOp.walk([&](Operation *_op) {
      opBuiltSuccessfully &=
          TypeSwitch<mlir::Operation *, bool>(_op)
              .template Case<mlir::arith::ConstantOp, mlir::func::ReturnOp,
                             mlir::BranchOpInterface,
                             /// memref
                             mlir::memref::AllocOp, mlir::memref::AllocaOp,
                             mlir::memref::LoadOp, mlir::memref::StoreOp,
                             /// standard arithmetic
                             mlir::arith::AddIOp, mlir::arith::SubIOp,
                             mlir::arith::CmpIOp, mlir::arith::ShLIOp,
                             mlir::arith::ShRUIOp, mlir::arith::ShRSIOp,
                             mlir::arith::AndIOp, mlir::arith::XOrIOp,
                             mlir::arith::OrIOp, mlir::arith::ExtUIOp,
                             mlir::arith::TruncIOp, mlir::arith::MulIOp,
                             mlir::arith::DivUIOp, mlir::arith::RemUIOp,
                             mlir::arith::IndexCastOp,
                             /// static logic
                             staticlogic::PipelineTerminatorOp>(
                  [&](auto op) { return buildOp(rewriter, op).succeeded(); })
              .template Case<mlir::func::FuncOp, staticlogic::PipelineWhileOp,
                             staticlogic::PipelineRegisterOp,
                             staticlogic::PipelineStageOp>([&](auto) {
                /// Skip: these special cases will be handled separately.
                return true;
              })
              .Default([&](auto op) {
                op->emitError() << "Unhandled operation during BuildOpGroups()";
                return false;
              });

      return opBuiltSuccessfully ? WalkResult::advance()
                                 : WalkResult::interrupt();
    });

    return success(opBuiltSuccessfully);
  }

private:
  /// Op builder specializations.
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::BranchOpInterface brOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::ConstantOp constOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::AddIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::SubIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::MulIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::DivUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::RemUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::ShRUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::ShRSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::ShLIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::AndIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, mlir::arith::OrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::XOrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::CmpIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::TruncIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::ExtUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::func::ReturnOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::arith::IndexCastOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::memref::AllocOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::memref::AllocaOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::memref::LoadOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        mlir::memref::StoreOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        staticlogic::PipelineTerminatorOp op) const;

  /// buildLibraryOp will build a TCalyxLibOp inside a TGroupOp based on the
  /// source operation TSrcOp.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op,
                               TypeRange srcTypes, TypeRange dstTypes) const {
    SmallVector<Type> types;
    llvm::append_range(types, srcTypes);
    llvm::append_range(types, dstTypes);

    auto calyxOp =
        this->getComponentState().template getNewLibraryOpInstance<TCalyxLibOp>(
            rewriter, op.getLoc(), types);

    auto directions = calyxOp.portDirections();
    SmallVector<Value, 4> opInputPorts;
    SmallVector<Value, 4> opOutputPorts;
    for (auto dir : llvm::enumerate(directions)) {
      if (dir.value() == calyx::Direction::Input)
        opInputPorts.push_back(calyxOp.getResult(dir.index()));
      else
        opOutputPorts.push_back(calyxOp.getResult(dir.index()));
    }
    assert(
        opInputPorts.size() == op->getNumOperands() &&
        opOutputPorts.size() == op->getNumResults() &&
        "Expected an equal number of in/out ports in the Calyx library op with "
        "respect to the number of operands/results of the source operation.");

    /// Create assignments to the inputs of the library op.
    auto group = createGroupForOp<TGroupOp>(rewriter, op);
    rewriter.setInsertionPointToEnd(group.getBody());
    for (auto dstOp : enumerate(opInputPorts))
      rewriter.create<calyx::AssignOp>(op.getLoc(), dstOp.value(),
                                       op->getOperand(dstOp.index()));

    /// Replace the result values of the source operator with the new operator.
    for (auto res : enumerate(opOutputPorts)) {
      this->getComponentState().registerEvaluatingGroup(res.value(), group);
      op->getResult(res.index()).replaceAllUsesWith(res.value());
    }
    return success();
  }

  /// buildLibraryOp which provides in- and output types based on the operands
  /// and results of the op argument.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op) const {
    return buildLibraryOp<TGroupOp, TCalyxLibOp, TSrcOp>(
        rewriter, op, op.getOperandTypes(), op->getResultTypes());
  }

  /// Creates a group named by the basic block which the input op resides in.
  template <typename TGroupOp>
  TGroupOp createGroupForOp(PatternRewriter &rewriter, Operation *op) const {
    Block *block = op->getBlock();
    auto groupName = this->getComponentState().getUniqueName(
        this->programState().blockName(block));
    return calyx::createGroup<TGroupOp>(rewriter, this->getComponentOp(),
                                        op->getLoc(), groupName);
  }

  /// buildLibraryBinaryPipeOp will build a TCalyxLibBinaryPipeOp, to
  /// deal with MulIOp, DivUIOp and RemUIOp.
  template <typename TOpType, typename TSrcOp>
  LogicalResult buildLibraryBinaryPipeOp(PatternRewriter &rewriter, TSrcOp op,
                                         TOpType opPipe, Value out) const {
    StringRef opName = TSrcOp::getOperationName().split(".").second;
    Location loc = op.getLoc();
    Type width = op.getResult().getType();
    // Pass the result from the Operation to the Calyx primitive.
    op.getResult().replaceAllUsesWith(out);
    auto reg = createRegister(op.getLoc(), rewriter, *this->getComponent(),
                              width.getIntOrFloatBitWidth(),
                              this->getComponentState().getUniqueName(opName));
    // Operation pipelines are not combinational, so a GroupOp is required.
    auto group = createGroupForOp<calyx::GroupOp>(rewriter, op);
    this->getComponentState().addBlockScheduleable(op->getBlock(), group);

    rewriter.setInsertionPointToEnd(group.getBody());
    rewriter.create<calyx::AssignOp>(loc, opPipe.left(), op.getLhs());
    rewriter.create<calyx::AssignOp>(loc, opPipe.right(), op.getRhs());
    // Write the output to this register.
    rewriter.create<calyx::AssignOp>(loc, reg.in(), out);
    // The write enable port is high when the pipeline is done.
    rewriter.create<calyx::AssignOp>(loc, reg.write_en(), opPipe.done());
    rewriter.create<calyx::AssignOp>(
        loc, opPipe.go(), createConstant(loc, rewriter, *this->getComponent(), 1, 1));
    // The group is done when the register write is complete.
    rewriter.create<calyx::GroupDoneOp>(loc, reg.done());

    // Register the values for the pipeline.
    this->getComponentState().registerEvaluatingGroup(out, group);
    this->getComponentState().registerEvaluatingGroup(opPipe.left(), group);
    this->getComponentState().registerEvaluatingGroup(opPipe.right(), group);

    return success();
  }

  /// Creates assignments within the provided group to the address ports of the
  /// memoryOp based on the provided addressValues.
  void assignAddressPorts(PatternRewriter &rewriter, Location loc,
                          calyx::GroupInterface group,
                          calyx::MemoryInterface memoryInterface,
                          Operation::operand_range addressValues) const {
    mlir::IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(group.getBody());
    auto addrPorts = memoryInterface.addrPorts();
    assert(addrPorts.size() == addressValues.size() &&
           "Mismatch between number of address ports of the provided memory "
           "and address assignment values");
    for (auto &address : llvm::enumerate(addressValues))
      rewriter.create<calyx::AssignOp>(loc, addrPorts[address.index()],
                                       address.value());
  }
};

} // namespace calyx
} // namespace circt

#endif // CIRCT_DIALECT_CALYX_CALYXLOWERINGUTILS_H
