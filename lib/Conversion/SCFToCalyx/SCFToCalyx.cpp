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

#include "circt/Conversion/SCFToCalyx.h"
#include "circt/Dialect/Calyx/CalyxHelpers.h"
#include "circt/Dialect/Calyx/CalyxLoweringUtils.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <filesystem>
#include <fstream>

#include <locale>
#include <numeric>
#include <variant>

namespace circt {
#define GEN_PASS_DEF_SCFTOCALYX
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace llvm;
using namespace mlir;
using namespace mlir::arith;
using namespace mlir::cf;
using namespace mlir::func;
namespace circt {
class ComponentLoweringStateInterface;
namespace scftocalyx {

//===----------------------------------------------------------------------===//
// Utility types
//===----------------------------------------------------------------------===//

class ScfWhileOp : public calyx::WhileOpInterface<scf::WhileOp> {
public:
  explicit ScfWhileOp(scf::WhileOp op)
      : calyx::WhileOpInterface<scf::WhileOp>(op) {}

  Block::BlockArgListType getBodyArgs() override {
    return getOperation().getAfterArguments();
  }

  Block *getBodyBlock() override { return &getOperation().getAfter().front(); }

  Block *getConditionBlock() override {
    return &getOperation().getBefore().front();
  }

  Value getConditionValue() override {
    return getOperation().getConditionOp().getOperand(0);
  }

  std::optional<int64_t> getBound() override { return std::nullopt; }
};

class ScfForOp : public calyx::RepeatOpInterface<scf::ForOp> {
public:
  explicit ScfForOp(scf::ForOp op) : calyx::RepeatOpInterface<scf::ForOp>(op) {}

  Block::BlockArgListType getBodyArgs() override {
    return getOperation().getRegion().getArguments();
  }

  Block *getBodyBlock() override {
    return &getOperation().getRegion().getBlocks().front();
  }

  std::optional<int64_t> getBound() override {
    return constantTripCount(getOperation().getLowerBound(),
                             getOperation().getUpperBound(),
                             getOperation().getStep());
  }
};

//===----------------------------------------------------------------------===//
// Lowering state classes
//===----------------------------------------------------------------------===//

struct IfScheduleable {
  scf::IfOp ifOp;
};

struct WhileScheduleable {
  /// While operation to schedule.
  ScfWhileOp whileOp;
};

struct ForScheduleable {
  /// For operation to schedule.
  ScfForOp forOp;
  /// Bound
  uint64_t bound;
};

struct CallScheduleable {
  /// Instance for invoking.
  calyx::InstanceOp instanceOp;
  // CallOp for getting the arguments.
  func::CallOp callOp;
};

struct ParScheduleable {
  /// Parallel operation to schedule.
  scf::ParallelOp parOp;
};

/// A variant of types representing scheduleable operations.
using Scheduleable =
    std::variant<calyx::GroupOp, WhileScheduleable, ForScheduleable,
                 IfScheduleable, CallScheduleable, ParScheduleable>;

class IfLoweringStateInterface {
public:
  void setThenGroup(scf::IfOp op, calyx::GroupOp group) {
    Operation *operation = op.getOperation();
    assert(thenGroup.count(operation) == 0 &&
           "A then group was already set for this scf::IfOp!\n");
    thenGroup[operation] = group;
  }

  calyx::GroupOp getThenGroup(scf::IfOp op) {
    auto it = thenGroup.find(op.getOperation());
    assert(it != thenGroup.end() &&
           "No then group was set for this scf::IfOp!\n");
    return it->second;
  }

  void setElseGroup(scf::IfOp op, calyx::GroupOp group) {
    Operation *operation = op.getOperation();
    assert(elseGroup.count(operation) == 0 &&
           "An else group was already set for this scf::IfOp!\n");
    elseGroup[operation] = group;
  }

  calyx::GroupOp getElseGroup(scf::IfOp op) {
    auto it = elseGroup.find(op.getOperation());
    assert(it != elseGroup.end() &&
           "No else group was set for this scf::IfOp!\n");
    return it->second;
  }

  void setResultRegs(scf::IfOp op, calyx::RegisterOp reg, unsigned idx) {
    assert(resultRegs[op.getOperation()].count(idx) == 0 &&
           "A register was already registered for the given yield result.\n");
    assert(idx < op->getNumOperands());
    resultRegs[op.getOperation()][idx] = reg;
  }

  const DenseMap<unsigned, calyx::RegisterOp> &getResultRegs(scf::IfOp op) {
    return resultRegs[op.getOperation()];
  }

  calyx::RegisterOp getResultRegs(scf::IfOp op, unsigned idx) {
    auto regs = getResultRegs(op);
    auto it = regs.find(idx);
    assert(it != regs.end() && "resultReg not found");
    return it->second;
  }

private:
  DenseMap<Operation *, calyx::GroupOp> thenGroup;
  DenseMap<Operation *, calyx::GroupOp> elseGroup;
  DenseMap<Operation *, DenseMap<unsigned, calyx::RegisterOp>> resultRegs;
};

class WhileLoopLoweringStateInterface
    : calyx::LoopLoweringStateInterface<ScfWhileOp> {
public:
  SmallVector<calyx::GroupOp> getWhileLoopInitGroups(ScfWhileOp op) {
    return getLoopInitGroups(std::move(op));
  }
  calyx::GroupOp buildWhileLoopIterArgAssignments(
      OpBuilder &builder, ScfWhileOp op, calyx::ComponentOp componentOp,
      Twine uniqueSuffix, MutableArrayRef<OpOperand> ops) {
    return buildLoopIterArgAssignments(builder, std::move(op), componentOp,
                                       uniqueSuffix, ops);
  }
  void addWhileLoopIterReg(ScfWhileOp op, calyx::RegisterOp reg, unsigned idx) {
    return addLoopIterReg(std::move(op), reg, idx);
  }
  const DenseMap<unsigned, calyx::RegisterOp> &
  getWhileLoopIterRegs(ScfWhileOp op) {
    return getLoopIterRegs(std::move(op));
  }
  void setWhileLoopLatchGroup(ScfWhileOp op, calyx::GroupOp group) {
    return setLoopLatchGroup(std::move(op), group);
  }
  calyx::GroupOp getWhileLoopLatchGroup(ScfWhileOp op) {
    return getLoopLatchGroup(std::move(op));
  }
  void setWhileLoopInitGroups(ScfWhileOp op,
                              SmallVector<calyx::GroupOp> groups) {
    return setLoopInitGroups(std::move(op), std::move(groups));
  }
};

class ForLoopLoweringStateInterface
    : calyx::LoopLoweringStateInterface<ScfForOp> {
public:
  SmallVector<calyx::GroupOp> getForLoopInitGroups(ScfForOp op) {
    return getLoopInitGroups(std::move(op));
  }
  calyx::GroupOp buildForLoopIterArgAssignments(
      OpBuilder &builder, ScfForOp op, calyx::ComponentOp componentOp,
      Twine uniqueSuffix, MutableArrayRef<OpOperand> ops) {
    return buildLoopIterArgAssignments(builder, std::move(op), componentOp,
                                       uniqueSuffix, ops);
  }
  void addForLoopIterReg(ScfForOp op, calyx::RegisterOp reg, unsigned idx) {
    return addLoopIterReg(std::move(op), reg, idx);
  }
  const DenseMap<unsigned, calyx::RegisterOp> &getForLoopIterRegs(ScfForOp op) {
    return getLoopIterRegs(std::move(op));
  }
  calyx::RegisterOp getForLoopIterReg(ScfForOp op, unsigned idx) {
    return getLoopIterReg(std::move(op), idx);
  }
  void setForLoopLatchGroup(ScfForOp op, calyx::GroupOp group) {
    return setLoopLatchGroup(std::move(op), group);
  }
  calyx::GroupOp getForLoopLatchGroup(ScfForOp op) {
    return getLoopLatchGroup(std::move(op));
  }
  void setForLoopInitGroups(ScfForOp op, SmallVector<calyx::GroupOp> groups) {
    return setLoopInitGroups(std::move(op), std::move(groups));
  }
};

/// Handles the current state of lowering of a Calyx component. It is mainly
/// used as a key/value store for recording information during partial lowering,
/// which is required at later lowering passes.
class ComponentLoweringState : public calyx::ComponentLoweringStateInterface,
                               public WhileLoopLoweringStateInterface,
                               public ForLoopLoweringStateInterface,
                               public IfLoweringStateInterface,
                               public calyx::SchedulerInterface<Scheduleable> {
public:
  ComponentLoweringState(calyx::ComponentOp component)
      : calyx::ComponentLoweringStateInterface(component) {}
};

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Iterate through the operations of a source function and instantiate
/// components or primitives based on the type of the operations.
class BuildOpGroups : public calyx::FuncOpPartialLoweringPattern {
public:
  BuildOpGroups(MLIRContext *context, LogicalResult &resRef,
                calyx::PatternApplicationState &patternState,
                DenseMap<mlir::func::FuncOp, calyx::ComponentOp> &map,
                calyx::CalyxLoweringState &state,
                mlir::Pass::Option<std::string> &writeJsonOpt)
      : FuncOpPartialLoweringPattern(context, resRef, patternState, map, state),
        writeJson(writeJsonOpt) {}
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// We walk the operations of the funcOp to ensure that all def's have
    /// been visited before their uses.
    bool opBuiltSuccessfully = true;
    funcOp.walk([&](Operation *_op) {
      opBuiltSuccessfully &=
          TypeSwitch<mlir::Operation *, bool>(_op)
              .template Case<arith::ConstantOp, ReturnOp, BranchOpInterface,
                             /// SCF
                             scf::YieldOp, scf::WhileOp, scf::ForOp, scf::IfOp,
                             scf::ParallelOp, scf::ReduceOp,
                             /// memref
                             memref::AllocOp, memref::AllocaOp, memref::LoadOp,
                             memref::StoreOp, memref::GetGlobalOp,
                             /// standard arithmetic
                             AddIOp, SubIOp, CmpIOp, ShLIOp, ShRUIOp, ShRSIOp,
                             AndIOp, XOrIOp, OrIOp, ExtUIOp, ExtSIOp, TruncIOp,
                             MulIOp, DivUIOp, DivSIOp, RemUIOp, RemSIOp,
                             /// floating point
                             AddFOp, MulFOp, CmpFOp,
                             /// others
                             SelectOp, IndexCastOp, CallOp>(
                  [&](auto op) { return buildOp(rewriter, op).succeeded(); })
              .template Case<FuncOp, scf::ConditionOp>([&](auto) {
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

    if (!writeJson.empty()) {
      if (auto fileLoc = dyn_cast<mlir::FileLineColLoc>(funcOp->getLoc())) {
        std::string filename = fileLoc.getFilename().str();
        std::filesystem::path path(filename);
        std::string jsonFileName = writeJson.getValue() + ".json";
        auto outFileName = path.parent_path().append(jsonFileName);
        std::ofstream outFile(outFileName);

        if (!outFile.is_open()) {
          llvm::errs() << "Unable to open file: " << outFileName.string()
                       << " for writing\n";
          return failure();
        }
        llvm::raw_os_ostream llvmOut(outFile);
        llvm::json::OStream jsonOS(llvmOut, /*IndentSize=*/2);
        jsonOS.value(getState<ComponentLoweringState>().getExtMemData());
        jsonOS.flush();
        outFile.close();
      }
    }

    return success(opBuiltSuccessfully);
  }

private:
  mlir::Pass::Option<std::string> &writeJson;
  /// Op builder specializations.
  LogicalResult buildOp(PatternRewriter &rewriter, scf::YieldOp yieldOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        BranchOpInterface brOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        arith::ConstantOp constOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SelectOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AddIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, SubIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, MulIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, DivUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, DivSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, RemUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, RemSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AddFOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, MulFOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CmpFOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShRUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShRSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ShLIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, AndIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, OrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, XOrIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CmpIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, TruncIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ExtUIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ExtSIOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, ReturnOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, IndexCastOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::AllocOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::AllocaOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        memref::GetGlobalOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::LoadOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, memref::StoreOp op) const;
  LogicalResult buildOp(PatternRewriter &rewriter, scf::WhileOp whileOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, scf::ForOp forOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, scf::IfOp ifOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        scf::ReduceOp reduceOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter,
                        scf::ParallelOp parallelOp) const;
  LogicalResult buildOp(PatternRewriter &rewriter, CallOp callOp) const;

  /// buildLibraryOp will build a TCalyxLibOp inside a TGroupOp based on the
  /// source operation TSrcOp.
  template <typename TGroupOp, typename TCalyxLibOp, typename TSrcOp>
  LogicalResult buildLibraryOp(PatternRewriter &rewriter, TSrcOp op,
                               TypeRange srcTypes, TypeRange dstTypes) const {
    SmallVector<Type> types;
    for (Type srcType : srcTypes)
      types.push_back(calyx::toBitVector(srcType));
    for (Type dstType : dstTypes)
      types.push_back(calyx::toBitVector(dstType));

    auto calyxOp =
        getState<ComponentLoweringState>().getNewLibraryOpInstance<TCalyxLibOp>(
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
    assert(
        opInputPorts.size() == op->getNumOperands() &&
        opOutputPorts.size() == op->getNumResults() &&
        "Expected an equal number of in/out ports in the Calyx library op with "
        "respect to the number of operands/results of the source operation.");

    /// Create assignments to the inputs of the library op.
    auto group = createGroupForOp<TGroupOp>(rewriter, op);
    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    for (auto dstOp : enumerate(opInputPorts))
      rewriter.create<calyx::AssignOp>(op.getLoc(), dstOp.value(),
                                       op->getOperand(dstOp.index()));

    /// Replace the result values of the source operator with the new operator.
    for (auto res : enumerate(opOutputPorts)) {
      getState<ComponentLoweringState>().registerEvaluatingGroup(res.value(),
                                                                 group);
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
    auto groupName = getState<ComponentLoweringState>().getUniqueName(
        loweringState().blockName(block));
    return calyx::createGroup<TGroupOp>(
        rewriter, getState<ComponentLoweringState>().getComponentOp(),
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
    auto reg = createRegister(
        op.getLoc(), rewriter, getComponent(), width.getIntOrFloatBitWidth(),
        getState<ComponentLoweringState>().getUniqueName(opName));
    // Operation pipelines are not combinational, so a GroupOp is required.
    auto group = createGroupForOp<calyx::GroupOp>(rewriter, op);
    OpBuilder builder(group->getRegion(0));
    getState<ComponentLoweringState>().addBlockScheduleable(op->getBlock(),
                                                            group);

    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    rewriter.create<calyx::AssignOp>(loc, opPipe.getLeft(), op.getLhs());
    rewriter.create<calyx::AssignOp>(loc, opPipe.getRight(), op.getRhs());
    // Write the output to this register.
    rewriter.create<calyx::AssignOp>(loc, reg.getIn(), out);
    // The write enable port is high when the pipeline is done.
    rewriter.create<calyx::AssignOp>(loc, reg.getWriteEn(), opPipe.getDone());
    // Set pipelineOp to high as long as its done signal is not high.
    // This prevents the pipelineOP from executing for the cycle that we write
    // to register. To get !(pipelineOp.done) we do 1 xor pipelineOp.done
    hw::ConstantOp c1 = createConstant(loc, rewriter, getComponent(), 1, 1);
    rewriter.create<calyx::AssignOp>(
        loc, opPipe.getGo(), c1,
        comb::createOrFoldNot(group.getLoc(), opPipe.getDone(), builder));
    // The group is done when the register write is complete.
    rewriter.create<calyx::GroupDoneOp>(loc, reg.getDone());

    // Pass the result from the source operation to register holding the resullt
    // from the Calyx primitive.
    op.getResult().replaceAllUsesWith(reg.getOut());

    if (isa<calyx::AddFOpIEEE754>(opPipe)) {
      auto opFOp = cast<calyx::AddFOpIEEE754>(opPipe);
      hw::ConstantOp subOp;
      if (isa<arith::AddFOp>(op)) {
        subOp = createConstant(loc, rewriter, getComponent(), /*width=*/1,
                               /*subtract=*/0);
      } else {
        subOp = createConstant(loc, rewriter, getComponent(), /*width=*/1,
                               /*subtract=*/1);
      }
      rewriter.create<calyx::AssignOp>(loc, opFOp.getSubOp(), subOp);
    }

    // Register the values for the pipeline.
    getState<ComponentLoweringState>().registerEvaluatingGroup(out, group);
    getState<ComponentLoweringState>().registerEvaluatingGroup(opPipe.getLeft(),
                                                               group);
    getState<ComponentLoweringState>().registerEvaluatingGroup(
        opPipe.getRight(), group);

    return success();
  }

  /// Creates assignments within the provided group to the address ports of the
  /// memoryOp based on the provided addressValues.
  void assignAddressPorts(PatternRewriter &rewriter, Location loc,
                          calyx::GroupInterface group,
                          calyx::MemoryInterface memoryInterface,
                          Operation::operand_range addressValues) const {
    IRRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(group.getBody());
    auto addrPorts = memoryInterface.addrPorts();
    if (addressValues.empty()) {
      assert(
          addrPorts.size() == 1 &&
          "We expected a 1 dimensional memory of size 1 because there were no "
          "address assignment values");
      // Assign to address 1'd0 in memory.
      rewriter.create<calyx::AssignOp>(
          loc, addrPorts[0],
          createConstant(loc, rewriter, getComponent(), 1, 0));
    } else {
      assert(addrPorts.size() == addressValues.size() &&
             "Mismatch between number of address ports of the provided memory "
             "and address assignment values");
      for (auto address : enumerate(addressValues))
        rewriter.create<calyx::AssignOp>(loc, addrPorts[address.index()],
                                         address.value());
    }
  }

  calyx::RegisterOp createSignalRegister(PatternRewriter &rewriter,
                                         Value signal, bool invert,
                                         StringRef nameSuffix,
                                         calyx::CompareFOpIEEE754 calyxCmpFOp,
                                         calyx::GroupOp group) const {
    Location loc = calyxCmpFOp.getLoc();
    IntegerType one = rewriter.getI1Type();
    auto component = getComponent();
    OpBuilder builder(group->getRegion(0));
    auto reg = createRegister(
        loc, rewriter, component, 1,
        getState<ComponentLoweringState>().getUniqueName(nameSuffix));
    rewriter.create<calyx::AssignOp>(loc, reg.getWriteEn(),
                                     calyxCmpFOp.getDone());
    if (invert) {
      auto notLibOp = getState<ComponentLoweringState>()
                          .getNewLibraryOpInstance<calyx::NotLibOp>(
                              rewriter, loc, {one, one});
      rewriter.create<calyx::AssignOp>(loc, notLibOp.getIn(), signal);
      rewriter.create<calyx::AssignOp>(loc, reg.getIn(), notLibOp.getOut());
      getState<ComponentLoweringState>().registerEvaluatingGroup(
          notLibOp.getOut(), group);
    } else
      rewriter.create<calyx::AssignOp>(loc, reg.getIn(), signal);
    return reg;
  };
};

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::LoadOp loadOp) const {
  Value memref = loadOp.getMemref();
  auto memoryInterface =
      getState<ComponentLoweringState>().getMemoryInterface(memref);
  auto group = createGroupForOp<calyx::GroupOp>(rewriter, loadOp);
  assignAddressPorts(rewriter, loadOp.getLoc(), group, memoryInterface,
                     loadOp.getIndices());

  rewriter.setInsertionPointToEnd(group.getBodyBlock());

  bool needReg = true;
  Value res;
  Value regWriteEn =
      createConstant(loadOp.getLoc(), rewriter, getComponent(), 1, 1);
  if (memoryInterface.readEnOpt().has_value()) {
    auto oneI1 =
        calyx::createConstant(loadOp.getLoc(), rewriter, getComponent(), 1, 1);
    rewriter.create<calyx::AssignOp>(loadOp.getLoc(), memoryInterface.readEn(),
                                     oneI1);
    regWriteEn = memoryInterface.done();
    if (calyx::noStoresToMemory(memref) &&
        calyx::singleLoadFromMemory(memref)) {
      // Single load from memory; we do not need to write the output to a
      // register. The readData value will be held until readEn is asserted
      // again
      needReg = false;
      rewriter.create<calyx::GroupDoneOp>(loadOp.getLoc(),
                                          memoryInterface.done());
      // We refrain from replacing the loadOp result with
      // memoryInterface.readData, since multiple loadOp's need to be converted
      // to a single memory's ReadData. If this replacement is done now, we lose
      // the link between which SSA memref::LoadOp values map to which groups
      // for loading a value from the Calyx memory. At this point of lowering,
      // we keep the memref::LoadOp SSA value, and do value replacement _after_
      // control has been generated (see LateSSAReplacement). This is *vital*
      // for things such as calyx::InlineCombGroups to be able to properly track
      // which memory assignment groups belong to which accesses.
      res = loadOp.getResult();
    }
  } else if (memoryInterface.contentEnOpt().has_value()) {
    auto oneI1 =
        calyx::createConstant(loadOp.getLoc(), rewriter, getComponent(), 1, 1);
    auto zeroI1 =
        calyx::createConstant(loadOp.getLoc(), rewriter, getComponent(), 1, 0);
    rewriter.create<calyx::AssignOp>(loadOp.getLoc(),
                                     memoryInterface.contentEn(), oneI1);
    rewriter.create<calyx::AssignOp>(loadOp.getLoc(), memoryInterface.writeEn(),
                                     zeroI1);
    regWriteEn = memoryInterface.done();
    if (calyx::noStoresToMemory(memref) &&
        calyx::singleLoadFromMemory(memref)) {
      // Single load from memory; we do not need to write the output to a
      // register. The readData value will be held until contentEn is asserted
      // again
      needReg = false;
      rewriter.create<calyx::GroupDoneOp>(loadOp.getLoc(),
                                          memoryInterface.done());
      // We refrain from replacing the loadOp result with
      // memoryInterface.readData, since multiple loadOp's need to be converted
      // to a single memory's ReadData. If this replacement is done now, we lose
      // the link between which SSA memref::LoadOp values map to which groups
      // for loading a value from the Calyx memory. At this point of lowering,
      // we keep the memref::LoadOp SSA value, and do value replacement _after_
      // control has been generated (see LateSSAReplacement). This is *vital*
      // for things such as calyx::InlineCombGroups to be able to properly track
      // which memory assignment groups belong to which accesses.
      res = loadOp.getResult();
    }
  }

  if (needReg) {
    // Multiple loads from the same memory; In this case, we _may_ have a
    // structural hazard in the design we generate. To get around this, we
    // conservatively place a register in front of each load operation, and
    // replace all uses of the loaded value with the register output. Reading
    // for sequential memories will cause a read to take at least 2 cycles,
    // but it will usually be better because combinational reads on memories
    // can significantly decrease the maximum achievable frequency.
    auto reg = createRegister(
        loadOp.getLoc(), rewriter, getComponent(),
        loadOp.getMemRefType().getElementTypeBitWidth(),
        getState<ComponentLoweringState>().getUniqueName("load"));
    rewriter.setInsertionPointToEnd(group.getBodyBlock());
    rewriter.create<calyx::AssignOp>(loadOp.getLoc(), reg.getIn(),
                                     memoryInterface.readData());
    rewriter.create<calyx::AssignOp>(loadOp.getLoc(), reg.getWriteEn(),
                                     regWriteEn);
    rewriter.create<calyx::GroupDoneOp>(loadOp.getLoc(), reg.getDone());
    loadOp.getResult().replaceAllUsesWith(reg.getOut());
    res = reg.getOut();
  }

  getState<ComponentLoweringState>().registerEvaluatingGroup(res, group);
  getState<ComponentLoweringState>().addBlockScheduleable(loadOp->getBlock(),
                                                          group);
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::StoreOp storeOp) const {
  auto memoryInterface = getState<ComponentLoweringState>().getMemoryInterface(
      storeOp.getMemref());
  auto group = createGroupForOp<calyx::GroupOp>(rewriter, storeOp);

  // This is a sequential group, so register it as being scheduleable for the
  // block.
  getState<ComponentLoweringState>().addBlockScheduleable(storeOp->getBlock(),
                                                          group);
  assignAddressPorts(rewriter, storeOp.getLoc(), group, memoryInterface,
                     storeOp.getIndices());
  rewriter.setInsertionPointToEnd(group.getBodyBlock());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeData(), storeOp.getValueToStore());
  rewriter.create<calyx::AssignOp>(
      storeOp.getLoc(), memoryInterface.writeEn(),
      createConstant(storeOp.getLoc(), rewriter, getComponent(), 1, 1));
  if (memoryInterface.contentEnOpt().has_value()) {
    // If memory has content enable, it must be asserted when writing
    rewriter.create<calyx::AssignOp>(
        storeOp.getLoc(), memoryInterface.contentEn(),
        createConstant(storeOp.getLoc(), rewriter, getComponent(), 1, 1));
  }
  rewriter.create<calyx::GroupDoneOp>(storeOp.getLoc(), memoryInterface.done());

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     MulIOp mul) const {
  Location loc = mul.getLoc();
  Type width = mul.getResult().getType(), one = rewriter.getI1Type();
  auto mulPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::MultPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::MultPipeLibOp>(
      rewriter, mul, mulPipe,
      /*out=*/mulPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     DivUIOp div) const {
  Location loc = div.getLoc();
  Type width = div.getResult().getType(), one = rewriter.getI1Type();
  auto divPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::DivUPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::DivUPipeLibOp>(
      rewriter, div, divPipe,
      /*out=*/divPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     DivSIOp div) const {
  Location loc = div.getLoc();
  Type width = div.getResult().getType(), one = rewriter.getI1Type();
  auto divPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::DivSPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::DivSPipeLibOp>(
      rewriter, div, divPipe,
      /*out=*/divPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     RemUIOp rem) const {
  Location loc = rem.getLoc();
  Type width = rem.getResult().getType(), one = rewriter.getI1Type();
  auto remPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::RemUPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::RemUPipeLibOp>(
      rewriter, rem, remPipe,
      /*out=*/remPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     RemSIOp rem) const {
  Location loc = rem.getLoc();
  Type width = rem.getResult().getType(), one = rewriter.getI1Type();
  auto remPipe =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::RemSPipeLibOp>(
              rewriter, loc, {one, one, one, width, width, width, one});
  return buildLibraryBinaryPipeOp<calyx::RemSPipeLibOp>(
      rewriter, rem, remPipe,
      /*out=*/remPipe.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AddFOp addf) const {
  Location loc = addf.getLoc();
  IntegerType one = rewriter.getI1Type(), three = rewriter.getIntegerType(3),
              five = rewriter.getIntegerType(5),
              width = rewriter.getIntegerType(
                  addf.getType().getIntOrFloatBitWidth());
  auto addFOp =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::AddFOpIEEE754>(
              rewriter, loc,
              {one, one, one, one, one, width, width, three, width, five, one});
  return buildLibraryBinaryPipeOp<calyx::AddFOpIEEE754>(rewriter, addf, addFOp,
                                                        addFOp.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     MulFOp mulf) const {
  Location loc = mulf.getLoc();
  IntegerType one = rewriter.getI1Type(), three = rewriter.getIntegerType(3),
              five = rewriter.getIntegerType(5),
              width = rewriter.getIntegerType(
                  mulf.getType().getIntOrFloatBitWidth());
  auto mulFOp =
      getState<ComponentLoweringState>()
          .getNewLibraryOpInstance<calyx::MulFOpIEEE754>(
              rewriter, loc,
              {one, one, one, one, width, width, three, width, five, one});
  return buildLibraryBinaryPipeOp<calyx::MulFOpIEEE754>(rewriter, mulf, mulFOp,
                                                        mulFOp.getOut());
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CmpFOp cmpf) const {
  Location loc = cmpf.getLoc();
  IntegerType one = rewriter.getI1Type(), five = rewriter.getIntegerType(5),
              width = rewriter.getIntegerType(
                  cmpf.getLhs().getType().getIntOrFloatBitWidth());
  auto calyxCmpFOp = getState<ComponentLoweringState>()
                         .getNewLibraryOpInstance<calyx::CompareFOpIEEE754>(
                             rewriter, loc,
                             {one, one, one, width, width, one, one, one, one,
                              one, five, one});
  hw::ConstantOp c0 = createConstant(loc, rewriter, getComponent(), 1, 0);
  hw::ConstantOp c1 = createConstant(loc, rewriter, getComponent(), 1, 1);
  rewriter.setInsertionPointToStart(getComponent().getBodyBlock());

  using calyx::PredicateInfo;
  using CombLogic = PredicateInfo::CombLogic;
  using Port = PredicateInfo::InputPorts::Port;
  PredicateInfo info = calyx::getPredicateInfo(cmpf.getPredicate());
  if (info.logic == CombLogic::None) {
    if (cmpf.getPredicate() == CmpFPredicate::AlwaysTrue) {
      cmpf.getResult().replaceAllUsesWith(c1);
      return success();
    }

    if (cmpf.getPredicate() == CmpFPredicate::AlwaysFalse) {
      cmpf.getResult().replaceAllUsesWith(c0);
      return success();
    }
  }

  // General case
  StringRef opName = cmpf.getOperationName().split(".").second;
  auto reg =
      createRegister(loc, rewriter, getComponent(), 1,
                     getState<ComponentLoweringState>().getUniqueName(opName));

  // Operation pipelines are not combinational, so a GroupOp is required.
  auto group = createGroupForOp<calyx::GroupOp>(rewriter, cmpf);
  OpBuilder builder(group->getRegion(0));
  getState<ComponentLoweringState>().addBlockScheduleable(cmpf->getBlock(),
                                                          group);

  rewriter.setInsertionPointToEnd(group.getBodyBlock());
  rewriter.create<calyx::AssignOp>(loc, calyxCmpFOp.getLeft(), cmpf.getLhs());
  rewriter.create<calyx::AssignOp>(loc, calyxCmpFOp.getRight(), cmpf.getRhs());

  bool signalingFlag = false;
  switch (cmpf.getPredicate()) {
  case CmpFPredicate::UGT:
  case CmpFPredicate::UGE:
  case CmpFPredicate::ULT:
  case CmpFPredicate::ULE:
  case CmpFPredicate::OGT:
  case CmpFPredicate::OGE:
  case CmpFPredicate::OLT:
  case CmpFPredicate::OLE:
    signalingFlag = true;
    break;
  case CmpFPredicate::UEQ:
  case CmpFPredicate::UNE:
  case CmpFPredicate::OEQ:
  case CmpFPredicate::ONE:
  case CmpFPredicate::UNO:
  case CmpFPredicate::ORD:
  case CmpFPredicate::AlwaysTrue:
  case CmpFPredicate::AlwaysFalse:
    signalingFlag = false;
    break;
  }

  // The IEEE Standard mandates that equality comparisons ordinarily are quiet,
  // while inequality comparisons ordinarily are signaling.
  rewriter.create<calyx::AssignOp>(loc, calyxCmpFOp.getSignaling(),
                                   signalingFlag ? c1 : c0);

  // Prepare signals and create registers
  SmallVector<calyx::RegisterOp> inputRegs;
  for (const auto &input : info.inputPorts) {
    Value signal;
    switch (input.port) {
    case Port::Eq: {
      signal = calyxCmpFOp.getEq();
      break;
    }
    case Port::Gt: {
      signal = calyxCmpFOp.getGt();
      break;
    }
    case Port::Lt: {
      signal = calyxCmpFOp.getLt();
      break;
    }
    case Port::Unordered: {
      signal = calyxCmpFOp.getUnordered();
      break;
    }
    }
    std::string nameSuffix =
        (input.port == PredicateInfo::InputPorts::Port::Unordered)
            ? "unordered_port"
            : "compare_port";
    auto signalReg = createSignalRegister(rewriter, signal, input.invert,
                                          nameSuffix, calyxCmpFOp, group);
    inputRegs.push_back(signalReg);
  }

  // Create the output logical operation
  Value outputValue, doneValue;
  switch (info.logic) {
  case CombLogic::None: {
    // it's guaranteed to be either ORD or UNO
    outputValue = inputRegs[0].getOut();
    doneValue = inputRegs[0].getOut();
    break;
  }
  case CombLogic::And: {
    auto outputLibOp = getState<ComponentLoweringState>()
                           .getNewLibraryOpInstance<calyx::AndLibOp>(
                               rewriter, loc, {one, one, one});
    rewriter.create<calyx::AssignOp>(loc, outputLibOp.getLeft(),
                                     inputRegs[0].getOut());
    rewriter.create<calyx::AssignOp>(loc, outputLibOp.getRight(),
                                     inputRegs[1].getOut());

    outputValue = outputLibOp.getOut();
    break;
  }
  case CombLogic::Or: {
    auto outputLibOp = getState<ComponentLoweringState>()
                           .getNewLibraryOpInstance<calyx::OrLibOp>(
                               rewriter, loc, {one, one, one});
    rewriter.create<calyx::AssignOp>(loc, outputLibOp.getLeft(),
                                     inputRegs[0].getOut());
    rewriter.create<calyx::AssignOp>(loc, outputLibOp.getRight(),
                                     inputRegs[1].getOut());

    outputValue = outputLibOp.getOut();
    break;
  }
  }

  if (info.logic != CombLogic::None) {
    auto doneLibOp = getState<ComponentLoweringState>()
                         .getNewLibraryOpInstance<calyx::AndLibOp>(
                             rewriter, loc, {one, one, one});
    rewriter.create<calyx::AssignOp>(loc, doneLibOp.getLeft(),
                                     inputRegs[0].getDone());
    rewriter.create<calyx::AssignOp>(loc, doneLibOp.getRight(),
                                     inputRegs[1].getDone());
    doneValue = doneLibOp.getOut();
  }

  // Write to the output register
  rewriter.create<calyx::AssignOp>(loc, reg.getIn(), outputValue);
  rewriter.create<calyx::AssignOp>(loc, reg.getWriteEn(), doneValue);

  // Set the go and done signal
  rewriter.create<calyx::AssignOp>(
      loc, calyxCmpFOp.getGo(), c1,
      comb::createOrFoldNot(loc, calyxCmpFOp.getDone(), builder));
  rewriter.create<calyx::GroupDoneOp>(loc, reg.getDone());

  cmpf.getResult().replaceAllUsesWith(reg.getOut());

  // Register evaluating groups
  getState<ComponentLoweringState>().registerEvaluatingGroup(outputValue,
                                                             group);
  getState<ComponentLoweringState>().registerEvaluatingGroup(doneValue, group);
  getState<ComponentLoweringState>().registerEvaluatingGroup(
      calyxCmpFOp.getLeft(), group);
  getState<ComponentLoweringState>().registerEvaluatingGroup(
      calyxCmpFOp.getRight(), group);

  return success();
}

template <typename TAllocOp>
static LogicalResult buildAllocOp(ComponentLoweringState &componentState,
                                  PatternRewriter &rewriter, TAllocOp allocOp) {
  rewriter.setInsertionPointToStart(
      componentState.getComponentOp().getBodyBlock());
  MemRefType memtype = allocOp.getType();
  SmallVector<int64_t> addrSizes;
  SmallVector<int64_t> sizes;
  for (int64_t dim : memtype.getShape()) {
    sizes.push_back(dim);
    addrSizes.push_back(calyx::handleZeroWidth(dim));
  }
  // If memref has no size (e.g., memref<i32>) create a 1 dimensional memory of
  // size 1.
  if (sizes.empty() && addrSizes.empty()) {
    sizes.push_back(1);
    addrSizes.push_back(1);
  }
  auto memoryOp = rewriter.create<calyx::SeqMemoryOp>(
      allocOp.getLoc(), componentState.getUniqueName("mem"),
      memtype.getElementType().getIntOrFloatBitWidth(), sizes, addrSizes);

  // Externalize memories conditionally (only in the top-level component because
  // Calyx compiler requires it as a well-formness check).
  memoryOp->setAttr("external",
                    IntegerAttr::get(rewriter.getI1Type(), llvm::APInt(1, 1)));
  componentState.registerMemoryInterface(allocOp.getResult(),
                                         calyx::MemoryInterface(memoryOp));

  unsigned elmTyBitWidth = memtype.getElementTypeBitWidth();
  assert(elmTyBitWidth <= 64 && "element bitwidth should not exceed 64");
  bool isFloat = !memtype.getElementType().isInteger();

  auto shape = allocOp.getType().getShape();
  int totalSize =
      std::reduce(shape.begin(), shape.end(), 1, std::multiplies<int>());
  // The `totalSize <= 1` check is a hack to:
  // https://github.com/llvm/circt/pull/2661, where a multi-dimensional memory
  // whose size in some dimension equals 1, e.g. memref<1x1x1x1xi32>, will be
  // collapsed to `memref<1xi32>` with `totalSize == 1`. While the above case is
  // a trivial fix, Calyx expects 1-dimensional memories in general:
  // https://github.com/calyxir/calyx/issues/907
  if (!(shape.size() <= 1 || totalSize <= 1)) {
    allocOp.emitError("input memory dimension must be empty or one.");
    return failure();
  }

  std::vector<uint64_t> flattenedVals(totalSize, 0);
  if (isa<memref::GetGlobalOp>(allocOp)) {
    auto getGlobalOp = cast<memref::GetGlobalOp>(allocOp);
    auto *symbolTableOp =
        getGlobalOp->template getParentWithTrait<mlir::OpTrait::SymbolTable>();
    auto globalOp = dyn_cast_or_null<memref::GlobalOp>(
        SymbolTable::lookupSymbolIn(symbolTableOp, getGlobalOp.getNameAttr()));
    // Flatten the values in the attribute
    auto cstAttr = llvm::dyn_cast_or_null<DenseElementsAttr>(
        globalOp.getConstantInitValue());
    int sizeCount = 0;
    for (auto attr : cstAttr.template getValues<Attribute>()) {
      assert((isa<mlir::FloatAttr, mlir::IntegerAttr>(attr)) &&
             "memory attributes must be float or int");
      if (auto fltAttr = dyn_cast<mlir::FloatAttr>(attr)) {
        flattenedVals[sizeCount++] =
            bit_cast<uint64_t>(fltAttr.getValueAsDouble());
      } else {
        auto intAttr = dyn_cast<mlir::IntegerAttr>(attr);
        APInt value = intAttr.getValue();
        flattenedVals[sizeCount++] = *value.getRawData();
      }
    }

    rewriter.eraseOp(globalOp);
  }

  llvm::json::Array result;
  result.reserve(std::max(static_cast<int>(shape.size()), 1));

  Type elemType = memtype.getElementType();
  bool isSigned =
      !elemType.isSignlessInteger() && !elemType.isUnsignedInteger();
  for (uint64_t bitValue : flattenedVals) {
    llvm::json::Value value = 0;
    if (isFloat) {
      // We cast to `double` and let downstream calyx to deal with the actual
      // value's precision handling.
      value = bit_cast<double>(bitValue);
    } else {
      APInt apInt(/*numBits=*/elmTyBitWidth, bitValue, isSigned,
                  /*implicitTrunc=*/true);
      // The conditional ternary operation will cause the `value` to interpret
      // the underlying data as unsigned regardless `isSigned` or not.
      if (isSigned)
        value = static_cast<int64_t>(apInt.getSExtValue());
      else
        value = apInt.getZExtValue();
    }
    result.push_back(std::move(value));
  }

  componentState.setDataField(memoryOp.getName(), result);
  std::string numType =
      memtype.getElementType().isInteger() ? "bitnum" : "ieee754_float";
  componentState.setFormat(memoryOp.getName(), numType, isSigned,
                           elmTyBitWidth);

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocOp allocOp) const {
  return buildAllocOp(getState<ComponentLoweringState>(), rewriter, allocOp);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::AllocaOp allocOp) const {
  return buildAllocOp(getState<ComponentLoweringState>(), rewriter, allocOp);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     memref::GetGlobalOp getGlobalOp) const {
  return buildAllocOp(getState<ComponentLoweringState>(), rewriter,
                      getGlobalOp);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::YieldOp yieldOp) const {
  if (yieldOp.getOperands().empty()) {
    // If yield operands are empty, we assume we have a for loop.
    auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
    assert(forOp && "Empty yieldOps should only be located within ForOps");
    ScfForOp forOpInterface(forOp);

    // Get the ForLoop's Induction Register.
    auto inductionReg =
        getState<ComponentLoweringState>().getForLoopIterReg(forOpInterface, 0);

    Type regWidth = inductionReg.getOut().getType();
    // Adder should have same width as the inductionReg.
    SmallVector<Type> types(3, regWidth);
    auto addOp = getState<ComponentLoweringState>()
                     .getNewLibraryOpInstance<calyx::AddLibOp>(
                         rewriter, forOp.getLoc(), types);

    auto directions = addOp.portDirections();
    // For an add operation, we expect two input ports and one output port
    SmallVector<Value, 2> opInputPorts;
    Value opOutputPort;
    for (auto dir : enumerate(directions)) {
      switch (dir.value()) {
      case calyx::Direction::Input: {
        opInputPorts.push_back(addOp.getResult(dir.index()));
        break;
      }
      case calyx::Direction::Output: {
        opOutputPort = addOp.getResult(dir.index());
        break;
      }
      }
    }

    // "Latch Group" increments inductionReg by forLoop's step value.
    calyx::ComponentOp componentOp =
        getState<ComponentLoweringState>().getComponentOp();
    SmallVector<StringRef, 4> groupIdentifier = {
        "incr", getState<ComponentLoweringState>().getUniqueName(forOp),
        "induction", "var"};
    auto groupOp = calyx::createGroup<calyx::GroupOp>(
        rewriter, componentOp, forOp.getLoc(),
        llvm::join(groupIdentifier, "_"));
    rewriter.setInsertionPointToEnd(groupOp.getBodyBlock());

    // Assign inductionReg.out to the left port of the adder.
    Value leftOp = opInputPorts.front();
    rewriter.create<calyx::AssignOp>(forOp.getLoc(), leftOp,
                                     inductionReg.getOut());
    // Assign forOp.getConstantStep to the right port of the adder.
    Value rightOp = opInputPorts.back();
    rewriter.create<calyx::AssignOp>(
        forOp.getLoc(), rightOp,
        createConstant(forOp->getLoc(), rewriter, componentOp,
                       regWidth.getIntOrFloatBitWidth(),
                       forOp.getConstantStep().value().getSExtValue()));
    // Assign adder's output port to inductionReg.
    buildAssignmentsForRegisterWrite(rewriter, groupOp, componentOp,
                                     inductionReg, opOutputPort);
    // Set group as For Loop's "latch" group.
    getState<ComponentLoweringState>().setForLoopLatchGroup(forOpInterface,
                                                            groupOp);
    getState<ComponentLoweringState>().registerEvaluatingGroup(opOutputPort,
                                                               groupOp);
    return success();
  }
  // If yieldOp for a for loop is not empty, then we do not transform for loop.
  if (dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
    return yieldOp.getOperation()->emitError()
           << "Currently do not support non-empty yield operations inside for "
              "loops. Run --scf-for-to-while before running --scf-to-calyx.";
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(yieldOp->getParentOp())) {
    ScfWhileOp whileOpInterface(whileOp);

    auto assignGroup =
        getState<ComponentLoweringState>().buildWhileLoopIterArgAssignments(
            rewriter, whileOpInterface,
            getState<ComponentLoweringState>().getComponentOp(),
            getState<ComponentLoweringState>().getUniqueName(whileOp) +
                "_latch",
            yieldOp->getOpOperands());
    getState<ComponentLoweringState>().setWhileLoopLatchGroup(whileOpInterface,
                                                              assignGroup);
    return success();
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
    auto resultRegs = getState<ComponentLoweringState>().getResultRegs(ifOp);

    if (yieldOp->getParentRegion() == &ifOp.getThenRegion()) {
      auto thenGroup = getState<ComponentLoweringState>().getThenGroup(ifOp);
      for (auto op : enumerate(yieldOp.getOperands())) {
        auto resultReg =
            getState<ComponentLoweringState>().getResultRegs(ifOp, op.index());
        buildAssignmentsForRegisterWrite(
            rewriter, thenGroup,
            getState<ComponentLoweringState>().getComponentOp(), resultReg,
            op.value());
        getState<ComponentLoweringState>().registerEvaluatingGroup(
            ifOp.getResult(op.index()), thenGroup);
      }
    }

    if (!ifOp.getElseRegion().empty() &&
        (yieldOp->getParentRegion() == &ifOp.getElseRegion())) {
      auto elseGroup = getState<ComponentLoweringState>().getElseGroup(ifOp);
      for (auto op : enumerate(yieldOp.getOperands())) {
        auto resultReg =
            getState<ComponentLoweringState>().getResultRegs(ifOp, op.index());
        buildAssignmentsForRegisterWrite(
            rewriter, elseGroup,
            getState<ComponentLoweringState>().getComponentOp(), resultReg,
            op.value());
        getState<ComponentLoweringState>().registerEvaluatingGroup(
            ifOp.getResult(op.index()), elseGroup);
      }
    }
  }
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     BranchOpInterface brOp) const {
  /// Branch argument passing group creation
  /// Branch operands are passed through registers. In BuildBasicBlockRegs we
  /// created registers for all branch arguments of each block. We now
  /// create groups for assigning values to these registers.
  Block *srcBlock = brOp->getBlock();
  for (auto succBlock : enumerate(brOp->getSuccessors())) {
    auto succOperands = brOp.getSuccessorOperands(succBlock.index());
    if (succOperands.empty())
      continue;
    // Create operand passing group
    std::string groupName = loweringState().blockName(srcBlock) + "_to_" +
                            loweringState().blockName(succBlock.value());
    auto groupOp = calyx::createGroup<calyx::GroupOp>(rewriter, getComponent(),
                                                      brOp.getLoc(), groupName);
    // Fetch block argument registers associated with the basic block
    auto dstBlockArgRegs =
        getState<ComponentLoweringState>().getBlockArgRegs(succBlock.value());
    // Create register assignment for each block argument
    for (auto arg : enumerate(succOperands.getForwardedOperands())) {
      auto reg = dstBlockArgRegs[arg.index()];
      calyx::buildAssignmentsForRegisterWrite(
          rewriter, groupOp,
          getState<ComponentLoweringState>().getComponentOp(), reg,
          arg.value());
    }
    /// Register the group as a block argument group, to be executed
    /// when entering the successor block from this block (srcBlock).
    getState<ComponentLoweringState>().addBlockArgGroup(
        srcBlock, succBlock.value(), groupOp);
  }
  return success();
}

/// For each return statement, we create a new group for assigning to the
/// previously created return value registers.
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ReturnOp retOp) const {
  if (retOp.getNumOperands() == 0)
    return success();

  std::string groupName =
      getState<ComponentLoweringState>().getUniqueName("ret_assign");
  auto groupOp = calyx::createGroup<calyx::GroupOp>(rewriter, getComponent(),
                                                    retOp.getLoc(), groupName);
  for (auto op : enumerate(retOp.getOperands())) {
    auto reg = getState<ComponentLoweringState>().getReturnReg(op.index());
    calyx::buildAssignmentsForRegisterWrite(
        rewriter, groupOp, getState<ComponentLoweringState>().getComponentOp(),
        reg, op.value());
  }
  /// Schedule group for execution for when executing the return op block.
  getState<ComponentLoweringState>().addBlockScheduleable(retOp->getBlock(),
                                                          groupOp);
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     arith::ConstantOp constOp) const {
  if (isa<IntegerType>(constOp.getType())) {
    /// Move constant operations to the compOp body as hw::ConstantOp's.
    APInt value;
    calyx::matchConstantOp(constOp, value);
    auto hwConstOp =
        rewriter.replaceOpWithNewOp<hw::ConstantOp>(constOp, value);
    hwConstOp->moveAfter(getComponent().getBodyBlock(),
                         getComponent().getBodyBlock()->begin());
  } else {
    std::string name = getState<ComponentLoweringState>().getUniqueName("cst");
    auto floatAttr = cast<FloatAttr>(constOp.getValueAttr());
    auto intType =
        rewriter.getIntegerType(floatAttr.getType().getIntOrFloatBitWidth());
    auto calyxConstOp = rewriter.create<calyx::ConstantOp>(
        constOp.getLoc(), name, floatAttr, intType);
    calyxConstOp->moveAfter(getComponent().getBodyBlock(),
                            getComponent().getBodyBlock()->begin());
    rewriter.replaceAllUsesWith(constOp, calyxConstOp.getOut());
  }

  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AddIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AddLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     SubIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SubLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShRUIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::RshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShRSIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SrshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ShLIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::LshLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     AndIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::AndLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     OrIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::OrLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     XOrIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::XorLibOp>(rewriter, op);
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     SelectOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::MuxLibOp>(rewriter, op);
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CmpIOp op) const {
  switch (op.getPredicate()) {
  case CmpIPredicate::eq:
    return buildLibraryOp<calyx::CombGroupOp, calyx::EqLibOp>(rewriter, op);
  case CmpIPredicate::ne:
    return buildLibraryOp<calyx::CombGroupOp, calyx::NeqLibOp>(rewriter, op);
  case CmpIPredicate::uge:
    return buildLibraryOp<calyx::CombGroupOp, calyx::GeLibOp>(rewriter, op);
  case CmpIPredicate::ult:
    return buildLibraryOp<calyx::CombGroupOp, calyx::LtLibOp>(rewriter, op);
  case CmpIPredicate::ugt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::GtLibOp>(rewriter, op);
  case CmpIPredicate::ule:
    return buildLibraryOp<calyx::CombGroupOp, calyx::LeLibOp>(rewriter, op);
  case CmpIPredicate::sge:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgeLibOp>(rewriter, op);
  case CmpIPredicate::slt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SltLibOp>(rewriter, op);
  case CmpIPredicate::sgt:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SgtLibOp>(rewriter, op);
  case CmpIPredicate::sle:
    return buildLibraryOp<calyx::CombGroupOp, calyx::SleLibOp>(rewriter, op);
  }
  llvm_unreachable("unsupported comparison predicate");
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     TruncIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}
LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ExtUIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     ExtSIOp op) const {
  return buildLibraryOp<calyx::CombGroupOp, calyx::ExtSILibOp>(
      rewriter, op, {op.getOperand().getType()}, {op.getType()});
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     IndexCastOp op) const {
  Type sourceType = calyx::normalizeType(rewriter, op.getOperand().getType());
  Type targetType = calyx::normalizeType(rewriter, op.getResult().getType());
  unsigned targetBits = targetType.getIntOrFloatBitWidth();
  unsigned sourceBits = sourceType.getIntOrFloatBitWidth();
  LogicalResult res = success();

  if (targetBits == sourceBits) {
    /// Drop the index cast and replace uses of the target value with the source
    /// value.
    op.getResult().replaceAllUsesWith(op.getOperand());
  } else {
    /// pad/slice the source operand.
    if (sourceBits > targetBits)
      res = buildLibraryOp<calyx::CombGroupOp, calyx::SliceLibOp>(
          rewriter, op, {sourceType}, {targetType});
    else
      res = buildLibraryOp<calyx::CombGroupOp, calyx::PadLibOp>(
          rewriter, op, {sourceType}, {targetType});
  }
  rewriter.eraseOp(op);
  return res;
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::WhileOp whileOp) const {
  // Only need to add the whileOp to the BlockSchedulables scheduler interface.
  // Everything else was handled in the `BuildWhileGroups` pattern.
  ScfWhileOp scfWhileOp(whileOp);
  getState<ComponentLoweringState>().addBlockScheduleable(
      whileOp.getOperation()->getBlock(), WhileScheduleable{scfWhileOp});
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::ForOp forOp) const {
  // Only need to add the forOp to the BlockSchedulables scheduler interface.
  // Everything else was handled in the `BuildForGroups` pattern.
  ScfForOp scfForOp(forOp);
  // If we cannot compute the trip count of the for loop, then we should
  // emit an error saying to use --scf-for-to-while
  std::optional<uint64_t> bound = scfForOp.getBound();
  if (!bound.has_value()) {
    return scfForOp.getOperation()->emitError()
           << "Loop bound not statically known. Should "
              "transform into while loop using `--scf-for-to-while` before "
              "running --lower-scf-to-calyx.";
  }
  getState<ComponentLoweringState>().addBlockScheduleable(
      forOp.getOperation()->getBlock(), ForScheduleable{
                                            scfForOp,
                                            bound.value(),
                                        });
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::IfOp ifOp) const {
  getState<ComponentLoweringState>().addBlockScheduleable(
      ifOp.getOperation()->getBlock(), IfScheduleable{ifOp});
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::ReduceOp reduceOp) const {
  // we don't handle reduce operation and simply return success for now since
  // BuildParGroups would have already emitted an error and exited early
  // if a reduce operation was encountered.
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     scf::ParallelOp parOp) const {
  getState<ComponentLoweringState>().addBlockScheduleable(
      parOp.getOperation()->getBlock(), ParScheduleable{parOp});
  return success();
}

LogicalResult BuildOpGroups::buildOp(PatternRewriter &rewriter,
                                     CallOp callOp) const {
  std::string instanceName = calyx::getInstanceName(callOp);
  calyx::InstanceOp instanceOp =
      getState<ComponentLoweringState>().getInstance(instanceName);
  SmallVector<Value, 4> outputPorts;
  auto portInfos = instanceOp.getReferencedComponent().getPortInfo();
  for (auto [idx, portInfo] : enumerate(portInfos)) {
    if (portInfo.direction == calyx::Direction::Output)
      outputPorts.push_back(instanceOp.getResult(idx));
  }

  // Replacing a CallOp results in the out port of the instance.
  for (auto [idx, result] : llvm::enumerate(callOp.getResults()))
    rewriter.replaceAllUsesWith(result, outputPorts[idx]);

  // CallScheduleanle requires an instance, while CallOp can be used to get the
  // input ports.
  getState<ComponentLoweringState>().addBlockScheduleable(
      callOp.getOperation()->getBlock(), CallScheduleable{instanceOp, callOp});
  return success();
}

/// Inlines Calyx ExecuteRegionOp operations within their parent blocks.
/// An execution region op (ERO) is inlined by:
///  i  : add a sink basic block for all yield operations inside the
///       ERO to jump to
///  ii : Rewrite scf.yield calls inside the ERO to branch to the sink block
///  iii: inline the ERO region
/// TODO(#1850) evaluate the usefulness of this lowering pattern.
class InlineExecuteRegionOpPattern
    : public OpRewritePattern<scf::ExecuteRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ExecuteRegionOp execOp,
                                PatternRewriter &rewriter) const override {
    /// Determine type of "yield" operations inside the ERO.
    TypeRange yieldTypes = execOp.getResultTypes();

    /// Create sink basic block and rewrite uses of yield results to sink block
    /// arguments.
    rewriter.setInsertionPointAfter(execOp);
    auto *sinkBlock = rewriter.splitBlock(
        execOp->getBlock(),
        execOp.getOperation()->getIterator()->getNextNode()->getIterator());
    sinkBlock->addArguments(
        yieldTypes,
        SmallVector<Location, 4>(yieldTypes.size(), rewriter.getUnknownLoc()));
    for (auto res : enumerate(execOp.getResults()))
      res.value().replaceAllUsesWith(sinkBlock->getArgument(res.index()));

    /// Rewrite yield calls as branches.
    for (auto yieldOp :
         make_early_inc_range(execOp.getRegion().getOps<scf::YieldOp>())) {
      rewriter.setInsertionPointAfter(yieldOp);
      rewriter.replaceOpWithNewOp<BranchOp>(yieldOp, sinkBlock,
                                            yieldOp.getOperands());
    }

    /// Inline the regionOp.
    auto *preBlock = execOp->getBlock();
    auto *execOpEntryBlock = &execOp.getRegion().front();
    auto *postBlock = execOp->getBlock()->splitBlock(execOp);
    rewriter.inlineRegionBefore(execOp.getRegion(), postBlock);
    rewriter.mergeBlocks(postBlock, preBlock);
    rewriter.eraseOp(execOp);

    /// Finally, erase the unused entry block of the execOp region.
    rewriter.mergeBlocks(execOpEntryBlock, preBlock);

    return success();
  }
};

/// Creates a new Calyx component for each FuncOp in the program.
struct FuncOpConversion : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    /// Maintain a mapping between funcOp input arguments and the port index
    /// which the argument will eventually map to.
    DenseMap<Value, unsigned> funcOpArgRewrites;

    /// Maintain a mapping between funcOp output indexes and the component
    /// output port index which the return value will eventually map to.
    DenseMap<unsigned, unsigned> funcOpResultMapping;

    /// Maintain a mapping between an external memory argument (identified by a
    /// memref) and eventual component input- and output port indices that will
    /// map to the memory ports. The pair denotes the start index of the memory
    /// ports in the in- and output ports of the component. Ports are expected
    /// to be ordered in the same manner as they are added by
    /// calyx::appendPortsForExternalMemref.
    DenseMap<Value, std::pair<unsigned, unsigned>> extMemoryCompPortIndices;

    /// Create I/O ports. Maintain separate in/out port vectors to determine
    /// which port index each function argument will eventually map to.
    SmallVector<calyx::PortInfo> inPorts, outPorts;
    FunctionType funcType = funcOp.getFunctionType();
    for (auto arg : enumerate(funcOp.getArguments())) {
      if (!isa<MemRefType>(arg.value().getType())) {
        /// Single-port arguments
        std::string inName;
        if (auto portNameAttr = funcOp.getArgAttrOfType<StringAttr>(
                arg.index(), scfToCalyx::sPortNameAttr))
          inName = portNameAttr.str();
        else
          inName = "in" + std::to_string(arg.index());
        funcOpArgRewrites[arg.value()] = inPorts.size();
        inPorts.push_back(calyx::PortInfo{
            rewriter.getStringAttr(inName),
            calyx::normalizeType(rewriter, arg.value().getType()),
            calyx::Direction::Input,
            DictionaryAttr::get(rewriter.getContext(), {})});
      }
    }
    for (auto res : enumerate(funcType.getResults())) {
      std::string resName;
      if (auto portNameAttr = funcOp.getResultAttrOfType<StringAttr>(
              res.index(), scfToCalyx::sPortNameAttr))
        resName = portNameAttr.str();
      else
        resName = "out" + std::to_string(res.index());
      funcOpResultMapping[res.index()] = outPorts.size();

      outPorts.push_back(calyx::PortInfo{
          rewriter.getStringAttr(resName),
          calyx::normalizeType(rewriter, res.value()), calyx::Direction::Output,
          DictionaryAttr::get(rewriter.getContext(), {})});
    }

    /// We've now recorded all necessary indices. Merge in- and output ports
    /// and add the required mandatory component ports.
    auto ports = inPorts;
    llvm::append_range(ports, outPorts);
    calyx::addMandatoryComponentPorts(rewriter, ports);

    /// Create a calyx::ComponentOp corresponding to the to-be-lowered function.
    auto compOp = rewriter.create<calyx::ComponentOp>(
        funcOp.getLoc(), rewriter.getStringAttr(funcOp.getSymName()), ports);

    std::string funcName = "func_" + funcOp.getSymName().str();
    rewriter.modifyOpInPlace(funcOp, [&]() { funcOp.setSymName(funcName); });

    /// Mark this component as the toplevel if it's the top-level function of
    /// the module.
    if (compOp.getName() == loweringState().getTopLevelFunction())
      compOp->setAttr("toplevel", rewriter.getUnitAttr());

    /// Store the function-to-component mapping.
    functionMapping[funcOp] = compOp;
    auto *compState = loweringState().getState<ComponentLoweringState>(compOp);
    compState->setFuncOpResultMapping(funcOpResultMapping);

    unsigned extMemCounter = 0;
    for (auto arg : enumerate(funcOp.getArguments())) {
      if (isa<MemRefType>(arg.value().getType())) {
        std::string memName =
            llvm::join_items("_", "arg_mem", std::to_string(extMemCounter++));

        rewriter.setInsertionPointToStart(compOp.getBodyBlock());
        MemRefType memtype = cast<MemRefType>(arg.value().getType());
        SmallVector<int64_t> addrSizes;
        SmallVector<int64_t> sizes;
        for (int64_t dim : memtype.getShape()) {
          sizes.push_back(dim);
          addrSizes.push_back(calyx::handleZeroWidth(dim));
        }
        if (sizes.empty() && addrSizes.empty()) {
          sizes.push_back(1);
          addrSizes.push_back(1);
        }
        auto memOp = rewriter.create<calyx::SeqMemoryOp>(
            funcOp.getLoc(), memName,
            memtype.getElementType().getIntOrFloatBitWidth(), sizes, addrSizes);
        // we don't set the memory to "external", which implies it's a reference

        compState->registerMemoryInterface(arg.value(),
                                           calyx::MemoryInterface(memOp));
      }
    }

    /// Rewrite funcOp SSA argument values to the CompOp arguments.
    for (auto &mapping : funcOpArgRewrites)
      mapping.getFirst().replaceAllUsesWith(
          compOp.getArgument(mapping.getSecond()));

    return success();
  }
};

/// In BuildWhileGroups, a register is created for each iteration argumenet of
/// the while op. These registers are then written to on the while op
/// terminating yield operation alongside before executing the whileOp in the
/// schedule, to set the initial values of the argument registers.
class BuildWhileGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](Operation *op) {
      // Only work on ops that support the ScfWhileOp.
      if (!isa<scf::WhileOp>(op))
        return WalkResult::advance();

      auto scfWhileOp = cast<scf::WhileOp>(op);
      ScfWhileOp whileOp(scfWhileOp);

      getState<ComponentLoweringState>().setUniqueName(whileOp.getOperation(),
                                                       "while");

      /// Check for do-while loops.
      /// TODO(mortbopet) can we support these? for now, do not support loops
      /// where iterargs are changed in the 'before' region. scf.WhileOp also
      /// has support for different types of iter_args and return args which we
      /// also do not support; iter_args and while return values are placed in
      /// the same registers.
      for (auto barg :
           enumerate(scfWhileOp.getBefore().front().getArguments())) {
        auto condOp = scfWhileOp.getConditionOp().getArgs()[barg.index()];
        if (barg.value() != condOp) {
          res = whileOp.getOperation()->emitError()
                << loweringState().irName(barg.value())
                << " != " << loweringState().irName(condOp)
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
      for (auto arg : enumerate(whileOp.getBodyArgs())) {
        std::string name = getState<ComponentLoweringState>()
                               .getUniqueName(whileOp.getOperation())
                               .str() +
                           "_arg" + std::to_string(arg.index());
        auto reg =
            createRegister(arg.value().getLoc(), rewriter, getComponent(),
                           arg.value().getType().getIntOrFloatBitWidth(), name);
        getState<ComponentLoweringState>().addWhileLoopIterReg(whileOp, reg,
                                                               arg.index());
        arg.value().replaceAllUsesWith(reg.getOut());

        /// Also replace uses in the "before" region of the while loop
        whileOp.getConditionBlock()
            ->getArgument(arg.index())
            .replaceAllUsesWith(reg.getOut());
      }

      /// Create iter args initial value assignment group(s), one per register.
      SmallVector<calyx::GroupOp> initGroups;
      auto numOperands = whileOp.getOperation()->getNumOperands();
      for (size_t i = 0; i < numOperands; ++i) {
        auto initGroupOp =
            getState<ComponentLoweringState>().buildWhileLoopIterArgAssignments(
                rewriter, whileOp,
                getState<ComponentLoweringState>().getComponentOp(),
                getState<ComponentLoweringState>().getUniqueName(
                    whileOp.getOperation()) +
                    "_init_" + std::to_string(i),
                whileOp.getOperation()->getOpOperand(i));
        initGroups.push_back(initGroupOp);
      }

      getState<ComponentLoweringState>().setWhileLoopInitGroups(whileOp,
                                                                initGroups);

      return WalkResult::advance();
    });
    return res;
  }
};

/// In BuildForGroups, a register is created for the iteration argument of
/// the for op. This register is then initialized to the lowerBound of the for
/// loop in a group that executes the for loop.
class BuildForGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](Operation *op) {
      // Only work on ops that support the ScfForOp.
      if (!isa<scf::ForOp>(op))
        return WalkResult::advance();

      auto scfForOp = cast<scf::ForOp>(op);
      ScfForOp forOp(scfForOp);

      getState<ComponentLoweringState>().setUniqueName(forOp.getOperation(),
                                                       "for");

      // Create a register for the InductionVar, and set that Register as the
      // only IterReg for the For Loop
      auto inductionVar = forOp.getOperation().getInductionVar();
      SmallVector<std::string, 3> inductionVarIdentifiers = {
          getState<ComponentLoweringState>()
              .getUniqueName(forOp.getOperation())
              .str(),
          "induction", "var"};
      std::string name = llvm::join(inductionVarIdentifiers, "_");
      auto reg =
          createRegister(inductionVar.getLoc(), rewriter, getComponent(),
                         inductionVar.getType().getIntOrFloatBitWidth(), name);
      getState<ComponentLoweringState>().addForLoopIterReg(forOp, reg, 0);
      inductionVar.replaceAllUsesWith(reg.getOut());

      // Create InitGroup that sets the InductionVar to LowerBound
      calyx::ComponentOp componentOp =
          getState<ComponentLoweringState>().getComponentOp();
      SmallVector<calyx::GroupOp> initGroups;
      SmallVector<std::string, 4> groupIdentifiers = {
          "init",
          getState<ComponentLoweringState>()
              .getUniqueName(forOp.getOperation())
              .str(),
          "induction", "var"};
      std::string groupName = llvm::join(groupIdentifiers, "_");
      auto groupOp = calyx::createGroup<calyx::GroupOp>(
          rewriter, componentOp, forOp.getLoc(), groupName);
      buildAssignmentsForRegisterWrite(rewriter, groupOp, componentOp, reg,
                                       forOp.getOperation().getLowerBound());
      initGroups.push_back(groupOp);
      getState<ComponentLoweringState>().setForLoopInitGroups(forOp,
                                                              initGroups);

      return WalkResult::advance();
    });
    return res;
  }
};

class BuildIfGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    LogicalResult res = success();
    funcOp.walk([&](Operation *op) {
      if (!isa<scf::IfOp>(op))
        return WalkResult::advance();

      auto scfIfOp = cast<scf::IfOp>(op);

      calyx::ComponentOp componentOp =
          getState<ComponentLoweringState>().getComponentOp();

      std::string thenGroupName =
          getState<ComponentLoweringState>().getUniqueName("then_br");
      auto thenGroupOp = calyx::createGroup<calyx::GroupOp>(
          rewriter, componentOp, scfIfOp.getLoc(), thenGroupName);
      getState<ComponentLoweringState>().setThenGroup(scfIfOp, thenGroupOp);

      if (!scfIfOp.getElseRegion().empty()) {
        std::string elseGroupName =
            getState<ComponentLoweringState>().getUniqueName("else_br");
        auto elseGroupOp = calyx::createGroup<calyx::GroupOp>(
            rewriter, componentOp, scfIfOp.getLoc(), elseGroupName);
        getState<ComponentLoweringState>().setElseGroup(scfIfOp, elseGroupOp);
      }

      for (auto ifOpRes : scfIfOp.getResults()) {
        auto reg = createRegister(
            scfIfOp.getLoc(), rewriter, getComponent(),
            ifOpRes.getType().getIntOrFloatBitWidth(),
            getState<ComponentLoweringState>().getUniqueName("if_res"));
        getState<ComponentLoweringState>().setResultRegs(
            scfIfOp, reg, ifOpRes.getResultNumber());
      }

      return WalkResult::advance();
    });
    return res;
  }
};

class BuildParGroups : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    WalkResult walkResult = funcOp.walk([&](scf::ParallelOp scfParOp) {
      if (!scfParOp.getResults().empty()) {
        scfParOp.emitError(
            "Reduce operations in scf.parallel is not supported yet");
        return WalkResult::interrupt();
      }

      if (failed(partialEval(rewriter, scfParOp)))
        return WalkResult::interrupt();

      return WalkResult::advance();
    });

    return walkResult.wasInterrupted() ? failure() : success();
  }

private:
  // Partially evaluate/pre-compute all blocks being executed in parallel by
  // statically generate loop indices combinations
  LogicalResult partialEval(PatternRewriter &rewriter,
                            scf::ParallelOp scfParOp) const {
    assert(scfParOp.getLoopSteps() && "Parallel loop must have steps");
    auto *body = scfParOp.getBody();
    auto parOpIVs = scfParOp.getInductionVars();
    auto steps = scfParOp.getStep();
    auto lowerBounds = scfParOp.getLowerBound();
    auto upperBounds = scfParOp.getUpperBound();
    rewriter.setInsertionPointAfter(scfParOp);
    scf::ParallelOp newParOp = scfParOp.cloneWithoutRegions();
    auto loc = newParOp.getLoc();
    rewriter.insert(newParOp);
    OpBuilder insideBuilder(newParOp);
    Block *currBlock = nullptr;
    auto &region = newParOp.getRegion();

    // extract lower bounds, upper bounds, and steps as integer index values
    SmallVector<int64_t> lbVals, ubVals, stepVals;
    for (auto lb : lowerBounds) {
      auto lbOp = lb.getDefiningOp<arith::ConstantIndexOp>();
      assert(lbOp &&
             "Lower bound must be a statically computable constant index");
      lbVals.push_back(lbOp.value());
    }
    for (auto ub : upperBounds) {
      auto ubOp = ub.getDefiningOp<arith::ConstantIndexOp>();
      assert(ubOp &&
             "Upper bound must be a statically computable constant index");
      ubVals.push_back(ubOp.value());
    }
    for (auto step : steps) {
      auto stepOp = step.getDefiningOp<arith::ConstantIndexOp>();
      assert(stepOp && "Step must be a statically computable constant index");
      stepVals.push_back(stepOp.value());
    }

    // Initialize indices with lower bounds
    SmallVector<int64_t> indices = lbVals;

    while (true) {
      // Each iteration starts with a fresh mapping, so each new block’s
      // argument of a region-based operation (such as `scf.for`) get re-mapped
      // independently.
      IRMapping operandMap;

      // Create a new block in the region for the current combination of indices
      currBlock = &region.emplaceBlock();
      insideBuilder.setInsertionPointToEnd(currBlock);

      // Map induction variables to constant indices
      for (unsigned i = 0; i < indices.size(); ++i) {
        Value ivConstant =
            insideBuilder.create<arith::ConstantIndexOp>(loc, indices[i]);
        operandMap.map(parOpIVs[i], ivConstant);
      }

      for (auto it = body->begin(); it != std::prev(body->end()); ++it)
        insideBuilder.clone(*it, operandMap);

      // Increment indices using `step`
      bool done = false;
      for (int dim = indices.size() - 1; dim >= 0; --dim) {
        indices[dim] += stepVals[dim];
        if (indices[dim] < ubVals[dim])
          break;
        indices[dim] = lbVals[dim];
        if (dim == 0)
          // All combinations have been generated
          done = true;
      }
      if (done)
        break;
    }

    rewriter.setInsertionPointToEnd(newParOp.getBody());
    rewriter.create<scf::ReduceOp>(newParOp.getLoc());

    rewriter.replaceOp(scfParOp, newParOp);
    return success();
  }
};

/// Builds a control schedule by traversing the CFG of the function and
/// associating this with the previously created groups.
/// For simplicity, the generated control flow is expanded for all possible
/// paths in the input DAG. This elaborated control flow is later reduced in
/// the runControlFlowSimplification passes.
class BuildControl : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    auto *entryBlock = &funcOp.getBlocks().front();
    rewriter.setInsertionPointToStart(
        getComponent().getControlOp().getBodyBlock());
    auto topLevelSeqOp = rewriter.create<calyx::SeqOp>(funcOp.getLoc());
    DenseSet<Block *> path;
    return buildCFGControl(path, rewriter, topLevelSeqOp.getBodyBlock(),
                           nullptr, entryBlock);
  }

private:
  /// Sequentially schedules the groups that registered themselves with
  /// 'block'.
  LogicalResult scheduleBasicBlock(PatternRewriter &rewriter,
                                   const DenseSet<Block *> &path,
                                   mlir::Block *parentCtrlBlock,
                                   mlir::Block *block) const {
    auto compBlockScheduleables =
        getState<ComponentLoweringState>().getBlockScheduleables(block);
    auto loc = block->front().getLoc();

    if (compBlockScheduleables.size() > 1 &&
        !isa<scf::ParallelOp>(block->getParentOp())) {
      auto seqOp = rewriter.create<calyx::SeqOp>(loc);
      parentCtrlBlock = seqOp.getBodyBlock();
    }

    for (auto &group : compBlockScheduleables) {
      rewriter.setInsertionPointToEnd(parentCtrlBlock);
      if (auto groupPtr = std::get_if<calyx::GroupOp>(&group); groupPtr) {
        rewriter.create<calyx::EnableOp>(groupPtr->getLoc(),
                                         groupPtr->getSymName());
      } else if (auto whileSchedPtr = std::get_if<WhileScheduleable>(&group);
                 whileSchedPtr) {
        auto &whileOp = whileSchedPtr->whileOp;

        auto whileCtrlOp = buildWhileCtrlOp(
            whileOp,
            getState<ComponentLoweringState>().getWhileLoopInitGroups(whileOp),
            rewriter);
        rewriter.setInsertionPointToEnd(whileCtrlOp.getBodyBlock());
        auto whileBodyOp =
            rewriter.create<calyx::SeqOp>(whileOp.getOperation()->getLoc());
        auto *whileBodyOpBlock = whileBodyOp.getBodyBlock();

        /// Only schedule the 'after' block. The 'before' block is
        /// implicitly scheduled when evaluating the while condition.
        if (LogicalResult result =
                buildCFGControl(path, rewriter, whileBodyOpBlock, block,
                                whileOp.getBodyBlock());
            result.failed())
          return result;

        // Insert loop-latch at the end of the while group
        rewriter.setInsertionPointToEnd(whileBodyOpBlock);
        calyx::GroupOp whileLatchGroup =
            getState<ComponentLoweringState>().getWhileLoopLatchGroup(whileOp);
        rewriter.create<calyx::EnableOp>(whileLatchGroup.getLoc(),
                                         whileLatchGroup.getName());
      } else if (auto *parSchedPtr = std::get_if<ParScheduleable>(&group)) {
        auto parOp = parSchedPtr->parOp;
        auto calyxParOp = rewriter.create<calyx::ParOp>(parOp.getLoc());
        for (auto &innerBlock : parOp.getRegion().getBlocks()) {
          rewriter.setInsertionPointToEnd(calyxParOp.getBodyBlock());
          auto seqOp = rewriter.create<calyx::SeqOp>(parOp.getLoc());
          rewriter.setInsertionPointToEnd(seqOp.getBodyBlock());
          if (LogicalResult res = scheduleBasicBlock(
                  rewriter, path, seqOp.getBodyBlock(), &innerBlock);
              res.failed())
            return res;
        }
      } else if (auto *forSchedPtr = std::get_if<ForScheduleable>(&group);
                 forSchedPtr) {
        auto forOp = forSchedPtr->forOp;

        auto forCtrlOp = buildForCtrlOp(
            forOp,
            getState<ComponentLoweringState>().getForLoopInitGroups(forOp),
            forSchedPtr->bound, rewriter);
        rewriter.setInsertionPointToEnd(forCtrlOp.getBodyBlock());
        auto forBodyOp =
            rewriter.create<calyx::SeqOp>(forOp.getOperation()->getLoc());
        auto *forBodyOpBlock = forBodyOp.getBodyBlock();

        // Schedule the body of the for loop.
        if (LogicalResult res = buildCFGControl(path, rewriter, forBodyOpBlock,
                                                block, forOp.getBodyBlock());
            res.failed())
          return res;

        // Insert loop-latch at the end of the while group.
        rewriter.setInsertionPointToEnd(forBodyOpBlock);
        calyx::GroupOp forLatchGroup =
            getState<ComponentLoweringState>().getForLoopLatchGroup(forOp);
        rewriter.create<calyx::EnableOp>(forLatchGroup.getLoc(),
                                         forLatchGroup.getName());
      } else if (auto *ifSchedPtr = std::get_if<IfScheduleable>(&group);
                 ifSchedPtr) {
        auto ifOp = ifSchedPtr->ifOp;

        Location loc = ifOp->getLoc();

        auto cond = ifOp.getCondition();
        auto condGroup = getState<ComponentLoweringState>()
                             .getEvaluatingGroup<calyx::CombGroupOp>(cond);

        auto symbolAttr = FlatSymbolRefAttr::get(
            StringAttr::get(getContext(), condGroup.getSymName()));

        bool initElse = !ifOp.getElseRegion().empty();
        auto ifCtrlOp = rewriter.create<calyx::IfOp>(
            loc, cond, symbolAttr, /*initializeElseBody=*/initElse);

        rewriter.setInsertionPointToEnd(ifCtrlOp.getBodyBlock());

        auto thenSeqOp =
            rewriter.create<calyx::SeqOp>(ifOp.getThenRegion().getLoc());
        auto *thenSeqOpBlock = thenSeqOp.getBodyBlock();

        auto *thenBlock = &ifOp.getThenRegion().front();
        LogicalResult res = buildCFGControl(path, rewriter, thenSeqOpBlock,
                                            /*preBlock=*/block, thenBlock);
        if (res.failed())
          return res;

        rewriter.setInsertionPointToEnd(thenSeqOpBlock);
        calyx::GroupOp thenGroup =
            getState<ComponentLoweringState>().getThenGroup(ifOp);
        rewriter.create<calyx::EnableOp>(thenGroup.getLoc(),
                                         thenGroup.getName());

        if (!ifOp.getElseRegion().empty()) {
          rewriter.setInsertionPointToEnd(ifCtrlOp.getElseBody());

          auto elseSeqOp =
              rewriter.create<calyx::SeqOp>(ifOp.getElseRegion().getLoc());
          auto *elseSeqOpBlock = elseSeqOp.getBodyBlock();

          auto *elseBlock = &ifOp.getElseRegion().front();
          res = buildCFGControl(path, rewriter, elseSeqOpBlock,
                                /*preBlock=*/block, elseBlock);
          if (res.failed())
            return res;

          rewriter.setInsertionPointToEnd(elseSeqOpBlock);
          calyx::GroupOp elseGroup =
              getState<ComponentLoweringState>().getElseGroup(ifOp);
          rewriter.create<calyx::EnableOp>(elseGroup.getLoc(),
                                           elseGroup.getName());
        }
      } else if (auto *callSchedPtr = std::get_if<CallScheduleable>(&group)) {
        auto instanceOp = callSchedPtr->instanceOp;
        OpBuilder::InsertionGuard g(rewriter);
        auto callBody = rewriter.create<calyx::SeqOp>(instanceOp.getLoc());
        rewriter.setInsertionPointToStart(callBody.getBodyBlock());
        std::string initGroupName = "init_" + instanceOp.getSymName().str();
        rewriter.create<calyx::EnableOp>(instanceOp.getLoc(), initGroupName);

        auto callee = callSchedPtr->callOp.getCallee();
        auto *calleeOp = SymbolTable::lookupNearestSymbolFrom(
            callSchedPtr->callOp.getOperation()->getParentOp(),
            StringAttr::get(rewriter.getContext(), "func_" + callee.str()));
        FuncOp calleeFunc = dyn_cast_or_null<FuncOp>(calleeOp);

        auto instanceOpComp =
            llvm::cast<calyx::ComponentOp>(instanceOp.getReferencedComponent());
        auto *instanceOpLoweringState =
            loweringState().getState(instanceOpComp);

        SmallVector<Value, 4> instancePorts;
        SmallVector<Value, 4> inputPorts;
        SmallVector<Attribute, 4> refCells;
        for (auto operandEnum : enumerate(callSchedPtr->callOp.getOperands())) {
          auto operand = operandEnum.value();
          auto index = operandEnum.index();
          if (!isa<MemRefType>(operand.getType())) {
            inputPorts.push_back(operand);
            continue;
          }

          auto memOpName = getState<ComponentLoweringState>()
                               .getMemoryInterface(operand)
                               .memName();
          auto memOpNameAttr =
              SymbolRefAttr::get(rewriter.getContext(), memOpName);
          Value argI = calleeFunc.getArgument(index);
          if (isa<MemRefType>(argI.getType())) {
            NamedAttrList namedAttrList;
            namedAttrList.append(
                rewriter.getStringAttr(
                    instanceOpLoweringState->getMemoryInterface(argI)
                        .memName()),
                memOpNameAttr);
            refCells.push_back(
                DictionaryAttr::get(rewriter.getContext(), namedAttrList));
          }
        }
        llvm::copy(instanceOp.getResults().take_front(inputPorts.size()),
                   std::back_inserter(instancePorts));

        ArrayAttr refCellsAttr =
            ArrayAttr::get(rewriter.getContext(), refCells);

        rewriter.create<calyx::InvokeOp>(
            instanceOp.getLoc(), instanceOp.getSymName(), instancePorts,
            inputPorts, refCellsAttr, ArrayAttr::get(rewriter.getContext(), {}),
            ArrayAttr::get(rewriter.getContext(), {}));
      } else
        llvm_unreachable("Unknown scheduleable");
    }
    return success();
  }

  /// Schedules a block by inserting a branch argument assignment block (if any)
  /// before recursing into the scheduling of the block innards.
  /// Blocks 'from' and 'to' refer to blocks in the source program.
  /// parentCtrlBlock refers to the control block wherein control operations are
  /// to be inserted.
  LogicalResult schedulePath(PatternRewriter &rewriter,
                             const DenseSet<Block *> &path, Location loc,
                             Block *from, Block *to,
                             Block *parentCtrlBlock) const {
    /// Schedule any registered block arguments to be executed before the body
    /// of the branch.
    rewriter.setInsertionPointToEnd(parentCtrlBlock);
    auto preSeqOp = rewriter.create<calyx::SeqOp>(loc);
    rewriter.setInsertionPointToEnd(preSeqOp.getBodyBlock());
    for (auto barg :
         getState<ComponentLoweringState>().getBlockArgGroups(from, to))
      rewriter.create<calyx::EnableOp>(barg.getLoc(), barg.getSymName());

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
    LogicalResult bbSchedResult =
        scheduleBasicBlock(rewriter, path, parentCtrlBlock, block);
    if (bbSchedResult.failed())
      return bbSchedResult;

    path.insert(block);
    auto successors = block->getSuccessors();
    auto nSuccessors = successors.size();
    if (nSuccessors > 0) {
      auto brOp = dyn_cast<BranchOpInterface>(block->getTerminator());
      assert(brOp);
      if (nSuccessors > 1) {
        /// TODO(mortbopet): we could choose to support ie. std.switch, but it
        /// would probably be easier to just require it to be lowered
        /// beforehand.
        assert(nSuccessors == 2 &&
               "only conditional branches supported for now...");
        /// Wrap each branch inside an if/else.
        auto cond = brOp->getOperand(0);
        auto condGroup = getState<ComponentLoweringState>()
                             .getEvaluatingGroup<calyx::CombGroupOp>(cond);
        auto symbolAttr = FlatSymbolRefAttr::get(
            StringAttr::get(getContext(), condGroup.getSymName()));

        auto ifOp = rewriter.create<calyx::IfOp>(
            brOp->getLoc(), cond, symbolAttr, /*initializeElseBody=*/true);
        rewriter.setInsertionPointToStart(ifOp.getThenBody());
        auto thenSeqOp = rewriter.create<calyx::SeqOp>(brOp.getLoc());
        rewriter.setInsertionPointToStart(ifOp.getElseBody());
        auto elseSeqOp = rewriter.create<calyx::SeqOp>(brOp.getLoc());

        bool trueBrSchedSuccess =
            schedulePath(rewriter, path, brOp.getLoc(), block, successors[0],
                         thenSeqOp.getBodyBlock())
                .succeeded();
        bool falseBrSchedSuccess = true;
        if (trueBrSchedSuccess) {
          falseBrSchedSuccess =
              schedulePath(rewriter, path, brOp.getLoc(), block, successors[1],
                           elseSeqOp.getBodyBlock())
                  .succeeded();
        }

        return success(trueBrSchedSuccess && falseBrSchedSuccess);
      } else {
        /// Schedule sequentially within the current parent control block.
        return schedulePath(rewriter, path, brOp.getLoc(), block,
                            successors.front(), parentCtrlBlock);
      }
    }
    return success();
  }

  // Insert a Par of initGroups at Location loc. Used as helper for
  // `buildWhileCtrlOp` and `buildForCtrlOp`.
  void
  insertParInitGroups(PatternRewriter &rewriter, Location loc,
                      const SmallVector<calyx::GroupOp> &initGroups) const {
    PatternRewriter::InsertionGuard g(rewriter);
    auto parOp = rewriter.create<calyx::ParOp>(loc);
    rewriter.setInsertionPointToStart(parOp.getBodyBlock());
    for (calyx::GroupOp group : initGroups)
      rewriter.create<calyx::EnableOp>(group.getLoc(), group.getName());
  }

  calyx::WhileOp buildWhileCtrlOp(ScfWhileOp whileOp,
                                  SmallVector<calyx::GroupOp> initGroups,
                                  PatternRewriter &rewriter) const {
    Location loc = whileOp.getLoc();
    /// Insert while iter arg initialization group(s). Emit a
    /// parallel group to assign one or more registers all at once.
    insertParInitGroups(rewriter, loc, initGroups);

    /// Insert the while op itself.
    auto cond = whileOp.getConditionValue();
    auto condGroup = getState<ComponentLoweringState>()
                         .getEvaluatingGroup<calyx::CombGroupOp>(cond);
    auto symbolAttr = FlatSymbolRefAttr::get(
        StringAttr::get(getContext(), condGroup.getSymName()));
    return rewriter.create<calyx::WhileOp>(loc, cond, symbolAttr);
  }

  calyx::RepeatOp buildForCtrlOp(ScfForOp forOp,
                                 SmallVector<calyx::GroupOp> const &initGroups,
                                 uint64_t bound,
                                 PatternRewriter &rewriter) const {
    Location loc = forOp.getLoc();
    // Insert for iter arg initialization group(s). Emit a
    // parallel group to assign one or more registers all at once.
    insertParInitGroups(rewriter, loc, initGroups);

    // Insert the repeatOp that corresponds to the For loop.
    return rewriter.create<calyx::RepeatOp>(loc, bound);
  }
};

/// LateSSAReplacement contains various functions for replacing SSA values that
/// were not replaced during op construction.
class LateSSAReplacement : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult partiallyLowerFuncToComp(FuncOp funcOp,
                                         PatternRewriter &) const override {
    funcOp.walk([&](scf::IfOp op) {
      for (auto res : getState<ComponentLoweringState>().getResultRegs(op))
        op.getOperation()->getResults()[res.first].replaceAllUsesWith(
            res.second.getOut());
    });

    funcOp.walk([&](scf::WhileOp op) {
      /// The yielded values returned from the while op will be present in the
      /// iterargs registers post execution of the loop.
      /// This is done now, as opposed to during BuildWhileGroups since if the
      /// results of the whileOp were replaced before
      /// BuildOpGroups/BuildControl, the whileOp would get dead-code
      /// eliminated.
      ScfWhileOp whileOp(op);
      for (auto res :
           getState<ComponentLoweringState>().getWhileLoopIterRegs(whileOp))
        whileOp.getOperation()->getResults()[res.first].replaceAllUsesWith(
            res.second.getOut());
    });

    funcOp.walk([&](memref::LoadOp loadOp) {
      if (calyx::singleLoadFromMemory(loadOp)) {
        /// In buildOpGroups we did not replace loadOp's results, to ensure a
        /// link between evaluating groups (which fix the input addresses of a
        /// memory op) and a readData result. Now, we may replace these SSA
        /// values with their memoryOp readData output.
        loadOp.getResult().replaceAllUsesWith(
            getState<ComponentLoweringState>()
                .getMemoryInterface(loadOp.getMemref())
                .readData());
      }
    });

    return success();
  }
};

/// Erases FuncOp operations.
class CleanupFuncOps : public calyx::FuncOpPartialLoweringPattern {
  using FuncOpPartialLoweringPattern::FuncOpPartialLoweringPattern;

  LogicalResult matchAndRewrite(FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(funcOp);
    return success();
  }

  LogicalResult
  partiallyLowerFuncToComp(FuncOp funcOp,
                           PatternRewriter &rewriter) const override {
    return success();
  }
};

} // namespace scftocalyx

namespace {

using namespace circt::scftocalyx;

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//
class SCFToCalyxPass : public circt::impl::SCFToCalyxBase<SCFToCalyxPass> {
public:
  SCFToCalyxPass()
      : SCFToCalyxBase<SCFToCalyxPass>(), partialPatternRes(success()) {}
  void runOnOperation() override;

  LogicalResult setTopLevelFunction(mlir::ModuleOp moduleOp,
                                    std::string &topLevelFunction) {
    if (!topLevelFunctionOpt.empty()) {
      if (SymbolTable::lookupSymbolIn(moduleOp, topLevelFunctionOpt) ==
          nullptr) {
        moduleOp.emitError() << "Top level function '" << topLevelFunctionOpt
                             << "' not found in module.";
        return failure();
      }
      topLevelFunction = topLevelFunctionOpt;
    } else {
      /// No top level function set; infer top level if the module only contains
      /// a single function, else, throw error.
      auto funcOps = moduleOp.getOps<FuncOp>();
      if (std::distance(funcOps.begin(), funcOps.end()) == 1)
        topLevelFunction = (*funcOps.begin()).getSymName().str();
      else {
        moduleOp.emitError()
            << "Module contains multiple functions, but no top level "
               "function was set. Please see --top-level-function";
        return failure();
      }
    }

    return createOptNewTopLevelFn(moduleOp, topLevelFunction);
  }

  struct LoweringPattern {
    enum class Strategy { Once, Greedy };
    RewritePatternSet pattern;
    Strategy strategy;
  };

  //// Labels the entry point of a Calyx program.
  /// Furthermore, this function performs validation on the input function,
  /// to ensure that we've implemented the capabilities necessary to convert
  /// it.
  LogicalResult labelEntryPoint(StringRef topLevelFunction) {
    // Program legalization - the partial conversion driver will not run
    // unless some pattern is provided - provide a dummy pattern.
    struct DummyPattern : public OpRewritePattern<mlir::ModuleOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(mlir::ModuleOp,
                                    PatternRewriter &) const override {
        return failure();
      }
    };

    ConversionTarget target(getContext());
    target.addLegalDialect<calyx::CalyxDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<hw::HWDialect>();
    target.addIllegalDialect<comb::CombDialect>();

    // Only accept std operations which we've added lowerings for
    target.addIllegalDialect<FuncDialect>();
    target.addIllegalDialect<ArithDialect>();
    target.addLegalOp<AddIOp, SelectOp, SubIOp, CmpIOp, ShLIOp, ShRUIOp,
                      ShRSIOp, AndIOp, XOrIOp, OrIOp, ExtUIOp, TruncIOp,
                      CondBranchOp, BranchOp, MulIOp, DivUIOp, DivSIOp, RemUIOp,
                      RemSIOp, ReturnOp, arith::ConstantOp, IndexCastOp, FuncOp,
                      ExtSIOp, CallOp, AddFOp, MulFOp, CmpFOp>();

    RewritePatternSet legalizePatterns(&getContext());
    legalizePatterns.add<DummyPattern>(&getContext());
    DenseSet<Operation *> legalizedOps;
    if (applyPartialConversion(getOperation(), target,
                               std::move(legalizePatterns))
            .failed())
      return failure();

    // Program conversion
    return calyx::applyModuleOpConversion(getOperation(), topLevelFunction);
  }

  /// 'Once' patterns are expected to take an additional LogicalResult&
  /// argument, to forward their result state (greedyPatternRewriteDriver
  /// results are skipped for Once patterns).
  template <typename TPattern, typename... PatternArgs>
  void addOncePattern(SmallVectorImpl<LoweringPattern> &patterns,
                      PatternArgs &&...args) {
    RewritePatternSet ps(&getContext());
    ps.add<TPattern>(&getContext(), partialPatternRes, args...);
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
    assert(pattern.getNativePatterns().size() == 1 &&
           "Should only apply 1 partial lowering pattern at once");

    // During component creation, the function body is inlined into the
    // component body for further processing. However, proper control flow
    // will only be established later in the conversion process, so ensure
    // that rewriter optimizations (especially DCE) are disabled.
    GreedyRewriteConfig config;
    config.enableRegionSimplification =
        mlir::GreedySimplifyRegionLevel::Disabled;
    if (runOnce)
      config.maxIterations = 1;

    /// Can't return applyPatternsGreedily. Root isn't
    /// necessarily erased so it will always return failed(). Instead,
    /// forward the 'succeeded' value from PartialLoweringPatternBase.
    (void)applyPatternsGreedily(getOperation(), std::move(pattern), config);
    return partialPatternRes;
  }

private:
  LogicalResult partialPatternRes;
  std::shared_ptr<calyx::CalyxLoweringState> loweringState = nullptr;

  /// Creates a new new top-level function based on `baseName`.
  FuncOp createNewTopLevelFn(ModuleOp moduleOp, std::string &baseName) {
    std::string newName = "main";

    if (auto *existingMainOp = SymbolTable::lookupSymbolIn(moduleOp, newName)) {
      auto existingMainFunc = dyn_cast<FuncOp>(existingMainOp);
      if (existingMainFunc == nullptr) {
        moduleOp.emitError() << "Symbol 'main' exists but is not a function";
        return nullptr;
      }
      unsigned counter = 0;
      std::string newOldName = baseName;
      while (SymbolTable::lookupSymbolIn(moduleOp, newOldName))
        newOldName = llvm::join_items("_", baseName, std::to_string(++counter));
      existingMainFunc.setName(newOldName);
      if (baseName == "main")
        baseName = newOldName;
    }

    // Create the new "main" function
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToStart(moduleOp.getBody());

    FunctionType funcType = builder.getFunctionType({}, {});

    if (auto newFunc =
            builder.create<FuncOp>(moduleOp.getLoc(), newName, funcType))
      return newFunc;

    return nullptr;
  }

  /// Insert a call from the newly created top-level function/`caller` to the
  /// old top-level function/`callee`; and create `memref.alloc`s inside the new
  /// top-level function for arguments with `memref` types and for the
  /// `memref.alloc`s inside `callee`.
  void insertCallFromNewTopLevel(OpBuilder &builder, FuncOp caller,
                                 FuncOp callee) {
    if (caller.getBody().empty()) {
      caller.addEntryBlock();
    }

    Block *callerEntryBlock = &caller.getBody().front();
    builder.setInsertionPointToStart(callerEntryBlock);

    // For those non-memref arguments passing to the original top-level
    // function, we need to copy them to the new top-level function.
    SmallVector<Type, 4> nonMemRefCalleeArgTypes;
    for (auto arg : callee.getArguments()) {
      if (!isa<MemRefType>(arg.getType())) {
        nonMemRefCalleeArgTypes.push_back(arg.getType());
      }
    }

    for (Type type : nonMemRefCalleeArgTypes) {
      callerEntryBlock->addArgument(type, caller.getLoc());
    }

    FunctionType callerFnType = caller.getFunctionType();
    SmallVector<Type, 4> updatedCallerArgTypes(
        caller.getFunctionType().getInputs());
    updatedCallerArgTypes.append(nonMemRefCalleeArgTypes.begin(),
                                 nonMemRefCalleeArgTypes.end());
    caller.setType(FunctionType::get(caller.getContext(), updatedCallerArgTypes,
                                     callerFnType.getResults()));

    Block *calleeFnBody = &callee.getBody().front();
    unsigned originalCalleeArgNum = callee.getArguments().size();

    SmallVector<Value, 4> extraMemRefArgs;
    SmallVector<Type, 4> extraMemRefArgTypes;
    SmallVector<Value, 4> extraMemRefOperands;
    SmallVector<Operation *, 4> opsToModify;
    for (auto &op : callee.getBody().getOps()) {
      if (isa<memref::AllocaOp, memref::AllocOp, memref::GetGlobalOp>(op))
        opsToModify.push_back(&op);
    }

    // Replace `alloc`/`getGlobal` in the original top-level with new
    // corresponding operations in the new top-level.
    builder.setInsertionPointToEnd(callerEntryBlock);
    for (auto *op : opsToModify) {
      // TODO (https://github.com/llvm/circt/issues/7764)
      Value newOpRes;
      TypeSwitch<Operation *>(op)
          .Case<memref::AllocaOp>([&](memref::AllocaOp allocaOp) {
            newOpRes = builder.create<memref::AllocaOp>(callee.getLoc(),
                                                        allocaOp.getType());
          })
          .Case<memref::AllocOp>([&](memref::AllocOp allocOp) {
            newOpRes = builder.create<memref::AllocOp>(callee.getLoc(),
                                                       allocOp.getType());
          })
          .Case<memref::GetGlobalOp>([&](memref::GetGlobalOp getGlobalOp) {
            newOpRes = builder.create<memref::GetGlobalOp>(
                caller.getLoc(), getGlobalOp.getType(), getGlobalOp.getName());
          })
          .Default([&](Operation *defaultOp) {
            llvm::report_fatal_error("Unsupported operation in TypeSwitch");
          });
      extraMemRefOperands.push_back(newOpRes);

      calleeFnBody->addArgument(newOpRes.getType(), callee.getLoc());
      BlockArgument newBodyArg = calleeFnBody->getArguments().back();
      op->getResult(0).replaceAllUsesWith(newBodyArg);
      op->erase();
      extraMemRefArgs.push_back(newBodyArg);
      extraMemRefArgTypes.push_back(newBodyArg.getType());
    }

    SmallVector<Type, 4> updatedCalleeArgTypes(
        callee.getFunctionType().getInputs());
    updatedCalleeArgTypes.append(extraMemRefArgTypes.begin(),
                                 extraMemRefArgTypes.end());
    callee.setType(FunctionType::get(callee.getContext(), updatedCalleeArgTypes,
                                     callee.getFunctionType().getResults()));

    unsigned otherArgsCount = 0;
    SmallVector<Value, 4> calleeArgFnOperands;
    builder.setInsertionPointToStart(callerEntryBlock);
    for (auto arg : callee.getArguments().take_front(originalCalleeArgNum)) {
      if (isa<MemRefType>(arg.getType())) {
        auto memrefType = cast<MemRefType>(arg.getType());
        auto allocOp =
            builder.create<memref::AllocOp>(callee.getLoc(), memrefType);
        calleeArgFnOperands.push_back(allocOp);
      } else {
        auto callerArg = callerEntryBlock->getArgument(otherArgsCount++);
        calleeArgFnOperands.push_back(callerArg);
      }
    }

    SmallVector<Value, 4> fnOperands;
    fnOperands.append(calleeArgFnOperands.begin(), calleeArgFnOperands.end());
    fnOperands.append(extraMemRefOperands.begin(), extraMemRefOperands.end());
    auto calleeName =
        SymbolRefAttr::get(builder.getContext(), callee.getSymName());
    auto resultTypes = callee.getResultTypes();

    builder.setInsertionPointToEnd(callerEntryBlock);
    builder.create<CallOp>(caller.getLoc(), calleeName, resultTypes,
                           fnOperands);
    builder.create<ReturnOp>(caller.getLoc());
  }

  /// Conditionally creates an optional new top-level function; and inserts a
  /// call from the new top-level function to the old top-level function if we
  /// did create one
  LogicalResult createOptNewTopLevelFn(ModuleOp moduleOp,
                                       std::string &topLevelFunction) {
    auto hasMemrefArguments = [](FuncOp func) {
      return std::any_of(
          func.getArguments().begin(), func.getArguments().end(),
          [](BlockArgument arg) { return isa<MemRefType>(arg.getType()); });
    };

    /// We only create a new top-level function and call the original top-level
    /// function from the new one if the original top-level has `memref` in its
    /// argument
    auto funcOps = moduleOp.getOps<FuncOp>();
    bool hasMemrefArgsInTopLevel =
        std::any_of(funcOps.begin(), funcOps.end(), [&](auto funcOp) {
          return funcOp.getName() == topLevelFunction &&
                 hasMemrefArguments(funcOp);
        });

    if (hasMemrefArgsInTopLevel) {
      auto newTopLevelFunc = createNewTopLevelFn(moduleOp, topLevelFunction);
      if (!newTopLevelFunc)
        return failure();

      OpBuilder builder(moduleOp.getContext());
      Operation *oldTopLevelFuncOp =
          SymbolTable::lookupSymbolIn(moduleOp, topLevelFunction);
      if (auto oldTopLevelFunc = dyn_cast<FuncOp>(oldTopLevelFuncOp))
        insertCallFromNewTopLevel(builder, newTopLevelFunc, oldTopLevelFunc);
      else {
        moduleOp.emitOpError("Original top-level function not found!");
        return failure();
      }
      topLevelFunction = "main";
    }

    return success();
  }
};

void SCFToCalyxPass::runOnOperation() {
  // Clear internal state. See https://github.com/llvm/circt/issues/3235
  loweringState.reset();
  partialPatternRes = LogicalResult::failure();

  std::string topLevelFunction;
  if (failed(setTopLevelFunction(getOperation(), topLevelFunction))) {
    signalPassFailure();
    return;
  }

  /// Start conversion
  if (failed(labelEntryPoint(topLevelFunction))) {
    signalPassFailure();
    return;
  }
  loweringState = std::make_shared<calyx::CalyxLoweringState>(getOperation(),
                                                              topLevelFunction);

  /// --------------------------------------------------------------------------
  /// If you are a developer, it may be helpful to add a
  /// 'getOperation()->dump()' call after the execution of each stage to
  /// view the transformations that's going on.
  /// --------------------------------------------------------------------------

  /// A mapping is maintained between a function operation and its corresponding
  /// Calyx component.
  DenseMap<FuncOp, calyx::ComponentOp> funcMap;
  SmallVector<LoweringPattern, 8> loweringPatterns;
  calyx::PatternApplicationState patternState;

  /// Creates a new Calyx component for each FuncOp in the inpurt module.
  addOncePattern<FuncOpConversion>(loweringPatterns, patternState, funcMap,
                                   *loweringState);

  /// This pass inlines scf.ExecuteRegionOp's by adding control-flow.
  addGreedyPattern<InlineExecuteRegionOpPattern>(loweringPatterns);

  addOncePattern<BuildParGroups>(loweringPatterns, patternState, funcMap,
                                 *loweringState);

  /// This pattern converts all index typed values to an i32 integer.
  addOncePattern<calyx::ConvertIndexTypes>(loweringPatterns, patternState,
                                           funcMap, *loweringState);

  /// This pattern creates registers for all basic-block arguments.
  addOncePattern<calyx::BuildBasicBlockRegs>(loweringPatterns, patternState,
                                             funcMap, *loweringState);

  addOncePattern<calyx::BuildCallInstance>(loweringPatterns, patternState,
                                           funcMap, *loweringState);

  /// This pattern creates registers for the function return values.
  addOncePattern<calyx::BuildReturnRegs>(loweringPatterns, patternState,
                                         funcMap, *loweringState);

  /// This pattern creates registers for iteration arguments of scf.while
  /// operations. Additionally, creates a group for assigning the initial
  /// value of the iteration argument registers.
  addOncePattern<BuildWhileGroups>(loweringPatterns, patternState, funcMap,
                                   *loweringState);

  /// This pattern creates registers for iteration arguments of scf.for
  /// operations. Additionally, creates a group for assigning the initial
  /// value of the iteration argument registers.
  addOncePattern<BuildForGroups>(loweringPatterns, patternState, funcMap,
                                 *loweringState);

  addOncePattern<BuildIfGroups>(loweringPatterns, patternState, funcMap,
                                *loweringState);

  /// This pattern converts operations within basic blocks to Calyx library
  /// operators. Combinational operations are assigned inside a
  /// calyx::CombGroupOp, and sequential inside calyx::GroupOps.
  /// Sequential groups are registered with the Block* of which the operation
  /// originated from. This is used during control schedule generation. By
  /// having a distinct group for each operation, groups are analogous to SSA
  /// values in the source program.
  addOncePattern<BuildOpGroups>(loweringPatterns, patternState, funcMap,
                                *loweringState, writeJsonOpt);

  /// This pattern traverses the CFG of the program and generates a control
  /// schedule based on the calyx::GroupOp's which were registered for each
  /// basic block in the source function.
  addOncePattern<BuildControl>(loweringPatterns, patternState, funcMap,
                               *loweringState);

  /// This pass recursively inlines use-def chains of combinational logic (from
  /// non-stateful groups) into groups referenced in the control schedule.
  addOncePattern<calyx::InlineCombGroups>(loweringPatterns, patternState,
                                          *loweringState);

  /// This pattern performs various SSA replacements that must be done
  /// after control generation.
  addOncePattern<LateSSAReplacement>(loweringPatterns, patternState, funcMap,
                                     *loweringState);

  /// Eliminate any unused combinational groups. This is done before
  /// calyx::RewriteMemoryAccesses to avoid inferring slice components for
  /// groups that will be removed.
  addGreedyPattern<calyx::EliminateUnusedCombGroups>(loweringPatterns);

  /// This pattern rewrites accesses to memories which are too wide due to
  /// index types being converted to a fixed-width integer type.
  addOncePattern<calyx::RewriteMemoryAccesses>(loweringPatterns, patternState,
                                               *loweringState);

  /// This pattern removes the source FuncOp which has now been converted into
  /// a Calyx component.
  addOncePattern<CleanupFuncOps>(loweringPatterns, patternState, funcMap,
                                 *loweringState);

  /// Sequentially apply each lowering pattern.
  for (auto &pat : loweringPatterns) {
    LogicalResult partialPatternRes = runPartialPattern(
        pat.pattern,
        /*runOnce=*/pat.strategy == LoweringPattern::Strategy::Once);
    if (succeeded(partialPatternRes))
      continue;
    signalPassFailure();
    return;
  }

  //===--------------------------------------------------------------------===//
  // Cleanup patterns
  //===--------------------------------------------------------------------===//
  RewritePatternSet cleanupPatterns(&getContext());
  cleanupPatterns.add<calyx::MultipleGroupDonePattern,
                      calyx::NonTerminatingGroupDonePattern>(&getContext());
  if (failed(
          applyPatternsGreedily(getOperation(), std::move(cleanupPatterns)))) {
    signalPassFailure();
    return;
  }

  if (ciderSourceLocationMetadata) {
    // Debugging information for the Cider debugger.
    // Reference: https://docs.calyxir.org/debug/cider.html
    SmallVector<Attribute, 16> sourceLocations;
    getOperation()->walk([&](calyx::ComponentOp component) {
      return getCiderSourceLocationMetadata(component, sourceLocations);
    });

    MLIRContext *context = getOperation()->getContext();
    getOperation()->setAttr("calyx.metadata",
                            ArrayAttr::get(context, sourceLocations));
  }
}
} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSCFToCalyxPass() {
  return std::make_unique<SCFToCalyxPass>();
}

} // namespace circt
