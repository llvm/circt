#include "circt/Conversion/LLHDToLLVM/LLHDToLLVM.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace llhd {
#define GEN_PASS_CLASSES
#include "circt/Conversion/LLHDToLLVM/Passes.h.inc"
} // namespace llhd
} // namespace mlir

using namespace mlir;
using namespace mlir::llhd;

namespace {
// keep a counter to infer a signal's index in his entity's signal table
static int signalCounter = 0;
// keep a counter to infer the resume index after a wait instruction in a
// process
static int resumeCounter = 0;
//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Get an existing global string
Value getGlobalString(Location loc, OpBuilder &builder,
                      LLVMTypeConverter &typeConverter, LLVM::GlobalOp &str) {
  auto i8PtrTy = LLVM::LLVMType::getInt8PtrTy(typeConverter.getDialect());
  auto i32Ty = LLVM::LLVMType::getInt32Ty(typeConverter.getDialect());

  auto addr = builder.create<LLVM::AddressOfOp>(
      loc, str.getType().getPointerTo(), str.getName());
  auto idx = builder.create<LLVM::ConstantOp>(loc, i32Ty,
                                              builder.getI32IntegerAttr(0));
  llvm::SmallVector<Value, 2> idxs({idx, idx});
  auto gep = builder.create<LLVM::GEPOp>(loc, i8PtrTy, addr, idxs);
  return gep;
}

/// Looks up a symbol and inserts a new functino at the beginning of the
/// module's region in case the function does not exists. If
/// insertBodyAndTerminator is set, also adds the entry block and return
/// terminator
LLVM::LLVMFuncOp getOrInsertFunction(ModuleOp &module,
                                     ConversionPatternRewriter &rewriter,
                                     std::string name, LLVM::LLVMType signature,
                                     bool insertBodyAndTerminator = false) {
  auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                  name, signature);
    if (insertBodyAndTerminator) {
      func.addEntryBlock();
      OpBuilder b(func.getBody());
      b.create<LLVM::ReturnOp>(rewriter.getUnknownLoc(), ValueRange());
    }
  }
  return func;
}

/// Insert probe runtime call and extraction of details from the struct. The
/// mlir::Values of the details are returned, in struct-order.
std::pair<Value, Value> insertProbeSignal(ModuleOp &module,
                                          ConversionPatternRewriter &rewriter,
                                          LLVM::LLVMDialect *dialect,
                                          Operation *op, Value statePtr,
                                          Value signal) {
  auto i8PtrTy = LLVM::LLVMType::getInt8PtrTy(dialect);
  auto i32Ty = LLVM::LLVMType::getInt32Ty(dialect);
  auto i64Ty = LLVM::LLVMType::getInt64Ty(dialect);
  auto sigTy = LLVM::LLVMType::getStructTy(dialect, {i8PtrTy, i64Ty});

  auto prbSignature = LLVM::LLVMType::getFunctionTy(sigTy.getPointerTo(),
                                                    {i8PtrTy, i32Ty}, false);
  auto prbFunc =
      getOrInsertFunction(module, rewriter, "probe_signal", prbSignature);
  SmallVector<Value, 2> prbArgs({statePtr, signal});
  auto prbCall =
      rewriter
          .create<LLVM::CallOp>(op->getLoc(), sigTy.getPointerTo(),
                                rewriter.getSymbolRefAttr(prbFunc), prbArgs)
          .getResult(0);
  auto zeroC = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto oneC = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i32Ty,
                                                rewriter.getI32IntegerAttr(1));

  auto sigPtrPtr =
      rewriter.create<LLVM::GEPOp>(op->getLoc(), i8PtrTy.getPointerTo(),
                                   prbCall, ArrayRef<Value>({zeroC, zeroC}));

  auto offsetPtr =
      rewriter.create<LLVM::GEPOp>(op->getLoc(), i64Ty.getPointerTo(), prbCall,
                                   ArrayRef<Value>({zeroC, oneC}));
  auto sigPtr = rewriter.create<LLVM::LoadOp>(op->getLoc(), i8PtrTy, sigPtrPtr);
  auto offset = rewriter.create<LLVM::LoadOp>(op->getLoc(), i64Ty, offsetPtr);

  return std::pair<Value, Value>(sigPtr, offset);
}

/// Gather the types of values that are used outside of the block they're
/// defined in. An LLVMType structure containing those types, in order of
/// appearance, is returned.
LLVM::LLVMType getProcPersistenceTy(LLVM::LLVMDialect *dialect,
                                    LLVMTypeConverter &converter,
                                    ProcOp &proc) {
  SmallVector<LLVM::LLVMType, 3> types = SmallVector<LLVM::LLVMType, 3>();

  auto i32Ty = LLVM::LLVMType::getInt32Ty(dialect);

  proc.walk([&](Operation *op) -> WalkResult {
    if (op->isUsedOutsideOfBlock(op->getBlock())) {
      if (op->getResult(0).getType().isa<IntegerType>())
        types.push_back(converter.convertType(op->getResult(0).getType())
                            .cast<LLVM::LLVMType>());
      else if (op->getResult(0).getType().isa<SigType>())
        types.push_back(i32Ty);
    }
    return WalkResult::advance();
  });
  return LLVM::LLVMType::getStructTy(dialect, types);
}

/// Insert a comparison block that either jumps to the trueDest block, if the
/// resume index mathces the current index, or to falseDest otherwise. If no
/// falseDest is provided, the next block is taken insead.
void insertComparisonBlock(ConversionPatternRewriter &rewriter,
                           LLVM::LLVMDialect *dialect, Location loc,
                           Region *body, Value resumeIdx, int currIdx,
                           Block *trueDest, Block *falseDest = nullptr) {
  auto i32Ty = LLVM::LLVMType::getInt32Ty(dialect);
  // redirect entry block to a first comparison block. If it is a fresh start,
  // start from where original entry would have jumped, else the process is in
  // an illegal state and jump to the abort block
  auto secondBlock = ++body->begin();
  auto newBlock = rewriter.createBlock(body, secondBlock);
  auto cmpIdx = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(currIdx));
  auto cmpRes = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                              resumeIdx, cmpIdx);

  // default to jump to the next block for false
  if (!falseDest)
    falseDest = &*secondBlock;

  rewriter.create<LLVM::CondBrOp>(loc, cmpRes, trueDest, falseDest);

  // redirect entry block terminator to the new comparison block
  auto entryTer = body->front().getTerminator();
  entryTer->setSuccessor(newBlock, 0);
}

/// Insert the blocks and operations needed to persist values across suspension,
/// as well as ones needed to resume execution at the right spot.
void insertPersistence(LLVMTypeConverter &converter,
                       ConversionPatternRewriter &rewriter,
                       LLVM::LLVMDialect *dialect, Location loc, ProcOp &proc,
                       LLVM::LLVMType &stateTy, LLVM::LLVMFuncOp &converted) {
  auto i32Ty = LLVM::LLVMType::getInt32Ty(dialect);
  DominanceInfo dom(converted);

  // load resume index
  rewriter.setInsertionPoint(converted.getBody().front().getTerminator());
  auto zeroC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto oneC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                rewriter.getI32IntegerAttr(1));
  auto gep = rewriter.create<LLVM::GEPOp>(loc, i32Ty.getPointerTo(),
                                          converted.getArgument(1),
                                          ArrayRef<Value>({zeroC, oneC}));

  auto larg = rewriter.create<LLVM::LoadOp>(loc, i32Ty, gep);

  // insert an abort block as the last block
  auto abortBlock =
      rewriter.createBlock(&converted.getBody(), converted.getBody().end());
  rewriter.create<LLVM::ReturnOp>(loc, ValueRange());

  auto body = &converted.getBody();
  assert(body->front().getTerminator()->getNumSuccessors() > 0 &&
         "the process entry block is expected to branch to another block");
  // redirect entry block to a first comparison block. If on a fresh start,
  // start from where original entry would have jumped, else the process is in
  // an illegal state and jump to the abort block
  insertComparisonBlock(rewriter, dialect, loc, body, larg, 0,
                        body->front().getTerminator()->getSuccessor(0),
                        abortBlock);

  // keep track of the index in the presistence table of the operation we
  // are currently processing
  int i = 0;
  // keep track of the current resume index for comparison blocks
  int waitInd = 0;

  // insert persistence keeping for each operation escaping its parent block
  converted.walk([&](Operation *op) -> WalkResult {
    if (op->isUsedOutsideOfBlock(op->getBlock()) &&
        op->getResult(0) != larg.getResult() &&
        !(op->getResult(0).getType().isa<TimeType>())) {
      auto elemTy = stateTy.getStructElementType(3).getStructElementType(i);

      // store the value escaping it's definingn block in the persistence table
      rewriter.setInsertionPointAfter(op);
      auto zeroC0 = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(0));
      auto threeC0 = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(3));
      auto indC0 = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(i));
      auto gep0 = rewriter.create<LLVM::GEPOp>(
          loc, elemTy.getPointerTo(), converted.getArgument(1),
          ArrayRef<Value>({zeroC0, threeC0, indC0}));
      rewriter.create<LLVM::StoreOp>(loc, op->getResult(0), gep0);

      // load the value from the persistence table and substitute the original
      // use with it
      for (auto &use : op->getUses()) {
        if (dom.properlyDominates(op->getBlock(), use.getOwner()->getBlock())) {
          auto user = use.getOwner();
          rewriter.setInsertionPointToStart(user->getBlock());
          auto zeroC1 = rewriter.create<LLVM::ConstantOp>(
              loc, i32Ty, rewriter.getI32IntegerAttr(0));
          auto threeC1 = rewriter.create<LLVM::ConstantOp>(
              loc, i32Ty, rewriter.getI32IntegerAttr(3));
          auto indC1 = rewriter.create<LLVM::ConstantOp>(
              loc, i32Ty, rewriter.getI32IntegerAttr(i));
          auto gep1 = rewriter.create<LLVM::GEPOp>(
              loc, elemTy.getPointerTo(), converted.getArgument(1),
              ArrayRef<Value>({zeroC1, threeC1, indC1}));
          auto load1 = rewriter.create<LLVM::LoadOp>(loc, elemTy, gep1);

          use.set(load1);
        }
      }
      i++;
    }

    // insert a comparison block for wait operations
    if (auto wait = dyn_cast<WaitOp>(op)) {
      insertComparisonBlock(rewriter, dialect, loc, body, larg, ++waitInd,
                            wait.dest());
    }
    return WalkResult::advance();
  });
}

//===----------------------------------------------------------------------===//
// Unit conversions
//===----------------------------------------------------------------------===//

/// Convert an `llhd.entity` unit to LLVM. The result is an `llvm.func` which
/// takes a pointer to the state as arguments.
struct EntityOpConversion : public ConvertToLLVMPattern {
  explicit EntityOpConversion(MLIRContext *ctx,
                              LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::EntityOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // reset signal counter
    signalCounter = 0;
    // get adapted operands
    EntityOpAdaptor transformed(operands);
    // get entity operation
    auto entityOp = cast<EntityOp>(op);

    // collect llvm types
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&getDialect());

    // have an intermediate signature conversion to add the arguments for the
    // state, signal table and argument table poitner arguments
    LLVMTypeConverter::SignatureConversion intermediate(
        entityOp.getNumArguments());
    // add state, signal table and arguments table arguments
    intermediate.addInputs(
        ArrayRef<Type>({i8PtrTy, i32Ty.getPointerTo(), i32Ty.getPointerTo()}));
    for (unsigned i = 0; i < entityOp.getNumArguments(); i++)
      intermediate.addInputs(i, voidTy);
    rewriter.applySignatureConversion(&entityOp.getBody(), intermediate);

    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&entityOp.getBlocks().front());
    LLVMTypeConverter::SignatureConversion final(
        intermediate.getConvertedTypes().size());
    final.addInputs(0, i8PtrTy);
    final.addInputs(1, i32Ty.getPointerTo());
    final.addInputs(2, i32Ty.getPointerTo());

    for (unsigned i = 0; i < entityOp.getNumArguments(); i++) {
      // create gep and load operations from arguments table for each original
      // argument
      auto index = bodyBuilder.create<LLVM::ConstantOp>(
          rewriter.getUnknownLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto bitcast = bodyBuilder.create<LLVM::GEPOp>(
          rewriter.getUnknownLoc(), i32Ty.getPointerTo(),
          entityOp.getArgument(2), ArrayRef<Value>(index));
      auto load =
          bodyBuilder.create<LLVM::LoadOp>(rewriter.getUnknownLoc(), bitcast);
      // remap i-th original argument to the loaded value
      final.remapInput(i + 3, load.getResult());
    }

    rewriter.applySignatureConversion(&entityOp.getBody(), final);

    // converted entity signature
    auto funcTy = LLVM::LLVMType::getFunctionTy(
        voidTy, {i8PtrTy, i32Ty.getPointerTo(), i32Ty.getPointerTo()}, false);
    // // create the llvm function
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(
        op->getLoc(), entityOp.getName(), funcTy);

    // inline the entity region in the new llvm function
    rewriter.inlineRegionBefore(entityOp.getBody(), llvmFunc.getBody(),
                                llvmFunc.end());

    // erase original operation
    rewriter.eraseOp(op);

    return success();
  }
};

/// Convert an `"llhd.terminator" operation to `llvm.return`.
struct TerminatorOpConversion : public ConvertToLLVMPattern {
  explicit TerminatorOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::TerminatorOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // just replace the original op with return void
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange());

    return success();
  }
};

/// Convert an `llhd.proc` operation to llvm dialect. This inserts the required
/// logic to resume execution after an `llhd.wait` operation, as well as state
/// keeping for values that need to persist across suspension.
struct ProcOpConversion : public ConvertToLLVMPattern {
  explicit ProcOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(ProcOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto procOp = cast<ProcOp>(op);

    // get adapted operands
    ProcOpAdaptor transformed(operands);

    // collect llvm types
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = LLVM::LLVMType::getInt1Ty(&getDialect());
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&getDialect());
    auto senseTableTy =
        LLVM::LLVMType::getArrayTy(i1Ty, procOp.insAttr().getInt())
            .getPointerTo();
    auto stateTy = LLVM::LLVMType::getStructTy(
        /* current instance  */ i8PtrTy, /* resume index */ i32Ty,
        /* sense flags */ senseTableTy, /* persistent types */
        getProcPersistenceTy(&getDialect(), typeConverter, procOp));

    // reset the resume index counter
    resumeCounter = 0;

    // have an intermediate signature conversion to add the arguments for the
    // state, process-specific state and signal arguments table
    LLVMTypeConverter::SignatureConversion intermediate(
        procOp.getNumArguments());
    // add state, process state table and arguments table arguments
    ArrayRef<Type> procSigTys(
        {i8PtrTy, stateTy.getPointerTo(), i32Ty.getPointerTo()});
    intermediate.addInputs(procSigTys);
    for (unsigned int i = 0; i < procOp.getNumArguments(); i++)
      intermediate.addInputs(i, voidTy);
    rewriter.applySignatureConversion(&procOp.getBody(), intermediate);
    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&procOp.getBlocks().front());
    LLVMTypeConverter::SignatureConversion final(
        intermediate.getConvertedTypes().size());
    final.addInputs(0, i8PtrTy);
    final.addInputs(1, stateTy.getPointerTo());
    final.addInputs(2, i32Ty.getPointerTo());

    for (unsigned int i = 0; i < procOp.getNumArguments(); i++) {
      // create gep and load operations from arguments table for each original
      // argument
      auto index = bodyBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto bitcast = bodyBuilder.create<LLVM::GEPOp>(
          op->getLoc(), i32Ty.getPointerTo(), procOp.getArgument(2),
          ArrayRef<Value>({index}));
      auto load = bodyBuilder.create<LLVM::LoadOp>(op->getLoc(), bitcast);

      // remap i-th original argument to the loaded value
      final.remapInput(i + 3, load.getResult());
    }

    rewriter.applySignatureConversion(&procOp.getBody(), final);

    // converted entity signature
    auto funcTy = LLVM::LLVMType::getFunctionTy(
        voidTy, {i8PtrTy, stateTy.getPointerTo(), i32Ty.getPointerTo()}, false);
    // create the llvm function
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(),
                                                      procOp.getName(), funcTy);

    // inline the entity region in the new llvm function
    rewriter.inlineRegionBefore(procOp.getBody(), llvmFunc.getBody(),
                                llvmFunc.end());

    insertPersistence(typeConverter, rewriter, &getDialect(), op->getLoc(),
                      procOp, stateTy, llvmFunc);

    rewriter.eraseOp(op);

    return success();
  }
};

/// Convert an `llhd.halt` operation to llvm dialect. This zeroes out all the
/// senses and returns, effectively making the process unable to be invoked
/// again.
struct HaltOpConversion : public ConvertToLLVMPattern {
  explicit HaltOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(HaltOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1Ty = LLVM::LLVMType::getInt1Ty(&getDialect());
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&getDialect());

    auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto procState = llvmFunc.getArgument(1);
    auto senseTableTy = procState.getType()
                            .cast<LLVM::LLVMType>()
                            .getPointerElementTy()
                            .getStructElementType(2)
                            .getPointerElementTy();

    // get senses ptr from process state argument
    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
    auto twoC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(2));
    auto sensePtrGep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), senseTableTy.getPointerTo().getPointerTo(), procState,
        ArrayRef<Value>({zeroC, twoC}));
    auto sensePtr = rewriter.create<LLVM::LoadOp>(
        op->getLoc(), senseTableTy.getPointerTo(), sensePtrGep);

    // zero out all the senses flags
    for (unsigned int i = 0; i < senseTableTy.getArrayNumElements(); i++) {
      auto indC = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto zeroB = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getI32IntegerAttr(0));
      auto senseElemPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), i1Ty.getPointerTo(), sensePtr,
          ArrayRef<Value>({zeroC, indC}));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), zeroB, senseElemPtr);
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange());
    return success();
  }
};

/// Covnert an `llhd.wait` operation to llvm dialect. This sets the current
/// resume point, sets the observed senses (if present) and schedules the timed
/// wake up (if present).
struct WaitOpConversion : public ConvertToLLVMPattern {
  explicit WaitOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(WaitOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto waitOp = cast<WaitOp>(op);
    WaitOpAdaptor transformed(operands, nullptr);
    auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>();

    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = LLVM::LLVMType::getInt1Ty(&getDialect());
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&getDialect());

    // get llhd_suspend runtime function
    auto llhdSuspendTy = LLVM::LLVMType::getFunctionTy(
        voidTy, {i8PtrTy, i8PtrTy, i32Ty, i32Ty, i32Ty}, false);
    auto module = op->getParentOfType<ModuleOp>();
    auto llhdSuspendFunc =
        getOrInsertFunction(module, rewriter, "llhd_suspend", llhdSuspendTy);

    auto statePtr = llvmFunc.getArgument(0);
    auto procState = llvmFunc.getArgument(1);
    auto procStateTy = procState.getType().dyn_cast<LLVM::LLVMType>();
    auto senseTableTy = procStateTy.getPointerElementTy()
                            .getStructElementType(2)
                            .getPointerElementTy();

    // get senses ptr
    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
    auto oneC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
    auto twoC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(2));
    auto sensePtrGep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), senseTableTy.getPointerTo().getPointerTo(), procState,
        ArrayRef<Value>({zeroC, twoC}));
    auto sensePtr = rewriter.create<LLVM::LoadOp>(
        op->getLoc(), senseTableTy.getPointerTo(), sensePtrGep);

    // set senses flags
    // TODO: actually handle observed signals
    for (unsigned int i = 0; i < senseTableTy.getArrayNumElements(); i++) {
      auto indC = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto zeroB = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getI32IntegerAttr(0));
      auto senseElemPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), i1Ty.getPointerTo(), sensePtr,
          ArrayRef<Value>({zeroC, indC}));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), zeroB, senseElemPtr);
    }

    // get time constats, if present
    Value realTimeConst;
    Value deltaConst;
    Value epsConst;
    if (waitOp.timeMutable().size() > 0) {
      auto timeAttr = cast<llhd::ConstOp>(waitOp.time().getDefiningOp())
                          .valueAttr()
                          .dyn_cast<TimeAttr>();
      // get real time as an attribute
      auto realTimeAttr = rewriter.getI32IntegerAttr(timeAttr.getTime());
      // create new time const operation
      realTimeConst =
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), i32Ty, realTimeAttr);
      // get the delta step as an attribute
      auto deltaAttr = rewriter.getI32IntegerAttr(timeAttr.getDelta());
      // create new delta const operation
      deltaConst =
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), i32Ty, deltaAttr);
      // get the epsilon step as an attribute
      auto epsAttr = rewriter.getI32IntegerAttr(timeAttr.getEps());
      // create new eps const operation
      epsConst =
          rewriter.create<LLVM::ConstantOp>(op->getLoc(), i32Ty, epsAttr);
    }

    auto procStateBC =
        rewriter.create<LLVM::BitcastOp>(op->getLoc(), i8PtrTy, procState);
    auto resumeIdxC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(++resumeCounter));
    auto resumeIdxPtr =
        rewriter.create<LLVM::GEPOp>(op->getLoc(), i32Ty.getPointerTo(),
                                     procState, ArrayRef<Value>({zeroC, oneC}));
    rewriter.create<LLVM::StoreOp>(op->getLoc(), resumeIdxC, resumeIdxPtr);

    SmallVector<Value, 5> args(
        {statePtr, procStateBC, realTimeConst, deltaConst, epsConst});
    rewriter.create<LLVM::CallOp>(
        op->getLoc(), voidTy, rewriter.getSymbolRefAttr(llhdSuspendFunc), args);

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange());
    return success();
  }
};

/// Lower an llhd.inst operation to llvm. This generates malloc calls and
/// alloc_signal calls (to store the pointer into the state) for each signal in
/// the instantiated entity.
struct InstOpConversion : public ConvertToLLVMPattern {
  explicit InstOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(InstOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get inst operation
    auto instOp = cast<InstOp>(op);
    // get parent module
    auto module = op->getParentOfType<ModuleOp>();

    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = LLVM::LLVMType::getInt1Ty(&getDialect());
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&getDialect());
    auto i64Ty = LLVM::LLVMType::getInt64Ty(&getDialect());

    // init function signature: (i8* %state) -> void
    auto initFuncTy = LLVM::LLVMType::getFunctionTy(voidTy, {i8PtrTy}, false);
    auto initFunc =
        getOrInsertFunction(module, rewriter, initCall, initFuncTy, true);

    //! get or insert malloc function definition
    // malloc function signature: (i64 %size) -> i8* %pointer
    auto mallocSigFuncTy =
        LLVM::LLVMType::getFunctionTy(i8PtrTy, {i64Ty}, false);
    auto mallFunc =
        getOrInsertFunction(module, rewriter, "malloc", mallocSigFuncTy);

    //! get or insert library call definition
    // alloc_signal function signature: (i8* %state, i8* %sig_name, i8*
    // %sig_owner, i32 %value) -> i32 %sig_index
    auto allocSigFuncTy = LLVM::LLVMType::getFunctionTy(
        i32Ty, {i8PtrTy, i32Ty, i8PtrTy, i8PtrTy, i64Ty}, false);
    auto sigFunc =
        getOrInsertFunction(module, rewriter, allocCall, allocSigFuncTy);

    // get or insert alloc_proc library call definition
    auto allocProcFuncTy = LLVM::LLVMType::getFunctionTy(
        voidTy, {i8PtrTy, i8PtrTy, i8PtrTy}, false);
    auto allocProcFunc =
        getOrInsertFunction(module, rewriter, "alloc_proc", allocProcFuncTy);

    Value initStatePtr = initFunc.getArgument(0);

    // get builder with insertion point before the init function terminator
    OpBuilder initBuilder =
        OpBuilder::atBlockTerminator(&initFunc.getBody().getBlocks().front());

    // use the instance name to retrieve the instance from the state
    auto ownerName = instOp.name();
    //! get or create owner name string
    Value owner;

    auto parentSym =
        module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName.str());
    if (!parentSym) {
      owner = LLVM::createGlobalString(
          rewriter.getUnknownLoc(), initBuilder, "instance." + ownerName.str(),
          ownerName.str() + '\0', LLVM::Linkage::Internal,
          typeConverter.getDialect());
      parentSym =
          module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName.str());
    } else {
      owner = getGlobalString(rewriter.getUnknownLoc(), initBuilder,
                              typeConverter, parentSym);
    }

    if (auto child = module.lookupSymbol<EntityOp>(instOp.callee())) {
      // walk over the unit and generate mallocs for each one of its signals
      // index of the signal in the unit's signal table
      int initCounter = 0;
      child.walk([&](Operation *op) -> WalkResult {
        if (auto sigOp = dyn_cast<SigOp>(op)) {
          // get index constant of the signal in the unit's signal table
          auto indexConst = initBuilder.create<LLVM::ConstantOp>(
              rewriter.getUnknownLoc(), i32Ty,
              rewriter.getI32IntegerAttr(initCounter));
          initCounter++;

          //! add signal allocation to the init function
          // clone and insert init's defining operation (assmued to be a
          // constant op)
          auto initDef =
              initBuilder.insert(sigOp.init().getDefiningOp()->clone())
                  ->getResult(0);
          // malloc required space
          int size =
              std::ceil(sigOp.init().getType().getIntOrFloatBitWidth() / 8.0);
          auto sizeConst = initBuilder.create<LLVM::ConstantOp>(
              rewriter.getUnknownLoc(), i64Ty,
              rewriter.getI64IntegerAttr(size));
          // malloc an extra byte to avoid segfaulting when loading an offset
          // signal
          auto mallocSize = initBuilder.create<LLVM::ConstantOp>(
              rewriter.getUnknownLoc(), i64Ty,
              rewriter.getI64IntegerAttr(size * 2));
          llvm::SmallVector<Value, 1> margs({mallocSize});
          auto mall = initBuilder
                          .create<LLVM::CallOp>(
                              rewriter.getUnknownLoc(), i8PtrTy,
                              rewriter.getSymbolRefAttr(mallFunc), margs)
                          .getResult(0);
          auto bitcast = initBuilder.create<LLVM::BitcastOp>(
              rewriter.getUnknownLoc(),
              typeConverter.convertType(sigOp.init().getType())
                  .cast<LLVM::LLVMType>()
                  .getPointerTo(),
              mall);
          initBuilder.create<LLVM::StoreOp>(rewriter.getUnknownLoc(), initDef,
                                            bitcast);

          llvm::SmallVector<Value, 5> args(
              {initStatePtr, indexConst, owner, mall, sizeConst});
          initBuilder.create<LLVM::CallOp>(rewriter.getUnknownLoc(), i32Ty,
                                           rewriter.getSymbolRefAttr(sigFunc),
                                           args);
        }
        return WalkResult::advance();
      });
    } else if (auto proc = module.lookupSymbol<ProcOp>(instOp.callee())) {
      auto sensesPtrTy =
          LLVM::LLVMType::getArrayTy(i1Ty, instOp.inputs().size())
              .getPointerTo();
      auto procStatePtrTy =
          LLVM::LLVMType::getStructTy(
              i8PtrTy, i32Ty, sensesPtrTy,
              getProcPersistenceTy(&getDialect(), typeConverter, proc))
              .getPointerTo();

      auto zeroC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
      auto oneC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
      auto twoC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(2));

      // malloc space for proc state
      auto procStateNullPtr =
          initBuilder.create<LLVM::NullOp>(op->getLoc(), procStatePtrTy);
      auto procStateGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), procStatePtrTy, procStateNullPtr,
          ArrayRef<Value>({oneC}));
      auto procStateSize = initBuilder.create<LLVM::PtrToIntOp>(
          op->getLoc(), i64Ty, procStateGep);
      llvm::SmallVector<Value, 1> procStateMArgs({procStateSize});
      auto procStateMall =
          initBuilder
              .create<LLVM::CallOp>(op->getLoc(), i8PtrTy,
                                    rewriter.getSymbolRefAttr(mallFunc),
                                    procStateMArgs)
              .getResult(0);

      auto procStateBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), procStatePtrTy, procStateMall);

      // malloc space for owner name
      auto strSizeC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i64Ty, rewriter.getI64IntegerAttr(ownerName.size()));

      llvm::SmallVector<Value, 1> strMallArgs({strSizeC});
      auto strMall = initBuilder
                         .create<LLVM::CallOp>(
                             op->getLoc(), i8PtrTy,
                             rewriter.getSymbolRefAttr(mallFunc), strMallArgs)
                         .getResult(0);
      auto procStateOwnerPtr = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), i8PtrTy.getPointerTo(), procStateBC,
          ArrayRef<Value>({zeroC, zeroC}));
      initBuilder.create<LLVM::StoreOp>(op->getLoc(), strMall,
                                        procStateOwnerPtr);

      // store initial resume index
      auto resumeGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), i32Ty.getPointerTo(), procStateBC,
          ArrayRef<Value>({zeroC, oneC}));
      initBuilder.create<LLVM::StoreOp>(op->getLoc(), zeroC, resumeGep);

      // malloc space for senses
      auto sensesNullPtr =
          initBuilder.create<LLVM::NullOp>(op->getLoc(), sensesPtrTy);
      auto sensesGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), sensesPtrTy, sensesNullPtr, ArrayRef<Value>({oneC}));
      auto sensesSize =
          initBuilder.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, sensesGep);
      SmallVector<Value, 1> senseMArgs({sensesSize});
      auto sensesMall = initBuilder
                            .create<LLVM::CallOp>(
                                op->getLoc(), i8PtrTy,
                                rewriter.getSymbolRefAttr(mallFunc), senseMArgs)
                            .getResult(0);

      auto sensesBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), sensesPtrTy, sensesMall);

      // set initial senses to 1
      for (unsigned i = 0;
           i < sensesPtrTy.getPointerElementTy().getArrayNumElements(); i++) {
        auto oneB = initBuilder.create<LLVM::ConstantOp>(
            op->getLoc(), i1Ty, rewriter.getBoolAttr(true));
        auto gepInd = initBuilder.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
        auto senseGep = initBuilder.create<LLVM::GEPOp>(
            op->getLoc(), i1Ty.getPointerTo(), sensesBC,
            ArrayRef<Value>({zeroC, gepInd}));
        initBuilder.create<LLVM::StoreOp>(op->getLoc(), oneB, senseGep);
      }

      // store senses ptr in procstate
      auto procStateSensesPtr = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), sensesPtrTy.getPointerTo(), procStateBC,
          ArrayRef<Value>({zeroC, twoC}));
      initBuilder.create<LLVM::StoreOp>(op->getLoc(), sensesBC,
                                        procStateSensesPtr);

      SmallVector<Value, 4> allocProcArgs({initStatePtr, owner, procStateMall});
      initBuilder.create<LLVM::CallOp>(op->getLoc(), voidTy,
                                       rewriter.getSymbolRefAttr(allocProcFunc),
                                       allocProcArgs);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const std::string allocCall = "alloc_signal";
  const std::string initCall = "llhd_init";
};

//===----------------------------------------------------------------------===//
// Signal conversions
//===----------------------------------------------------------------------===//

/// Convert an `llhd.sig` operation to LLVM. The i-th signal of an entity get's
/// lowered to a load of the i-th element of the signal table, passed as an
/// argument.
struct SigOpConversion : public ConvertToLLVMPattern {
  explicit SigOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::SigOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get adapted opreands
    SigOpAdaptor transformed(operands);

    // collect llvm types
    auto i32Ty = LLVM::LLVMType::getInt32Ty(typeConverter.getDialect());

    //! load the signal's index from the signal table
    // get signal table argument
    Value sigTablePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(1);

    // get index in the signal table and increase counter
    auto indexConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(signalCounter));
    signalCounter++;

    // gep&load index
    auto gep =
        rewriter.create<LLVM::GEPOp>(op->getLoc(), i32Ty.getPointerTo(),
                                     sigTablePtr, ArrayRef<Value>(indexConst));
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, gep);

    return success();
  }
};

/// Convert an `llhd.prb` operation to LLVM. The result is a library call to the
/// `@probe_signal` function. The signal details are then extracted and used to
/// load the final probe value.
struct PrbOpConversion : public ConvertToLLVMPattern {
  explicit PrbOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::PrbOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get adapted operands
    PrbOpAdaptor transformed(operands);
    // get probe operation
    auto prbOp = cast<PrbOp>(op);
    // get parent module
    auto module = op->getParentOfType<ModuleOp>();

    // collect llvm types
    auto finalTy =
        typeConverter.convertType(prbOp.getType()).cast<LLVM::LLVMType>();

    // get amount of bytes to load. An extra byte is always loaded to cover the
    // case where a subsignal spans halfway in the last byte.
    int resWidth = prbOp.getType().getIntOrFloatBitWidth();
    int loadWidth = (std::ceil(resWidth / 8.0) + 1) * 8;
    auto loadTy = LLVM::LLVMType::getIntNTy(&getDialect(), loadWidth);

    // get pointer to state from function arguments
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);

    // get signal pointer and offset
    auto sigDetail = insertProbeSignal(module, rewriter, &getDialect(), op,
                                       statePtr, transformed.signal());
    auto sigPtr = sigDetail.first;
    auto offset = sigDetail.second;

    // bitcast to adjusted load width
    auto bitcast = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), loadTy.getPointerTo(), sigPtr);

    // create load of probed value
    auto loadSig = rewriter.create<LLVM::LoadOp>(op->getLoc(), loadTy, bitcast);

    // TODO: cover the case of loadTy being larger than 64 bits (zext)
    // adjust offset constant width to perform the shift
    auto trOff = rewriter.create<LLVM::TruncOp>(op->getLoc(), loadTy, offset);

    // shift loaded value by offset
    auto shifted =
        rewriter.create<LLVM::LShrOp>(op->getLoc(), loadTy, loadSig, trOff);

    // truncate to signal width
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, finalTy, shifted);

    return success();
  }
};

/// Convert an `llhd.drv` operation to LLVM. The result is a library call to the
/// `@drive_signal` function, which declaration is inserted at the beginning of
/// the module if missing. The required arguments are either generated or
/// fetched.
struct DrvOpConversion : public ConvertToLLVMPattern {
  explicit DrvOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::DrvOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get adapted operands;
    DrvOpAdaptor transformed(operands);
    // get drive operation
    auto drvOp = cast<DrvOp>(op);
    // get parent module
    auto module = op->getParentOfType<ModuleOp>();

    // collect used llvm types
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i32Ty = LLVM::LLVMType::getInt32Ty(typeConverter.getDialect());
    auto i64Ty = LLVM::LLVMType::getInt64Ty(typeConverter.getDialect());

    // get or insert drive library call
    auto drvFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(libCall);
    if (!drvFunc) {
      OpBuilder moduleBuilder(module.getBodyRegion());
      // drv function signature: (i8* %state, i32 %sig_index, i8* %new_value,
      // i64 width i32 %time, i32 %delta, i32 %eps) -> ()
      auto drvFuncTy = LLVM::LLVMType::getFunctionTy(
          voidTy, {i8PtrTy, i32Ty, i8PtrTy, i64Ty, i32Ty, i32Ty, i32Ty},
          /*isVarArg=*/false);
      drvFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                       libCall, drvFuncTy);
    }

    // get state pointer from function arguments
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);

    int sigWidth = drvOp.signal()
                       .getType()
                       .dyn_cast<SigType>()
                       .getUnderlyingType()
                       .getIntOrFloatBitWidth();

    auto widthConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i64Ty, rewriter.getI64IntegerAttr(sigWidth));

    auto oneConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
    auto alloca = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(),
        transformed.value().getType().cast<LLVM::LLVMType>().getPointerTo(),
        oneConst, 4);
    rewriter.create<LLVM::StoreOp>(op->getLoc(), transformed.value(), alloca);
    auto bc = rewriter.create<LLVM::BitcastOp>(op->getLoc(), i8PtrTy, alloca);

    // get constant time operation
    auto timeAttr = cast<llhd::ConstOp>(drvOp.time().getDefiningOp())
                        .valueAttr()
                        .dyn_cast<TimeAttr>();
    // get real time as an attribute
    auto realTimeAttr = rewriter.getI32IntegerAttr(timeAttr.getTime());
    // create new time const operation
    auto realTimeConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(typeConverter.getDialect()),
        realTimeAttr);
    // get the delta step as an attribute
    auto deltaAttr = rewriter.getI32IntegerAttr(timeAttr.getDelta());
    // create new delta const operation
    auto deltaConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(typeConverter.getDialect()),
        deltaAttr);
    // get the epsilon step as an attribute
    auto epsAttr = rewriter.getI32IntegerAttr(timeAttr.getEps());
    // create new eps const operation
    auto epsConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), LLVM::LLVMType::getInt32Ty(typeConverter.getDialect()),
        epsAttr);

    // define library call arguments
    llvm::SmallVector<Value, 7> args({statePtr, transformed.signal(), bc,
                                      widthConst, realTimeConst, deltaConst,
                                      epsConst});
    // create library call
    rewriter.create<LLVM::CallOp>(op->getLoc(), voidTy,
                                  rewriter.getSymbolRefAttr(drvFunc), args);
    // erase original operation
    rewriter.eraseOp(op);

    return success();
  }

private:
  const std::string libCall = "drive_signal";
};

//===----------------------------------------------------------------------===//
// Bitwise conversions
//===----------------------------------------------------------------------===//

/// Convert an `llhd.not` operation. The result is an `llvm.xor` operation,
/// xor-ing the operand with all ones.
struct NotOpConversion : public ConvertToLLVMPattern {
  explicit NotOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::NotOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // get adapted operands
    NotOpAdaptor transformed(operands);
    // get `llhd.not` operation
    auto notOp = cast<NotOp>(op);
    // get integer width
    unsigned width = notOp.getType().getIntOrFloatBitWidth();
    // get llvm types
    auto iTy = LLVM::LLVMType::getIntNTy(typeConverter.getDialect(), width);

    // get mask operand
    APInt mask(width, 0);
    mask.setAllBits();
    auto rhs = rewriter.getIntegerAttr(rewriter.getIntegerType(width), mask);
    auto rhsConst = rewriter.create<LLVM::ConstantOp>(op->getLoc(), iTy, rhs);

    // replace original op with llvm equivalent
    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(
        op, typeConverter.convertType(notOp.getType()), transformed.value(),
        rhsConst);

    return success();
  }
};

/// Convert an `llhd.shr` operation to llvm. All the operands are extended to
/// the width obtained by combining the hidden and base values. This combined
/// value is then shifted (exposing the hidden value) and truncated to the base
/// length
struct ShrOpConversion : public ConvertToLLVMPattern {
  explicit ShrOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ShrOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    ShrOpAdaptor transformed(operands);
    auto shrOp = cast<ShrOp>(op);

    if (auto resTy = shrOp.result().getType().dyn_cast<IntegerType>()) {
      // get widths
      auto baseWidth = shrOp.getType().getIntOrFloatBitWidth();
      auto hdnWidth = shrOp.hidden().getType().getIntOrFloatBitWidth();
      auto full = baseWidth + hdnWidth;

      auto tmpTy = LLVM::LLVMType::getIntNTy(typeConverter.getDialect(), full);

      // extend all operands to the base and hidden combined  width
      auto baseZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                    transformed.base());
      auto hdnZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                   transformed.hidden());
      auto amntZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                    transformed.amount());

      // shift hidden operand to prepend to full value
      auto hdnShAmnt = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), tmpTy,
          rewriter.getIntegerAttr(rewriter.getIntegerType(full), baseWidth));
      auto hdnSh =
          rewriter.create<LLVM::ShlOp>(op->getLoc(), tmpTy, hdnZext, hdnShAmnt);

      // combine base and hidden operands
      auto combined =
          rewriter.create<LLVM::OrOp>(op->getLoc(), tmpTy, hdnSh, baseZext);

      // perform the right shift
      auto shifted = rewriter.create<LLVM::LShrOp>(op->getLoc(), tmpTy,
                                                   combined, amntZext);

      // truncate to final width
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(
          op, transformed.base().getType(), shifted);

      return success();
    } else if (auto resTy = shrOp.result().getType().dyn_cast<SigType>()) {
      auto module = op->getParentOfType<ModuleOp>();

      auto i8PtrTy = getVoidPtrType();
      auto i32Ty = LLVM::LLVMType::getInt32Ty(&getDialect());
      auto i64Ty = LLVM::LLVMType::getInt64Ty(&getDialect());

      // get add_subsignal runtime call
      auto addSubSignature = LLVM::LLVMType::getFunctionTy(
          i32Ty, {i8PtrTy, i32Ty, i8PtrTy, i64Ty, i64Ty}, false);
      auto addSubFunc = getOrInsertFunction(module, rewriter, "add_subsignal",
                                            addSubSignature);

      // get state pointer from arguments
      auto statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);

      // get signal pointer and offset
      auto sigDetail = insertProbeSignal(module, rewriter, &getDialect(), op,
                                         statePtr, transformed.base());
      auto sigPtr = sigDetail.first;
      auto offset = sigDetail.second;

      auto zextAmnt = rewriter.create<LLVM::ZExtOp>(op->getLoc(), i64Ty,
                                                    transformed.amount());

      // adjust slice start point from signal's offset
      auto adjustedAmnt =
          rewriter.create<LLVM::AddOp>(op->getLoc(), offset, zextAmnt);

      // shift pointer to the new start byte
      auto ptrToInt =
          rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, sigPtr);
      auto const8 = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i64Ty, rewriter.getI64IntegerAttr(8));
      auto ptrOffset =
          rewriter.create<LLVM::UDivOp>(op->getLoc(), adjustedAmnt, const8);
      auto shiftedPtr =
          rewriter.create<LLVM::AddOp>(op->getLoc(), ptrToInt, ptrOffset);
      auto newPtr =
          rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), i8PtrTy, shiftedPtr);

      // compute offset into the first byte
      auto bitOffset =
          rewriter.create<LLVM::URemOp>(op->getLoc(), adjustedAmnt, const8);

      auto lenConst = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i64Ty,
          rewriter.getI64IntegerAttr(
              resTy.getUnderlyingType().getIntOrFloatBitWidth()));

      // add subsignal to the state
      SmallVector<Value, 5> addSubArgs(
          {statePtr, transformed.base(), newPtr, lenConst, bitOffset});
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op, i32Ty, rewriter.getSymbolRefAttr(addSubFunc), addSubArgs);

      return success();
    }

    return failure();
  }
};

/// Convert an `llhd.shr` operation to llvm. All the operands are extended to
/// the width obtained by combining the hidden and base values. This combined
/// value is then shifted right by `hidden_width - amount` (exposing the hidden
/// value) and truncated to the base length
struct ShlOpConversion : public ConvertToLLVMPattern {
  explicit ShlOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ShlOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    ShlOpAdaptor transformed(operands);
    auto shlOp = cast<ShlOp>(op);
    assert(!(shlOp.getType().getKind() == llhd::LLHDTypes::Sig) &&
           "sig not yet supported");

    // get widths
    auto baseWidth = shlOp.getType().getIntOrFloatBitWidth();
    auto hdnWidth = shlOp.hidden().getType().getIntOrFloatBitWidth();
    auto full = baseWidth + hdnWidth;

    auto tmpTy = LLVM::LLVMType::getIntNTy(typeConverter.getDialect(), full);

    // extend all operands to the base and hidden combined  width
    auto baseZext =
        rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy, transformed.base());
    auto hdnZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                 transformed.hidden());
    auto amntZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                  transformed.amount());

    // shift hidden operand to
    // prepend to full value
    auto hdnWidthConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), tmpTy,
        rewriter.getIntegerAttr(rewriter.getIntegerType(full), hdnWidth));
    auto baseSh = rewriter.create<LLVM::ShlOp>(op->getLoc(), tmpTy, baseZext,
                                               hdnWidthConst);

    // combine base and hidden operands
    auto combined =
        rewriter.create<LLVM::OrOp>(op->getLoc(), tmpTy, baseSh, hdnZext);

    // get the final right shift amount
    auto shrAmnt = rewriter.create<LLVM::SubOp>(op->getLoc(), tmpTy,
                                                hdnWidthConst, amntZext);

    // perform the right shift
    auto shifted =
        rewriter.create<LLVM::LShrOp>(op->getLoc(), tmpTy, combined, shrAmnt);

    // truncate to final width
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, transformed.base().getType(),
                                               shifted);

    return success();
  }
};

using AndOpConversion = OneToOneConvertToLLVMPattern<llhd::AndOp, LLVM::AndOp>;
using OrOpConversion = OneToOneConvertToLLVMPattern<llhd::OrOp, LLVM::OrOp>;
using XorOpConversion = OneToOneConvertToLLVMPattern<llhd::XorOp, LLVM::XOrOp>;

//===----------------------------------------------------------------------===//
// Value manipulation conversions
//===----------------------------------------------------------------------===//

/// Lower an LLHD constant operation to LLVM. Time constant are treated as a
/// special case, by just erasing them. Operations that use time constants
/// are assumed to extract and convert the elements they require. Remaining
/// const types are lowered to an equivalent `llvm.mlir.constant` operation.
struct ConstOpConversion : public ConvertToLLVMPattern {
  explicit ConstOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ConstOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operand,
                  ConversionPatternRewriter &rewriter) const override {
    // get ConstOp
    auto constOp = cast<ConstOp>(op);
    // get const's attribute
    auto attr = constOp.value();
    // treat time const special case
    if (!attr.getType().isa<IntegerType>()) {
      rewriter.eraseOp(op);
      return success();
    }
    // get llvm converted type
    auto intType = typeConverter.convertType(attr.getType());
    // replace op with llvm constant op
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, intType,
                                                  constOp.valueAttr());

    return success();
  }
};

/// Convert an llhd.exts operation. For integers, the value is shifted to the
/// start index and then truncated to the final length. Other types are not yet
/// supported and fail the conversion.
struct ExtsOpConversion : public ConvertToLLVMPattern {
  explicit ExtsOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ExtsOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto extsOp = cast<ExtsOp>(op);

    ExtsOpAdaptor transformed(operands);

    auto indexTy = typeConverter.convertType(extsOp.startAttr().getType());
    auto i8PtrTy = getVoidPtrType();
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&getDialect());
    auto i64Ty = LLVM::LLVMType::getInt64Ty(&getDialect());

    // get attributes as constants
    auto startConst = rewriter.create<LLVM::ConstantOp>(op->getLoc(), indexTy,
                                                        extsOp.startAttr());
    auto lenConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), indexTy, rewriter.getIndexAttr(extsOp.getSliceSize()));

    if (auto retTy = extsOp.result().getType().dyn_cast<IntegerType>()) {
      auto resTy = typeConverter.convertType(extsOp.result().getType());
      // adjust index const for shifting
      Value adjusted;
      if (extsOp.target().getType().getIntOrFloatBitWidth() < 64) {
        adjusted = rewriter.create<LLVM::TruncOp>(
            op->getLoc(), transformed.target().getType(), startConst);
      } else {
        adjusted = rewriter.create<LLVM::ZExtOp>(
            op->getLoc(), transformed.target().getType(), startConst);
      }

      // shift right by index
      auto shr = rewriter.create<LLVM::LShrOp>(op->getLoc(),
                                               transformed.target().getType(),
                                               transformed.target(), adjusted);
      // truncate to length
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, resTy, shr);

      return success();
    } else if (auto resTy = extsOp.result().getType().dyn_cast<SigType>()) {
      auto module = op->getParentOfType<ModuleOp>();

      // get add_subsignal runtime call
      auto addSubSignature = LLVM::LLVMType::getFunctionTy(
          i32Ty, {i8PtrTy, i32Ty, i8PtrTy, i64Ty, i64Ty}, false);
      auto addSubFunc = getOrInsertFunction(module, rewriter, "add_subsignal",
                                            addSubSignature);

      // get state pointer from arguments
      auto statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);

      // get signal pointer and offset
      auto sigDetail = insertProbeSignal(module, rewriter, &getDialect(), op,
                                         statePtr, transformed.target());
      auto sigPtr = sigDetail.first;
      auto offset = sigDetail.second;

      // adjust slice start point from signal's offset
      auto adjustedStart =
          rewriter.create<LLVM::AddOp>(op->getLoc(), offset, startConst);

      // shift pointer to the new start byte
      auto ptrToInt =
          rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, sigPtr);
      auto const8 = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), indexTy, rewriter.getI64IntegerAttr(8));
      auto ptrOffset =
          rewriter.create<LLVM::UDivOp>(op->getLoc(), adjustedStart, const8);
      auto shiftedPtr =
          rewriter.create<LLVM::AddOp>(op->getLoc(), ptrToInt, ptrOffset);
      auto newPtr =
          rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), i8PtrTy, shiftedPtr);

      // compute offset into the first byte
      auto bitOffset =
          rewriter.create<LLVM::URemOp>(op->getLoc(), adjustedStart, const8);

      // add subsignal to the state
      SmallVector<Value, 5> addSubArgs(
          {statePtr, transformed.target(), newPtr, lenConst, bitOffset});
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op, i32Ty, rewriter.getSymbolRefAttr(addSubFunc), addSubArgs);

      return success();
    }
    return failure();
  }
};

struct LLHDToLLVMLoweringPass
    : public ConvertLLHDToLLVMBase<LLHDToLLVMLoweringPass> {
  void runOnOperation() override;
};
} // namespace

void llhd::populateLLHDToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();

  // value manipulation conversion patterns
  patterns.insert<ConstOpConversion, ExtsOpConversion>(ctx, converter);
  // bitwise conversion patterns
  patterns.insert<NotOpConversion, ShrOpConversion, ShlOpConversion>(ctx,
                                                                     converter);
  // patterns.insert<NotOpConversion, AndOpConversion>(ctx, converter);
  patterns.insert<AndOpConversion, OrOpConversion, XorOpConversion>(converter);
  // unit conversion patterns
  patterns.insert<EntityOpConversion, TerminatorOpConversion, ProcOpConversion,
                  WaitOpConversion, HaltOpConversion>(ctx, converter);
  // signal conversion patterns
  patterns.insert<SigOpConversion, PrbOpConversion, DrvOpConversion>(ctx,
                                                                     converter);
}

void LLHDToLLVMLoweringPass::runOnOperation() {
  OwningRewritePatternList patterns;
  auto converter = mlir::LLVMTypeConverter(&getContext());

  // partial conversion of only the inst op
  patterns.insert<InstOpConversion>(&getContext(), converter);
  // define target
  LLVMConversionTarget target(getContext());
  target.addIllegalOp<InstOp>();

  // apply partial conversion
  if (failed(applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();

  // setup full conversion
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateLLHDToLLVMConversionPatterns(converter, patterns);

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

  // apply full conversion
  if (failed(applyFullConversion(getOperation(), target, patterns)))
    signalPassFailure();
}

/// Create an LLHD to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>>
mlir::llhd::createConvertLLHDToLLVMPass() {
  return std::make_unique<LLHDToLLVMLoweringPass>();
}
