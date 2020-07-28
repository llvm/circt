//===- LLHDToLLVM.cpp - LLHD to LLVM Conversion Pass ----------------------===//
//
// This is the main LLHD to LLVM Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LLHDToLLVM/LLHDToLLVM.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace llhd {
#define GEN_PASS_CLASSES
#include "circt/Conversion/LLHDToLLVM/Passes.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::llhd;

// Keep a counter to infer a signal's index in his entity's signal table.
static size_t signalCounter = 0;

static size_t regCounter = 0;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static unsigned getStdOrLLVMIntegerWidth(Type type) {
  if (auto llvmTy = type.dyn_cast<LLVM::LLVMType>())
    return llvmTy.getIntegerBitWidth();
  return type.getIntOrFloatBitWidth();
}

/// Get an existing global string.
static Value getGlobalString(Location loc, OpBuilder &builder,
                             LLVMTypeConverter &typeConverter,
                             LLVM::GlobalOp &str) {
  auto i8PtrTy = LLVM::LLVMType::getInt8PtrTy(&typeConverter.getContext());
  auto i32Ty = LLVM::LLVMType::getInt32Ty(&typeConverter.getContext());

  auto addr = builder.create<LLVM::AddressOfOp>(
      loc, str.getType().getPointerTo(), str.getName());
  auto idx = builder.create<LLVM::ConstantOp>(loc, i32Ty,
                                              builder.getI32IntegerAttr(0));
  std::array<Value, 2> idxs({idx, idx});
  return builder.create<LLVM::GEPOp>(loc, i8PtrTy, addr, idxs);
}

/// Looks up a symbol and inserts a new functino at the beginning of the
/// module's region in case the function does not exists. If
/// insertBodyAndTerminator is set, also adds the entry block and return
/// terminator.
static LLVM::LLVMFuncOp
getOrInsertFunction(ModuleOp &module, ConversionPatternRewriter &rewriter,
                    Location loc, std::string name, LLVM::LLVMType signature,
                    bool insertBodyAndTerminator = false) {
  auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<LLVM::LLVMFuncOp>(loc, name, signature);
    if (insertBodyAndTerminator) {
      func.addEntryBlock();
      OpBuilder b(func.getBody());
      b.create<LLVM::ReturnOp>(loc, ValueRange());
    }
  }
  return func;
}

/// Return the LLVM type used to represent a signal. It corresponds to a struct
/// with the format: {valuePtr, bitOffset, instanceIndex, globalIndex}.
static LLVM::LLVMType getSigType(LLVM::LLVMDialect *dialect) {
  auto i8PtrTy = LLVM::LLVMType::getInt8PtrTy(dialect->getContext());
  auto i64Ty = LLVM::LLVMType::getInt64Ty(dialect->getContext());
  return LLVM::LLVMType::getStructTy(i8PtrTy, i64Ty, i64Ty, i64Ty);
}

/// Extract the details from the given signal struct. The details are returned
/// in the original struct order.
static std::vector<Value> getSignalDetail(ConversionPatternRewriter &rewriter,
                                          LLVM::LLVMDialect *dialect,
                                          Location loc, Value signal,
                                          bool extractIndices = false) {

  auto i8PtrTy = LLVM::LLVMType::getInt8PtrTy(dialect->getContext());
  auto i32Ty = LLVM::LLVMType::getInt32Ty(dialect->getContext());
  auto i64Ty = LLVM::LLVMType::getInt64Ty(dialect->getContext());

  std::vector<Value> result;

  // Extract the value and offset elements.
  auto zeroC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto oneC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                rewriter.getI32IntegerAttr(1));

  auto sigPtrPtr = rewriter.create<LLVM::GEPOp>(
      loc, i8PtrTy.getPointerTo(), signal, ArrayRef<Value>({zeroC, zeroC}));
  result.push_back(rewriter.create<LLVM::LoadOp>(loc, i8PtrTy, sigPtrPtr));

  auto offsetPtr = rewriter.create<LLVM::GEPOp>(
      loc, i64Ty.getPointerTo(), signal, ArrayRef<Value>({zeroC, oneC}));
  result.push_back(rewriter.create<LLVM::LoadOp>(loc, i64Ty, offsetPtr));

  // Extract the instance and global indices.
  if (extractIndices) {
    auto twoC = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(2));
    auto threeC = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(3));

    auto instIndexPtr = rewriter.create<LLVM::GEPOp>(
        loc, i64Ty.getPointerTo(), signal, ArrayRef<Value>({zeroC, twoC}));
    result.push_back(rewriter.create<LLVM::LoadOp>(loc, i64Ty, instIndexPtr));

    auto globalIndexPtr = rewriter.create<LLVM::GEPOp>(
        loc, i64Ty.getPointerTo(), signal, ArrayRef<Value>({zeroC, threeC}));
    result.push_back(rewriter.create<LLVM::LoadOp>(loc, i64Ty, globalIndexPtr));
  }

  return result;
}

/// Create a subsignal struct.
static Value createSubSig(LLVM::LLVMDialect *dialect,
                          ConversionPatternRewriter &rewriter, Location loc,
                          std::vector<Value> originDetail, Value newPtr,
                          Value newOffset) {
  auto i32Ty = LLVM::LLVMType::getInt32Ty(dialect->getContext());
  auto sigTy = getSigType(dialect);

  // Create signal struct.
  auto sigUndef = rewriter.create<LLVM::UndefOp>(loc, sigTy);
  auto storeSubPtr = rewriter.create<LLVM::InsertValueOp>(
      loc, sigUndef, newPtr, rewriter.getI32ArrayAttr(0));
  auto storeSubOffset = rewriter.create<LLVM::InsertValueOp>(
      loc, storeSubPtr, newOffset, rewriter.getI32ArrayAttr(1));
  auto storeSubInstIndex = rewriter.create<LLVM::InsertValueOp>(
      loc, storeSubOffset, originDetail[2], rewriter.getI32ArrayAttr(2));
  auto storeSubGlobalIndex = rewriter.create<LLVM::InsertValueOp>(
      loc, storeSubInstIndex, originDetail[3], rewriter.getI32ArrayAttr(3));

  // Allocate and store the subsignal.
  auto oneC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                rewriter.getI32IntegerAttr(1));
  auto allocaSubSig =
      rewriter.create<LLVM::AllocaOp>(loc, sigTy.getPointerTo(), oneC, 4);
  rewriter.create<LLVM::StoreOp>(loc, storeSubGlobalIndex, allocaSubSig);

  return allocaSubSig;
}

/// Returns true if the given value is passed as an argument to the destination
/// block of the given WaitOp.
static bool isWaitDestArg(WaitOp op, Value val) {
  for (auto arg : op.destOps()) {
    if (arg == val)
      return true;
  }
  return false;
}

// Returns true if the given operation is used as a destination argument in a
// WaitOp.
static bool isWaitDestArg(Operation *op) {
  for (auto user : op->getUsers()) {
    if (auto wait = dyn_cast<WaitOp>(user))
      return isWaitDestArg(wait, op->getResult(0));
  }
  return false;
}

/// Gather the types of values that are used outside of the block they're
/// defined in. An LLVMType structure containing those types, in order of
/// appearance, is returned.
static LLVM::LLVMType getProcPersistenceTy(LLVM::LLVMDialect *dialect,
                                           LLVMTypeConverter &converter,
                                           ProcOp &proc) {
  SmallVector<LLVM::LLVMType, 3> types = SmallVector<LLVM::LLVMType, 3>();
  proc.walk([&](Operation *op) -> void {
    if (op->isUsedOutsideOfBlock(op->getBlock()) || isWaitDestArg(op)) {
      if (auto ptr = op->getResult(0).getType().dyn_cast<PtrType>()) {
        // Persist the unwrapped value.
        auto converted = converter.convertType(ptr.getUnderlyingType());
        types.push_back(converted.cast<LLVM::LLVMType>());
      } else if (op->getResult(0).getType().isa<SigType>()) {
        types.push_back(getSigType(dialect));
      } else {
        types.push_back(converter.convertType(op->getResult(0).getType())
                            .cast<LLVM::LLVMType>());
      }
    }
  });

  // Also persist block arguments escaping their defining block.
  for (auto &block : proc.getBlocks()) {
    if (block.isEntryBlock())
      continue;
    for (auto arg : block.getArguments()) {
      if (arg.isUsedOutsideOfBlock(&block)) {
        types.push_back(
            converter.convertType(arg.getType()).cast<LLVM::LLVMType>());
      }
    }
  }

  return LLVM::LLVMType::getStructTy(dialect->getContext(), types);
}

/// Insert a comparison block that either jumps to the trueDest block, if the
/// resume index mathces the current index, or to falseDest otherwise. If no
/// falseDest is provided, the next block is taken insead.
static void insertComparisonBlock(ConversionPatternRewriter &rewriter,
                                  LLVM::LLVMDialect *dialect, Location loc,
                                  Region *body, Value resumeIdx, int currIdx,
                                  Block *trueDest, ValueRange trueDestArgs,
                                  Block *falseDest = nullptr) {
  auto i32Ty = LLVM::LLVMType::getInt32Ty(dialect->getContext());
  auto secondBlock = ++body->begin();
  auto newBlock = rewriter.createBlock(body, secondBlock);
  auto cmpIdx = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(currIdx));
  auto cmpRes = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                              resumeIdx, cmpIdx);

  // Default to jumpumping to the next block for the false, if it is not
  // provided.
  if (!falseDest)
    falseDest = &*secondBlock;

  rewriter.create<LLVM::CondBrOp>(loc, cmpRes, trueDest, trueDestArgs,
                                  falseDest, ValueRange());

  // Redirect the entry block terminator to the new comparison block.
  auto entryTer = body->front().getTerminator();
  entryTer->setSuccessor(newBlock, 0);
}

/// Insert a GEP operation to the pointer of the i-th value in the process
/// persistence table.
static Value gepPersistenceState(LLVM::LLVMDialect *dialect, Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 LLVM::LLVMType elementTy, int index,
                                 Value state) {
  auto i32Ty = LLVM::LLVMType::getInt32Ty(dialect->getContext());
  auto zeroC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto threeC = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(3));
  auto indC = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(index));
  return rewriter.create<LLVM::GEPOp>(loc, elementTy.getPointerTo(), state,
                                      ArrayRef<Value>({zeroC, threeC, indC}));
}

/// Persist a `Value` by storing it into the process persistence table, and
/// substituting the uses that escape the block the operation is defined in with
/// a load from the persistence table.
static void persistValue(LLVM::LLVMDialect *dialect, Location loc,
                         LLVMTypeConverter &converter,
                         ConversionPatternRewriter &rewriter,
                         LLVM::LLVMType stateTy, int &i, Value state,
                         Value persist) {
  auto elemTy = stateTy.getStructElementType(3).getStructElementType(i);

  // Store the value escaping it's definingn block in the persistence table.
  if (auto arg = persist.dyn_cast<BlockArgument>()) {
    rewriter.setInsertionPointToStart(arg.getParentBlock());
  } else {
    rewriter.setInsertionPointAfter(persist.getDefiningOp());
  }

  auto gep0 = gepPersistenceState(dialect, loc, rewriter, elemTy, i, state);

  Value toStore;
  if (auto ptr = persist.getType().dyn_cast<PtrType>()) {
    // Redirect uses of the pointer in the same block to the pointer in the
    // persistence state. This ensures that stores and loads all operate on the
    // same value.
    for (auto &use : llvm::make_early_inc_range(persist.getUses())) {
      if (persist.getParentBlock() == use.getOwner()->getBlock())
        use.set(gep0);
    }
    // Unwrap the pointer and store it's value.
    auto elemTy = converter.convertType(ptr.getUnderlyingType());
    toStore = rewriter.create<LLVM::LoadOp>(loc, elemTy, persist);
  } else if (persist.getType().isa<SigType>()) {
    toStore = rewriter.create<LLVM::LoadOp>(loc, getSigType(dialect), persist);
  } else {
    // Store the value directly.
    toStore = persist;
  }

  rewriter.create<LLVM::StoreOp>(loc, toStore, gep0);

  // Load the value from the persistence table and substitute the original
  // use with it, whenever it is in a different block.
  for (auto &use : llvm::make_early_inc_range(persist.getUses())) {
    auto user = use.getOwner();
    if (persist.getParentBlock() != user->getBlock() ||
        (isa<WaitOp>(user) && isWaitDestArg(cast<WaitOp>(user), persist))) {
      if (isa<WaitOp>(user) && isWaitDestArg(cast<WaitOp>(user), persist))
        rewriter.setInsertionPoint(
            user->getParentRegion()->front().getTerminator());
      else
        rewriter.setInsertionPointToStart(user->getBlock());

      auto gep1 = gepPersistenceState(dialect, loc, rewriter, elemTy, i, state);
      // Use the pointer in the state struct directly for pointer and signal
      // types.
      if (persist.getType().isa<PtrType, SigType>()) {
        use.set(gep1);
      } else {
        auto load1 = rewriter.create<LLVM::LoadOp>(loc, elemTy, gep1);
        // Load the value otherwise.
        use.set(load1);
      }
    }
  }
  i++;
}

/// Insert the blocks and operations needed to persist values across suspension,
/// as well as ones needed to resume execution at the right spot.
static void insertPersistence(LLVMTypeConverter &converter,
                              ConversionPatternRewriter &rewriter,
                              LLVM::LLVMDialect *dialect, Location loc,
                              ProcOp &proc, LLVM::LLVMType &stateTy,
                              LLVM::LLVMFuncOp &converted,
                              Operation *splitEntryBefore) {
  auto i32Ty = LLVM::LLVMType::getInt32Ty(dialect->getContext());

  auto &firstBB = converted.getBody().front();

  auto splitFirst = firstBB.splitBlock(splitEntryBefore);

  rewriter.setInsertionPointToEnd(&firstBB);
  rewriter.create<LLVM::BrOp>(loc, ValueRange(), splitFirst);

  // Load the resume index from the process state argument.
  rewriter.setInsertionPoint(converted.getBody().front().getTerminator());
  auto zeroC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto oneC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                rewriter.getI32IntegerAttr(1));
  auto gep = rewriter.create<LLVM::GEPOp>(loc, i32Ty.getPointerTo(),
                                          converted.getArgument(1),
                                          ArrayRef<Value>({zeroC, oneC}));

  auto larg = rewriter.create<LLVM::LoadOp>(loc, i32Ty, gep);

  // Insert an abort block as the last block.
  auto abortBlock =
      rewriter.createBlock(&converted.getBody(), converted.getBody().end());
  rewriter.create<LLVM::ReturnOp>(loc, ValueRange());

  auto body = &converted.getBody();

  // Redirect the entry block to a first comparison block. If on a fresh start,
  // start from where original entry would have jumped, else the process is in
  // an illegal state and jump to the abort block.
  insertComparisonBlock(rewriter, dialect, loc, body, larg, 0,
                        body->front().getSuccessor(0), ValueRange(),
                        abortBlock);

  // Keep track of the index in the presistence table of the operation we
  // are currently processing.
  int i = 0;
  // Keep track of the current resume index for comparison blocks.
  int waitInd = 0;

  // Insert operations required to persist values across process suspension.
  converted.walk([&](Operation *op) -> void {
    if ((op->isUsedOutsideOfBlock(op->getBlock()) || isWaitDestArg(op)) &&
        op->getResult(0) != larg.getResult()) {
      persistValue(dialect, loc, converter, rewriter, stateTy, i,
                   converted.getArgument(1), op->getResult(0));
    }

    // Insert a comparison block for wait operations.
    if (auto wait = dyn_cast<WaitOp>(op)) {
      insertComparisonBlock(rewriter, dialect, loc, body, larg, ++waitInd,
                            wait.dest(), wait.destOps());

      // Insert the resume index update at the wait operation location.
      rewriter.setInsertionPoint(op);
      auto procState = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(1);
      auto resumeIdxC = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(waitInd));
      auto resumeIdxPtr = rewriter.create<LLVM::GEPOp>(
          loc, i32Ty.getPointerTo(), procState, ArrayRef<Value>({zeroC, oneC}));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), resumeIdxC, resumeIdxPtr);
    }
  });

  // Also persist argument blocks escaping their defining block.
  for (auto &block : converted.getBlocks()) {
    if (block.isEntryBlock())
      continue;
    for (auto arg : block.getArguments()) {
      if (arg.isUsedOutsideOfBlock(&block)) {
        persistValue(dialect, loc, converter, rewriter, stateTy, i,
                     converted.getArgument(1), arg);
      }
    }
  }
}

/// Return a struct type of arrays containing one entry for each RegOp condition
/// that needs more than one state of the trigger to infer it (i.e. `both`,
/// `rise` and `fall`).
static LLVM::LLVMType getRegStateTy(LLVM::LLVMDialect *dialect,
                                    Operation *unit) {
  SmallVector<LLVM::LLVMType, 4> types;
  unit->walk([&](RegOp op) {
    size_t count = 0;
    for (size_t i = 0; i < op.modes().size(); ++i) {
      auto mode = op.getRegModeAt(i);
      if (mode == RegMode::fall || mode == RegMode::rise ||
          mode == RegMode::both)
        ++count;
    }
    if (count > 0)
      types.push_back(LLVM::LLVMType::getArrayTy(
          LLVM::LLVMType::getInt1Ty(dialect->getContext()), count));
  });
  return LLVM::LLVMType::getStructTy(dialect->getContext(), types);
}

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

static LLVM::LLVMType convertSigType(SigType type,
                                     LLVMTypeConverter &converter) {
  auto i64Ty = LLVM::LLVMType::getInt64Ty(&converter.getContext());
  auto i8PtrTy = LLVM::LLVMType::getInt8PtrTy(&converter.getContext());
  return LLVM::LLVMType::getStructTy(i8PtrTy, i64Ty, i64Ty, i64Ty)
      .getPointerTo();
}

static LLVM::LLVMType convertTimeType(TimeType type,
                                      LLVMTypeConverter &converter) {
  auto i32Ty = LLVM::LLVMType::getInt32Ty(&converter.getContext());
  return LLVM::LLVMType::getArrayTy(i32Ty, 3);
}

static LLVM::LLVMType convertPtrType(PtrType type,
                                     LLVMTypeConverter &converter) {
  return converter.convertType(type.getUnderlyingType())
      .cast<LLVM::LLVMType>()
      .getPointerTo();
}

//===----------------------------------------------------------------------===//
// Unit conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert an `llhd.entity` unit to LLVM dialect. The result is an `llvm.func`
/// which takes a pointer to the state as arguments.
struct EntityOpConversion : public ConvertToLLVMPattern {
  explicit EntityOpConversion(MLIRContext *ctx,
                              LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::EntityOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get adapted operands.
    EntityOpAdaptor transformed(operands);
    // Get entity operation.
    auto entityOp = cast<EntityOp>(op);

    // Collect used llvm types.
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&typeConverter.getContext());
    auto sigTy = getSigType(&getDialect());
    auto entityStatePtrTy = getRegStateTy(&getDialect(), op).getPointerTo();

    regCounter = 0;

    // Use an intermediate signature conversion to add the arguments for the
    // state and signal table pointer arguments.
    LLVMTypeConverter::SignatureConversion intermediate(
        entityOp.getNumArguments());
    // Add state and signal table arguments.
    intermediate.addInputs(
        std::array<Type, 3>({i8PtrTy, entityStatePtrTy, sigTy.getPointerTo()}));
    for (size_t i = 0, e = entityOp.getNumArguments(); i < e; ++i)
      intermediate.addInputs(i, voidTy);
    rewriter.applySignatureConversion(&entityOp.getBody(), intermediate);

    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&entityOp.getBlocks().front());
    LLVMTypeConverter::SignatureConversion final(
        intermediate.getConvertedTypes().size());
    final.addInputs(0, i8PtrTy);
    final.addInputs(1, entityStatePtrTy);
    final.addInputs(2, sigTy.getPointerTo());

    // The first n elements of the signal table represent the entity arguments,
    // while the remaining elements represent the entity's owned signals.
    signalCounter = entityOp.getNumArguments();
    for (size_t i = 0; i < signalCounter; ++i) {
      // Create gep operations from the signal table for each original argument.
      auto index = bodyBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto gep = bodyBuilder.create<LLVM::GEPOp>(
          op->getLoc(), sigTy.getPointerTo(), entityOp.getArgument(2),
          ArrayRef<Value>(index));
      // Remap i-th original argument to the gep'd signal pointer.
      final.remapInput(i + 3, gep.getResult());
    }

    rewriter.applySignatureConversion(&entityOp.getBody(), final);

    // Get the converted entity signature.
    auto funcTy = LLVM::LLVMType::getFunctionTy(
        voidTy, {i8PtrTy, entityStatePtrTy, sigTy.getPointerTo()},
        /*isVarArg=*/false);

    // Create the a new llvm function to house the lowered entity.
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(
        op->getLoc(), entityOp.getName(), funcTy);

    // Inline the entity region in the new llvm function.
    rewriter.inlineRegionBefore(entityOp.getBody(), llvmFunc.getBody(),
                                llvmFunc.end());

    // Erase the original operation.
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

namespace {
/// Convert an `"llhd.terminator" operation to `llvm.return`.
struct TerminatorOpConversion : public ConvertToLLVMPattern {
  explicit TerminatorOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::TerminatorOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Just replace the original op with return void.
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange());

    return success();
  }
};
} // namespace

namespace {
/// Convert an `llhd.proc` operation to LLVM dialect. This inserts the required
/// logic to resume execution after an `llhd.wait` operation, as well as state
/// keeping for values that need to persist across suspension.
struct ProcOpConversion : public ConvertToLLVMPattern {
  explicit ProcOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(ProcOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto procOp = cast<ProcOp>(op);

    // Get adapted operands.
    ProcOpAdaptor transformed(operands);

    // Collect used llvm types.
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = LLVM::LLVMType::getInt1Ty(&typeConverter.getContext());
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&typeConverter.getContext());
    auto senseTableTy =
        LLVM::LLVMType::getArrayTy(i1Ty, procOp.getNumArguments())
            .getPointerTo();
    auto stateTy = LLVM::LLVMType::getStructTy(
        /* current instance  */ i8PtrTy, /* resume index */ i32Ty,
        /* sense flags */ senseTableTy, /* persistent types */
        getProcPersistenceTy(&getDialect(), typeConverter, procOp));
    auto sigTy = getSigType(&getDialect());

    // Keep track of the original first operation of the process, to know where
    // to split the first block to insert comparison blocks.
    auto &firstOp = op->getRegion(0).front().front();

    // Have an intermediate signature conversion to add the arguments for the
    // state, process-specific state and signal table.
    LLVMTypeConverter::SignatureConversion intermediate(
        procOp.getNumArguments());
    // Add state, process state table and signal table arguments.
    std::array<Type, 3> procArgTys(
        {i8PtrTy, stateTy.getPointerTo(), sigTy.getPointerTo()});
    intermediate.addInputs(procArgTys);
    for (size_t i = 0, e = procOp.getNumArguments(); i < e; ++i)
      intermediate.addInputs(i, voidTy);
    rewriter.applySignatureConversion(&procOp.getBody(), intermediate);

    // Get the final signature conversion.
    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&procOp.getBlocks().front());
    LLVMTypeConverter::SignatureConversion final(
        intermediate.getConvertedTypes().size());
    final.addInputs(0, i8PtrTy);
    final.addInputs(1, stateTy.getPointerTo());
    final.addInputs(2, sigTy.getPointerTo());

    for (size_t i = 0, e = procOp.getNumArguments(); i < e; ++i) {
      // Create gep operations from the signal table for each original argument.
      auto index = bodyBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto gep = bodyBuilder.create<LLVM::GEPOp>(
          op->getLoc(), sigTy.getPointerTo(), procOp.getArgument(2),
          ArrayRef<Value>({index}));

      // Remap the i-th original argument to the gep'd value.
      final.remapInput(i + 3, gep.getResult());
    }

    // Get the converted process signature.
    auto funcTy = LLVM::LLVMType::getFunctionTy(
        voidTy, {i8PtrTy, stateTy.getPointerTo(), sigTy.getPointerTo()},
        /*isVarArg=*/false);
    // Create a new llvm function to house the lowered process.
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(),
                                                      procOp.getName(), funcTy);

    // Inline the process region in the new llvm function.
    rewriter.inlineRegionBefore(procOp.getBody(), llvmFunc.getBody(),
                                llvmFunc.end());

    insertPersistence(typeConverter, rewriter, &getDialect(), op->getLoc(),
                      procOp, stateTy, llvmFunc, &firstOp);

    // Convert the block argument types after inserting the persistence, since
    // this messes up the block argument uses.
    if (failed(rewriter.convertRegionTypes(&llvmFunc.getBody(), typeConverter,
                                           &final))) {
      return failure();
    }

    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

namespace {
/// Convert an `llhd.halt` operation to LLVM dialect. This zeroes out all the
/// senses and returns, effectively making the process unable to be invoked
/// again.
struct HaltOpConversion : public ConvertToLLVMPattern {
  explicit HaltOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(HaltOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1Ty = LLVM::LLVMType::getInt1Ty(&typeConverter.getContext());
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&typeConverter.getContext());

    auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto procState = llvmFunc.getArgument(1);
    auto senseTableTy = procState.getType()
                            .cast<LLVM::LLVMType>()
                            .getPointerElementTy()
                            .getStructElementType(2)
                            .getPointerElementTy();

    // Get senses ptr from the process state argument.
    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
    auto twoC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(2));
    auto sensePtrGep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), senseTableTy.getPointerTo().getPointerTo(), procState,
        ArrayRef<Value>({zeroC, twoC}));
    auto sensePtr = rewriter.create<LLVM::LoadOp>(
        op->getLoc(), senseTableTy.getPointerTo(), sensePtrGep);

    // Zero out all the senses flags.
    for (size_t i = 0, e = senseTableTy.getArrayNumElements(); i < e; ++i) {
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
} // namespace

namespace {
/// Convert an `llhd.wait` operation to LLVM dialect. This sets the current
/// resume point, sets the observed senses (if present) and schedules the timed
/// wake up (if present).
struct WaitOpConversion : public ConvertToLLVMPattern {
  explicit WaitOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(WaitOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto waitOp = cast<WaitOp>(op);
    WaitOpAdaptor transformed(operands, op->getAttrDictionary());
    auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>();

    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = LLVM::LLVMType::getInt1Ty(&typeConverter.getContext());
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&typeConverter.getContext());
    auto i64Ty = LLVM::LLVMType::getInt64Ty(&typeConverter.getContext());

    // Get the llhd_suspend runtime function.
    auto llhdSuspendTy = LLVM::LLVMType::getFunctionTy(
        voidTy, {i8PtrTy, i8PtrTy, i32Ty, i32Ty, i32Ty}, /*isVarArg=*/false);
    auto module = op->getParentOfType<ModuleOp>();
    auto llhdSuspendFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                               "llhd_suspend", llhdSuspendTy);

    auto statePtr = llvmFunc.getArgument(0);
    auto procState = llvmFunc.getArgument(1);
    auto procStateTy = procState.getType().dyn_cast<LLVM::LLVMType>();
    auto senseTableTy = procStateTy.getPointerElementTy()
                            .getStructElementType(2)
                            .getPointerElementTy();

    // Get senses ptr.
    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
    auto twoC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(2));
    auto sensePtrGep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), senseTableTy.getPointerTo().getPointerTo(), procState,
        ArrayRef<Value>({zeroC, twoC}));
    auto sensePtr = rewriter.create<LLVM::LoadOp>(
        op->getLoc(), senseTableTy.getPointerTo(), sensePtrGep);

    // Reset sense table, if not all signals are observed.
    if (waitOp.obs().size() < senseTableTy.getArrayNumElements()) {
      auto zeroB = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getBoolAttr(false));
      for (size_t i = 0, e = senseTableTy.getArrayNumElements(); i < e; ++i) {
        auto indC = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
        auto senseElemPtr = rewriter.create<LLVM::GEPOp>(
            op->getLoc(), i1Ty.getPointerTo(), sensePtr,
            ArrayRef<Value>({zeroC, indC}));
        rewriter.create<LLVM::StoreOp>(op->getLoc(), zeroB, senseElemPtr);
      }
    }

    // Set sense flags for observed signals.
    for (auto observation : transformed.obs()) {
      auto instIndexPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), i64Ty.getPointerTo(), observation,
          ArrayRef<Value>({zeroC, twoC}));
      auto instIndex =
          rewriter.create<LLVM::LoadOp>(op->getLoc(), i64Ty, instIndexPtr);
      auto oneB = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                    rewriter.getBoolAttr(true));
      auto senseElementPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), i1Ty.getPointerTo(), sensePtr,
          ArrayRef<Value>({zeroC, instIndex}));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), oneB, senseElementPtr);
    }

    // Update and store the new resume index in the process state.
    auto procStateBC =
        rewriter.create<LLVM::BitcastOp>(op->getLoc(), i8PtrTy, procState);

    // Spawn scheduled event, if present.
    if (waitOp.time()) {
      auto realTime = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), i32Ty, transformed.time(),
          rewriter.getI32ArrayAttr(ArrayRef<int32_t>({0})));
      auto delta = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), i32Ty, transformed.time(),
          rewriter.getI32ArrayAttr(ArrayRef<int32_t>({1})));
      auto eps = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), i32Ty, transformed.time(),
          rewriter.getI32ArrayAttr(ArrayRef<int32_t>({2})));

      std::array<Value, 5> args({statePtr, procStateBC, realTime, delta, eps});
      rewriter.create<LLVM::CallOp>(op->getLoc(), voidTy,
                                    rewriter.getSymbolRefAttr(llhdSuspendFunc),
                                    args);
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange());
    return success();
  }
};
} // namespace

namespace {
/// Lower an llhd.inst operation to LLVM dialect. This generates malloc calls
/// and alloc_signal calls (to store the pointer into the state) for each signal
/// in the instantiated entity.
struct InstOpConversion : public ConvertToLLVMPattern {
  explicit InstOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(InstOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the inst operation.
    auto instOp = cast<InstOp>(op);
    // Get the parent module.
    auto module = op->getParentOfType<ModuleOp>();

    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = LLVM::LLVMType::getInt1Ty(&typeConverter.getContext());
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&typeConverter.getContext());
    auto i64Ty = LLVM::LLVMType::getInt64Ty(&typeConverter.getContext());

    // Init function signature: (i8* %state) -> void.
    auto initFuncTy =
        LLVM::LLVMType::getFunctionTy(voidTy, {i8PtrTy}, /*isVarArg=*/false);
    auto initFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(), "llhd_init",
                            initFuncTy, /*insertBodyAndTerminator=*/true);

    // Get or insert the malloc function definition.
    // Malloc function signature: (i64 %size) -> i8* %pointer.
    auto mallocSigFuncTy =
        LLVM::LLVMType::getFunctionTy(i8PtrTy, {i64Ty}, /*isVarArg=*/false);
    auto mallFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                        "malloc", mallocSigFuncTy);

    // Get or insert the alloc_signal library call definition.
    // Alloc_signal function signature: (i8* %state, i8* %sig_name, i8*
    // %sig_owner, i32 %value) -> i32 %sig_index.
    auto allocSigFuncTy = LLVM::LLVMType::getFunctionTy(
        i32Ty, {i8PtrTy, i32Ty, i8PtrTy, i8PtrTy, i64Ty}, /*isVarArg=*/false);
    auto sigFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                       "alloc_signal", allocSigFuncTy);

    // Get or insert Alloc_proc library call definition.
    auto allocProcFuncTy = LLVM::LLVMType::getFunctionTy(
        voidTy, {i8PtrTy, i8PtrTy, i8PtrTy}, /*isVarArg=*/false);
    auto allocProcFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                             "alloc_proc", allocProcFuncTy);

    // Get or insert alloc_entity library call definition.
    auto allocEntityFuncTy = LLVM::LLVMType::getFunctionTy(
        voidTy, {i8PtrTy, i8PtrTy, i8PtrTy}, /*isVarArg=*/false);
    auto allocEntityFunc = getOrInsertFunction(
        module, rewriter, op->getLoc(), "alloc_entity", allocEntityFuncTy);

    Value initStatePtr = initFunc.getArgument(0);

    // Get a builder for the init function.
    OpBuilder initBuilder =
        OpBuilder::atBlockTerminator(&initFunc.getBody().getBlocks().front());

    // Use the instance name to retrieve the instance from the state.
    auto ownerName = instOp.name();
    // Get or create owner name string
    Value owner;
    auto parentSym =
        module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName.str());
    if (!parentSym) {
      owner = LLVM::createGlobalString(
          op->getLoc(), initBuilder, "instance." + ownerName.str(),
          ownerName.str() + '\0', LLVM::Linkage::Internal);
      parentSym =
          module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName.str());
    } else {
      owner =
          getGlobalString(op->getLoc(), initBuilder, typeConverter, parentSym);
    }

    // Handle entity instantiation.
    if (auto child = module.lookupSymbol<EntityOp>(instOp.callee())) {
      auto regStateTy = getRegStateTy(&getDialect(), child.getOperation());
      auto regStatePtrTy = regStateTy.getPointerTo();
      auto oneC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
      auto regNull =
          initBuilder.create<LLVM::NullOp>(op->getLoc(), regStatePtrTy);
      auto regGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), regStatePtrTy, regNull, ArrayRef<Value>({oneC}));
      auto regSize =
          initBuilder.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, regGep);
      auto regMall =
          initBuilder
              .create<LLVM::CallOp>(op->getLoc(), i8PtrTy,
                                    rewriter.getSymbolRefAttr(mallFunc),
                                    ArrayRef<Value>({regSize}))
              .getResult(0);
      auto regMallBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), regStatePtrTy, regMall);
      auto zeroB = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getBoolAttr(false));
      for (size_t i = 0, e = regStateTy.getStructNumElements(); i < e; ++i) {
        size_t f = regStateTy.getStructElementType(i).getArrayNumElements();
        for (size_t j = 0; j < f; ++j) {
          auto regIndexC = initBuilder.create<LLVM::ConstantOp>(
              op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
          auto triggerIndexC = initBuilder.create<LLVM::ConstantOp>(
              op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(j));
          auto regGep = initBuilder.create<LLVM::GEPOp>(
              op->getLoc(), i1Ty.getPointerTo(), regMallBC,
              ArrayRef<Value>({zeroB, regIndexC, triggerIndexC}));
          initBuilder.create<LLVM::StoreOp>(op->getLoc(), zeroB, regGep);
        }
      }
      initBuilder.create<LLVM::CallOp>(
          op->getLoc(), voidTy, rewriter.getSymbolRefAttr(allocEntityFunc),
          ArrayRef<Value>({initStatePtr, owner, regMall}));
      // Index of the signal in the unit's signal table.
      int initCounter = 0;
      // Walk over the entity and generate mallocs for each one of its signals.
      child.walk([&](Operation *op) -> void {
        if (auto sigOp = dyn_cast<SigOp>(op)) {
          // Get index constant of the signal in the unit's signal table.
          auto indexConst = initBuilder.create<LLVM::ConstantOp>(
              op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(initCounter));
          initCounter++;

          // Clone and insert the operation that defines the signal's init
          // operand (assmued to be a constant op)
          auto initDef =
              initBuilder.insert(sigOp.init().getDefiningOp()->clone())
                  ->getResult(0);
          // Get the required space, in bytes, to store the signal's value.
          int size = llvm::divideCeil(
              getStdOrLLVMIntegerWidth(sigOp.init().getType()), 8);
          auto sizeConst = initBuilder.create<LLVM::ConstantOp>(
              op->getLoc(), i64Ty, rewriter.getI64IntegerAttr(size));
          // Malloc double the required space to make sure signal shifts do not
          // segfault.
          auto mallocSize = initBuilder.create<LLVM::ConstantOp>(
              op->getLoc(), i64Ty, rewriter.getI64IntegerAttr(size * 2));
          std::array<Value, 1> margs({mallocSize});
          auto mall = initBuilder
                          .create<LLVM::CallOp>(
                              op->getLoc(), i8PtrTy,
                              rewriter.getSymbolRefAttr(mallFunc), margs)
                          .getResult(0);
          // Store the initial value.
          auto bitcast = initBuilder.create<LLVM::BitcastOp>(
              op->getLoc(),
              typeConverter.convertType(sigOp.init().getType())
                  .cast<LLVM::LLVMType>()
                  .getPointerTo(),
              mall);
          initBuilder.create<LLVM::StoreOp>(op->getLoc(), initDef, bitcast);

          std::array<Value, 5> args(
              {initStatePtr, indexConst, owner, mall, sizeConst});
          initBuilder.create<LLVM::CallOp>(
              op->getLoc(), i32Ty, rewriter.getSymbolRefAttr(sigFunc), args);
        }
      });
    } else if (auto proc = module.lookupSymbol<ProcOp>(instOp.callee())) {
      // Handle process instantiation.
      auto sensesPtrTy =
          LLVM::LLVMType::getArrayTy(i1Ty, proc.getNumArguments())
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

      // Malloc space for the process state.
      auto procStateNullPtr =
          initBuilder.create<LLVM::NullOp>(op->getLoc(), procStatePtrTy);
      auto procStateGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), procStatePtrTy, procStateNullPtr,
          ArrayRef<Value>({oneC}));
      auto procStateSize = initBuilder.create<LLVM::PtrToIntOp>(
          op->getLoc(), i64Ty, procStateGep);
      std::array<Value, 1> procStateMArgs({procStateSize});
      auto procStateMall =
          initBuilder
              .create<LLVM::CallOp>(op->getLoc(), i8PtrTy,
                                    rewriter.getSymbolRefAttr(mallFunc),
                                    procStateMArgs)
              .getResult(0);

      auto procStateBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), procStatePtrTy, procStateMall);

      // Malloc space for owner name.
      auto strSizeC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i64Ty, rewriter.getI64IntegerAttr(ownerName.size()));

      std::array<Value, 1> strMallArgs({strSizeC});
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

      // Store the initial resume index.
      auto resumeGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), i32Ty.getPointerTo(), procStateBC,
          ArrayRef<Value>({zeroC, oneC}));
      initBuilder.create<LLVM::StoreOp>(op->getLoc(), zeroC, resumeGep);

      // Malloc space for the senses table.
      auto sensesNullPtr =
          initBuilder.create<LLVM::NullOp>(op->getLoc(), sensesPtrTy);
      auto sensesGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), sensesPtrTy, sensesNullPtr, ArrayRef<Value>({oneC}));
      auto sensesSize =
          initBuilder.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, sensesGep);
      std::array<Value, 1> senseMArgs({sensesSize});
      auto sensesMall = initBuilder
                            .create<LLVM::CallOp>(
                                op->getLoc(), i8PtrTy,
                                rewriter.getSymbolRefAttr(mallFunc), senseMArgs)
                            .getResult(0);

      auto sensesBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), sensesPtrTy, sensesMall);

      // Set all initial senses to 1.
      for (size_t i = 0,
                  e = sensesPtrTy.getPointerElementTy().getArrayNumElements();
           i < e; ++i) {
        auto oneB = initBuilder.create<LLVM::ConstantOp>(
            op->getLoc(), i1Ty, rewriter.getBoolAttr(true));
        auto gepInd = initBuilder.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
        auto senseGep = initBuilder.create<LLVM::GEPOp>(
            op->getLoc(), i1Ty.getPointerTo(), sensesBC,
            ArrayRef<Value>({zeroC, gepInd}));
        initBuilder.create<LLVM::StoreOp>(op->getLoc(), oneB, senseGep);
      }

      // Store the senses pointer in the process state.
      auto procStateSensesPtr = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), sensesPtrTy.getPointerTo(), procStateBC,
          ArrayRef<Value>({zeroC, twoC}));
      initBuilder.create<LLVM::StoreOp>(op->getLoc(), sensesBC,
                                        procStateSensesPtr);

      std::array<Value, 3> allocProcArgs({initStatePtr, owner, procStateMall});
      initBuilder.create<LLVM::CallOp>(op->getLoc(), voidTy,
                                       rewriter.getSymbolRefAttr(allocProcFunc),
                                       allocProcArgs);
    }

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Signal conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert an `llhd.sig` operation to LLVM dialect. The i-th signal of an
/// entity get's lowered to a load of the i-th element of the signal table,
/// passed as an argument.
struct SigOpConversion : public ConvertToLLVMPattern {
  explicit SigOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::SigOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the adapted opreands.
    SigOpAdaptor transformed(operands);

    // Collect the used llvm types.
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&typeConverter.getContext());
    auto sigTy = getSigType(&getDialect());

    // Get the signal table pointer from the arguments.
    Value sigTablePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(2);

    // Get the index in the signal table and increase counter.
    auto indexConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(signalCounter));
    signalCounter++;

    // Insert a gep to the signal index in the signal table argument.
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, sigTy.getPointerTo(), sigTablePtr, ArrayRef<Value>(indexConst));

    return success();
  }
};
} // namespace

namespace {
/// Convert an `llhd.prb` operation to LLVM dialect. The result is a library
/// call to the
/// `@probe_signal` function. The signal details are then extracted and used to
/// load the final probe value.
struct PrbOpConversion : public ConvertToLLVMPattern {
  explicit PrbOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::PrbOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the adapted operands.
    PrbOpAdaptor transformed(operands);
    // Get the prb operation.
    auto prbOp = cast<PrbOp>(op);

    // Collect the used llvm types.
    auto finalTy =
        typeConverter.convertType(prbOp.getType()).cast<LLVM::LLVMType>();

    // Get the amount of bytes to load. An extra byte is always loaded to cover
    // the case where a subsignal spans halfway in the last byte.
    int resWidth = getStdOrLLVMIntegerWidth(prbOp.getType());
    int loadWidth = (llvm::divideCeil(resWidth, 8) + 1) * 8;
    auto loadTy =
        LLVM::LLVMType::getIntNTy(&typeConverter.getContext(), loadWidth);

    // Get the signal details from the signal struct.
    auto sigDetail = getSignalDetail(rewriter, &getDialect(), op->getLoc(),
                                     transformed.signal());

    auto bitcast = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), loadTy.getPointerTo(), sigDetail[0]);
    auto loadSig = rewriter.create<LLVM::LoadOp>(op->getLoc(), loadTy, bitcast);

    // TODO: cover the case of loadTy being larger than 64 bits (zext)
    // Shift the loaded value by the offset and truncate to the final width.
    auto trOff =
        rewriter.create<LLVM::TruncOp>(op->getLoc(), loadTy, sigDetail[1]);
    auto shifted =
        rewriter.create<LLVM::LShrOp>(op->getLoc(), loadTy, loadSig, trOff);
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, finalTy, shifted);

    return success();
  }
};
} // namespace

namespace {
/// Convert an `llhd.drv` operation to LLVM dialect. The result is a library
/// call to the
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
    // Get the adapted operands.
    DrvOpAdaptor transformed(operands);
    // Get the drive operation.
    auto drvOp = cast<DrvOp>(op);
    // Get the parent module.
    auto module = op->getParentOfType<ModuleOp>();

    // Collect used llvm types.
    auto voidTy = getVoidType();
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = LLVM::LLVMType::getInt1Ty(&typeConverter.getContext());
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&typeConverter.getContext());
    auto i64Ty = LLVM::LLVMType::getInt64Ty(&typeConverter.getContext());
    auto sigTy = getSigType(&getDialect());

    // Get or insert the drive library call.
    auto drvFuncTy = LLVM::LLVMType::getFunctionTy(
        voidTy,
        {i8PtrTy, sigTy.getPointerTo(), i8PtrTy, i64Ty, i32Ty, i32Ty, i32Ty},
        /*isVarArg=*/false);
    auto drvFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                       "drive_signal", drvFuncTy);

    // Get the state pointer from the function arguments.
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);

    int sigWidth = getStdOrLLVMIntegerWidth(
        drvOp.signal().getType().dyn_cast<SigType>().getUnderlyingType());

    // Insert enable comparison. Skip if the enable operand is 0.
    if (auto gate = drvOp.enable()) {
      auto block = op->getBlock();
      auto continueBlock = block->splitBlock(op);
      auto drvBlock = rewriter.createBlock(continueBlock);
      rewriter.setInsertionPointToEnd(drvBlock);
      rewriter.create<LLVM::BrOp>(op->getLoc(), ValueRange(), continueBlock);

      rewriter.setInsertionPointToEnd(block);
      auto oneC = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getI16IntegerAttr(1));
      auto cmp = rewriter.create<LLVM::ICmpOp>(
          op->getLoc(), LLVM::ICmpPredicate::eq, transformed.enable(), oneC);
      rewriter.create<LLVM::CondBrOp>(op->getLoc(), cmp, drvBlock,
                                      continueBlock);

      rewriter.setInsertionPointToStart(drvBlock);
    }

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

    // Get the time values.
    auto realTime = rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), i32Ty, transformed.time(),
        rewriter.getI32ArrayAttr(ArrayRef<int32_t>({0})));
    auto delta = rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), i32Ty, transformed.time(),
        rewriter.getI32ArrayAttr(ArrayRef<int32_t>({1})));
    auto eps = rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), i32Ty, transformed.time(),
        rewriter.getI32ArrayAttr(ArrayRef<int32_t>({2})));

    // Define the drive_signal library call arguments.
    std::array<Value, 7> args(
        {statePtr, transformed.signal(), bc, widthConst, realTime, delta, eps});
    // Create the library call.
    rewriter.create<LLVM::CallOp>(op->getLoc(), voidTy,
                                  rewriter.getSymbolRefAttr(drvFunc), args);

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
/// Convert an `llhd.reg` operation to LLVM dialect. This generates a series of
/// comparisons (blocks) that end up driving the signal with apropriate
/// arguments.
struct RegOpConversion : public ConvertToLLVMPattern {
  explicit RegOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(RegOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto regOp = cast<RegOp>(op);
    RegOpAdaptor transformed(operands, op->getAttrDictionary());

    auto i1Ty = LLVM::LLVMType::getInt1Ty(&typeConverter.getContext());
    auto i32Ty = LLVM::LLVMType::getInt32Ty(&typeConverter.getContext());

    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();

    // Retrieve and update previous trigger values for rising/falling edge
    // detection.
    SmallVector<Value, 4> prevTriggers;
    for (int i = 0, e = regOp.values().size(); i < e; ++i) {
      auto mode = regOp.getRegModeAt(i);
      if (mode == RegMode::both || mode == RegMode::fall ||
          mode == RegMode::rise) {
        auto zeroC = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
        auto regIndexC = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(regCounter));
        auto triggerIndexC = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
        auto gep = rewriter.create<LLVM::GEPOp>(
            op->getLoc(), i1Ty.getPointerTo(), func.getArgument(1),
            ArrayRef<Value>({zeroC, regIndexC, triggerIndexC}));
        prevTriggers.push_back(
            rewriter.create<LLVM::LoadOp>(op->getLoc(), gep));
        rewriter.create<LLVM::StoreOp>(op->getLoc(), transformed.triggers()[i],
                                       gep);
      }
    }
    // Create blocks for drive and continue
    auto block = op->getBlock();
    auto continueBlock = block->splitBlock(op);

    auto drvBlock = rewriter.createBlock(continueBlock);
    auto valArg = drvBlock->addArgument(transformed.values()[0].getType());
    auto delayArg = drvBlock->addArgument(transformed.delays()[0].getType());
    auto gateArg = drvBlock->addArgument(i1Ty);

    // Create a drive with the block arguments.
    rewriter.setInsertionPointToStart(drvBlock);
    rewriter.create<DrvOp>(op->getLoc(), regOp.signal(), valArg, delayArg,
                           gateArg);
    rewriter.create<LLVM::BrOp>(op->getLoc(), ValueRange(), continueBlock);

    int j = prevTriggers.size() - 1;
    // Create a comparison block for each of the reg tuples.
    for (int i = regOp.values().size() - 1, e = i; i >= 0; --i) {
      auto cmpBlock = rewriter.createBlock(block->getNextNode());
      rewriter.setInsertionPointToStart(cmpBlock);

      Value gate;
      if (regOp.hasGate(i)) {
        gate = regOp.getGateAt(i);
      } else {
        gate = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                 rewriter.getBoolAttr(true));
      }

      auto drvArgs = std::array<Value, 3>(
          {transformed.values()[i], transformed.delays()[i], gate});

      RegMode mode = regOp.getRegModeAt(i);

      // Create comparison constants for all modes other than both.
      Value rhs;
      if (mode == RegMode::low || mode == RegMode::fall) {
        rhs = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                rewriter.getBoolAttr(false));
      } else if (mode == RegMode::high || mode == RegMode::rise) {
        rhs = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                rewriter.getBoolAttr(true));
      }

      // Create comparison for non-both modes.
      Value comp;
      if (rhs)
        comp =
            rewriter.create<LLVM::ICmpOp>(op->getLoc(), LLVM::ICmpPredicate::eq,
                                          transformed.triggers()[i], rhs);

      // Create comparison for modes needing more than one state of the trigger.
      Value brCond;
      if (mode == RegMode::rise || mode == RegMode::fall ||
          mode == RegMode::both) {

        auto cmpPrev = rewriter.create<LLVM::ICmpOp>(
            op->getLoc(), LLVM::ICmpPredicate::ne, transformed.triggers()[i],
            prevTriggers[j--]);
        if (mode == RegMode::both)
          brCond = cmpPrev;
        else
          brCond =
              rewriter.create<LLVM::AndOp>(op->getLoc(), i1Ty, comp, cmpPrev);
      } else {
        brCond = comp;
      }

      Block *nextBlock;
      nextBlock = cmpBlock->getNextNode();
      // Don't go to next block for last comparison's false branch (skip the
      // drive block).
      if (i == e)
        nextBlock = continueBlock;

      rewriter.create<LLVM::CondBrOp>(op->getLoc(), brCond, drvBlock, drvArgs,
                                      nextBlock, ValueRange());
    }

    rewriter.setInsertionPointToEnd(block);
    rewriter.create<LLVM::BrOp>(op->getLoc(), ArrayRef<Value>(),
                                block->getNextNode());

    rewriter.eraseOp(op);

    ++regCounter;

    return success();
  }
}; // namespace
} // namespace

//===----------------------------------------------------------------------===//
// Bitwise conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert an `llhd.not` operation. The result is an `llvm.xor` operation,
/// xor-ing the operand with all ones.
struct NotOpConversion : public ConvertToLLVMPattern {
  explicit NotOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::NotOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the adapted operands.
    NotOpAdaptor transformed(operands);
    // Get the `llhd.not` operation.
    auto notOp = cast<NotOp>(op);
    // Get integer width.
    unsigned width = getStdOrLLVMIntegerWidth(notOp.getType());
    // Get the used llvm types.
    auto iTy = LLVM::LLVMType::getIntNTy(&typeConverter.getContext(), width);

    // Get the all-ones mask operand
    APInt mask(width, 0);
    mask.setAllBits();
    auto rhs = rewriter.getIntegerAttr(rewriter.getIntegerType(width), mask);
    auto rhsConst = rewriter.create<LLVM::ConstantOp>(op->getLoc(), iTy, rhs);

    rewriter.replaceOpWithNewOp<LLVM::XOrOp>(
        op, typeConverter.convertType(notOp.getType()), transformed.value(),
        rhsConst);

    return success();
  }
};
} // namespace

namespace {
/// Convert an `llhd.shr` operation to LLVM dialect. All the operands are
/// extended to the width obtained by combining the hidden and base values. This
/// combined value is then shifted (exposing the hidden value) and truncated to
/// the base length
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
      // Get width of the base and hidden values combined.
      auto baseWidth = getStdOrLLVMIntegerWidth(shrOp.getType());
      auto hdnWidth = getStdOrLLVMIntegerWidth(shrOp.hidden().getType());
      auto full = baseWidth + hdnWidth;

      auto tmpTy = LLVM::LLVMType::getIntNTy(&typeConverter.getContext(), full);

      // Extend all operands the combined width.
      auto baseZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                    transformed.base());
      auto hdnZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                   transformed.hidden());
      auto amntZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                    transformed.amount());

      // Shift the hidden operand such that it can be prepended to the full
      // value.
      auto hdnShAmnt = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), tmpTy,
          rewriter.getIntegerAttr(rewriter.getIntegerType(full), baseWidth));
      auto hdnSh =
          rewriter.create<LLVM::ShlOp>(op->getLoc(), tmpTy, hdnZext, hdnShAmnt);

      // Combine the base and hidden values.
      auto combined =
          rewriter.create<LLVM::OrOp>(op->getLoc(), tmpTy, hdnSh, baseZext);

      // Perform the right shift.
      auto shifted = rewriter.create<LLVM::LShrOp>(op->getLoc(), tmpTy,
                                                   combined, amntZext);

      // Truncate to final width.
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(
          op, transformed.base().getType(), shifted);

      return success();
    } else if (auto resTy = shrOp.result().getType().dyn_cast<SigType>()) {
      auto i8PtrTy = getVoidPtrType();
      auto i64Ty = LLVM::LLVMType::getInt64Ty(&typeConverter.getContext());

      // Get the signal pointer and offset.
      auto sigDetail =
          getSignalDetail(rewriter, &getDialect(), op->getLoc(),
                          transformed.base(), /*extractIndices=*/true);

      auto zextAmnt = rewriter.create<LLVM::ZExtOp>(op->getLoc(), i64Ty,
                                                    transformed.amount());

      // Adjust slice start point from signal's offset.
      auto adjustedAmnt =
          rewriter.create<LLVM::AddOp>(op->getLoc(), sigDetail[1], zextAmnt);

      // Shift pointer to the new start byte.
      auto ptrToInt =
          rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, sigDetail[0]);
      auto const8 = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i64Ty, rewriter.getI64IntegerAttr(8));
      auto ptrOffset =
          rewriter.create<LLVM::UDivOp>(op->getLoc(), adjustedAmnt, const8);
      auto shiftedPtr =
          rewriter.create<LLVM::AddOp>(op->getLoc(), ptrToInt, ptrOffset);
      auto newPtr =
          rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), i8PtrTy, shiftedPtr);

      // Compute the offset into the first byte.
      auto bitOffset =
          rewriter.create<LLVM::URemOp>(op->getLoc(), adjustedAmnt, const8);

      // Create a subsignal with the new pointer and offset.
      rewriter.replaceOp(op, createSubSig(&getDialect(), rewriter, op->getLoc(),
                                          sigDetail, newPtr, bitOffset));

      return success();
    }

    return failure();
  }
};
} // namespace

namespace {
/// Convert an `llhd.shr` operation to LLVM dialect. All the operands are
/// extended to the width obtained by combining the hidden and base values. This
/// combined value is then shifted right by `hidden_width - amount` (exposing
/// the hidden value) and truncated to the base length
struct ShlOpConversion : public ConvertToLLVMPattern {
  explicit ShlOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ShlOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    ShlOpAdaptor transformed(operands);
    auto shlOp = cast<ShlOp>(op);
    assert(!shlOp.getType().isa<llhd::SigType>() && "sig not yet supported");

    // Get the width of the base and hidden operands combined.
    auto baseWidth = getStdOrLLVMIntegerWidth(shlOp.getType());
    auto hdnWidth = getStdOrLLVMIntegerWidth(shlOp.hidden().getType());
    auto full = baseWidth + hdnWidth;

    auto tmpTy = LLVM::LLVMType::getIntNTy(&typeConverter.getContext(), full);

    // Extend all operands to the combined width.
    auto baseZext =
        rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy, transformed.base());
    auto hdnZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                 transformed.hidden());
    auto amntZext = rewriter.create<LLVM::ZExtOp>(op->getLoc(), tmpTy,
                                                  transformed.amount());

    // Shift the base operand such that it can be prepended to the full value.
    auto hdnWidthConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), tmpTy,
        rewriter.getIntegerAttr(rewriter.getIntegerType(full), hdnWidth));
    auto baseSh = rewriter.create<LLVM::ShlOp>(op->getLoc(), tmpTy, baseZext,
                                               hdnWidthConst);

    // Comvine the base and hidden operands.
    auto combined =
        rewriter.create<LLVM::OrOp>(op->getLoc(), tmpTy, baseSh, hdnZext);

    // Get the final right shift amount by subtracting the shift amount from the
    // hidden width .
    auto shrAmnt = rewriter.create<LLVM::SubOp>(op->getLoc(), tmpTy,
                                                hdnWidthConst, amntZext);

    // Perform the shift.
    auto shifted =
        rewriter.create<LLVM::LShrOp>(op->getLoc(), tmpTy, combined, shrAmnt);

    // Truncate to the final width.
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, transformed.base().getType(),
                                               shifted);

    return success();
  }
};
} // namespace

using AndOpConversion = OneToOneConvertToLLVMPattern<llhd::AndOp, LLVM::AndOp>;
using OrOpConversion = OneToOneConvertToLLVMPattern<llhd::OrOp, LLVM::OrOp>;
using XorOpConversion = OneToOneConvertToLLVMPattern<llhd::XorOp, LLVM::XOrOp>;

//===----------------------------------------------------------------------===//
// Value manipulation conversions
//===----------------------------------------------------------------------===//

namespace {
/// Lower an LLHD constant operation to LLVM dialect. Time constants are lowered
/// to an array of 3 integers, containing the 3 time values. The other const
/// types are lowered to an equivalent `llvm.mlir.constant` operation.
struct ConstOpConversion : public ConvertToLLVMPattern {
  explicit ConstOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ConstOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operand,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the ConstOp.
    auto constOp = cast<ConstOp>(op);
    // Get the constant's attribute.
    auto attr = constOp.value();
    // Handle the time const special case: create a new array containing the
    // three time values.
    if (auto timeAttr = attr.dyn_cast<TimeAttr>()) {
      auto timeTy = typeConverter.convertType(constOp.getResult().getType());
      auto denseAttr = DenseElementsAttr::get(
          VectorType::get(3, rewriter.getI32Type()),
          {timeAttr.getTime(), timeAttr.getDelta(), timeAttr.getEps()});
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, timeTy, denseAttr);
      return success();
    }
    // Get the converted llvm type.
    auto intType = typeConverter.convertType(attr.getType());
    // Replace the operation with an llvm constant op.
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, intType,
                                                  constOp.valueAttr());

    return success();
  }
};
} // namespace

namespace {
/// Convert an llhd.exts operation. For integers, the value is shifted to the
/// start index and then truncated to the final length. For signals, a new
/// subsignal is created, pointing to the defined slice.
struct ExtractSliceOpConversion : public ConvertToLLVMPattern {
  explicit ExtractSliceOpConversion(MLIRContext *ctx,
                                    LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ExtractSliceOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto extsOp = cast<ExtractSliceOp>(op);

    ExtractSliceOpAdaptor transformed(operands);

    auto indexTy = typeConverter.convertType(extsOp.startAttr().getType());
    auto i8PtrTy = getVoidPtrType();
    auto i64Ty = LLVM::LLVMType::getInt64Ty(&typeConverter.getContext());

    // Get the attributes as constants.
    auto startConst = rewriter.create<LLVM::ConstantOp>(op->getLoc(), indexTy,
                                                        extsOp.startAttr());

    if (auto retTy = extsOp.result().getType().dyn_cast<IntegerType>()) {
      auto resTy = typeConverter.convertType(extsOp.result().getType());
      // Adjust the index const for shifting.
      Value adjusted;
      if (getStdOrLLVMIntegerWidth(extsOp.target().getType()) < 64) {
        adjusted = rewriter.create<LLVM::TruncOp>(
            op->getLoc(), transformed.target().getType(), startConst);
      } else {
        adjusted = rewriter.create<LLVM::ZExtOp>(
            op->getLoc(), transformed.target().getType(), startConst);
      }

      // Shift right by the index.
      auto shr = rewriter.create<LLVM::LShrOp>(op->getLoc(),
                                               transformed.target().getType(),
                                               transformed.target(), adjusted);
      // Truncate to the slice length.
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, resTy, shr);

      return success();
    } else if (auto resTy = extsOp.result().getType().dyn_cast<SigType>()) {
      auto sigDetail =
          getSignalDetail(rewriter, &getDialect(), op->getLoc(),
                          transformed.target(), /*extractIndices=*/true);

      // Adjust the slice starting point by the signal's offset.
      auto adjustedStart =
          rewriter.create<LLVM::AddOp>(op->getLoc(), sigDetail[1], startConst);

      // Shift the pointer to the new start byte.
      auto ptrToInt =
          rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, sigDetail[0]);
      auto const8 = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), indexTy, rewriter.getI64IntegerAttr(8));
      auto ptrOffset =
          rewriter.create<LLVM::UDivOp>(op->getLoc(), adjustedStart, const8);
      auto shiftedPtr =
          rewriter.create<LLVM::AddOp>(op->getLoc(), ptrToInt, ptrOffset);
      auto newPtr =
          rewriter.create<LLVM::IntToPtrOp>(op->getLoc(), i8PtrTy, shiftedPtr);

      // Compute the new offset into the first byte.
      auto bitOffset =
          rewriter.create<LLVM::URemOp>(op->getLoc(), adjustedStart, const8);

      // Create a new subsignal with the new pointer and offset.
      rewriter.replaceOp(op, createSubSig(&getDialect(), rewriter, op->getLoc(),
                                          sigDetail, newPtr, bitOffset));

      return success();
    }
    return failure();
  }
};
} // namespace

namespace {
/// Lower an `llhd.inss` operation to the LLVM dialect.
struct InsertSliceOpConversion : public ConvertToLLVMPattern {
  explicit InsertSliceOpConversion(MLIRContext *ctx,
                                   LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::InsertSliceOp ::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto inssOp = cast<InsertSliceOp>(op);

    InsertSliceOpAdaptor transformed(operands);

    auto indexTy = typeConverter.convertType(inssOp.startAttr().getType());

    if (inssOp.result().getType().isa<IntegerType>()) {
      auto startConst = rewriter.create<LLVM::ConstantOp>(op->getLoc(), indexTy,
                                                          inssOp.startAttr());
      Value adjusted;
      if (getStdOrLLVMIntegerWidth(inssOp.result().getType()) < 64) {
        adjusted = rewriter.create<LLVM::TruncOp>(
            op->getLoc(), transformed.target().getType(), startConst);
      } else {
        adjusted = rewriter.create<LLVM::ZExtOp>(
            op->getLoc(), transformed.target().getType(), startConst);
      }
      // Generate a mask to set the affected bits.
      auto width = getStdOrLLVMIntegerWidth(inssOp.target().getType());
      auto sliceWidth = getStdOrLLVMIntegerWidth(inssOp.slice().getType());
      unsigned start = inssOp.startAttr().getInt();
      unsigned end = start + sliceWidth;
      APInt mask(width, 0);
      mask.setBits(start, end);
      auto maskConst = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), transformed.target().getType(),
          rewriter.getIntegerAttr(
              IntegerType::get(width, rewriter.getContext()), mask));

      // Generate a mask for the slice, to avoid resetting bits outside of the
      // slice.
      mask.flipAllBits();
      auto sliceMaskConst = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), transformed.target().getType(),
          rewriter.getIntegerAttr(
              IntegerType::get(width, rewriter.getContext()), mask));

      // Adjust the slice to the start index.
      auto sliceZext = rewriter.create<LLVM::ZExtOp>(
          op->getLoc(), transformed.target().getType(), transformed.slice());
      auto sliceShift = rewriter.create<LLVM::ShlOp>(
          op->getLoc(), transformed.target().getType(), sliceZext, adjusted);
      auto sliceMasked = rewriter.create<LLVM::OrOp>(
          op->getLoc(), transformed.target().getType(), sliceShift,
          sliceMaskConst);

      // Insert the slice.
      auto applyMask = rewriter.create<LLVM::OrOp>(
          op->getLoc(), transformed.target(), maskConst);
      rewriter.replaceOpWithNewOp<LLVM::AndOp>(op, applyMask, sliceMasked);

      return success();
    }

    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Memory operations
//===----------------------------------------------------------------------===//

namespace {
/// Lower a `llhd.var` operation to the LLVM dialect. This results in an alloca,
/// followed by storing the initial value.
struct VarOpConversion : ConvertToLLVMPattern {
  explicit VarOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(VarOp::getOperationName(), ctx, typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    VarOpAdaptor transformed(operands);

    auto i32Ty = LLVM::LLVMType::getInt32Ty(&typeConverter.getContext());

    auto oneC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
    auto alloca = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(),
        transformed.init().getType().cast<LLVM::LLVMType>().getPointerTo(),
        oneC, 4);
    rewriter.create<LLVM::StoreOp>(op->getLoc(), transformed.init(), alloca);
    rewriter.replaceOp(op, alloca.getResult());
    return success();
  }
};
} // namespace

namespace {
/// Convert an `llhd.store` operation to LLVM. This lowers the store
/// one-to-one as an LLVM store, but with the operands flipped.
struct StoreOpConversion : ConvertToLLVMPattern {
  explicit StoreOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::StoreOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llhd::StoreOpAdaptor transformed(operands);

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, transformed.value(),
                                               transformed.pointer());

    return success();
  }
};
} // namespace

using LoadOpConversion =
    OneToOneConvertToLLVMPattern<llhd::LoadOp, LLVM::LoadOp>;

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

namespace {
struct LLHDToLLVMLoweringPass
    : public ConvertLLHDToLLVMBase<LLHDToLLVMLoweringPass> {
  void runOnOperation() override;
};
} // namespace

void llhd::populateLLHDToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();

  // Value manipulation conversion patterns.
  patterns.insert<ConstOpConversion, ExtractSliceOpConversion,
                  InsertSliceOpConversion>(ctx, converter);

  // Bitwise conversion patterns.
  patterns.insert<NotOpConversion, ShrOpConversion, ShlOpConversion>(ctx,
                                                                     converter);
  patterns.insert<AndOpConversion, OrOpConversion, XorOpConversion>(converter);

  // Unit conversion patterns.
  patterns.insert<EntityOpConversion, TerminatorOpConversion, ProcOpConversion,
                  WaitOpConversion, HaltOpConversion>(ctx, converter);

  // Signal conversion patterns.
  patterns.insert<SigOpConversion, PrbOpConversion, DrvOpConversion,
                  RegOpConversion>(ctx, converter);

  // Memory conversion patterns.
  patterns.insert<VarOpConversion, StoreOpConversion>(ctx, converter);
  patterns.insert<LoadOpConversion>(converter);
}

void LLHDToLLVMLoweringPass::runOnOperation() {
  OwningRewritePatternList patterns;
  auto converter = mlir::LLVMTypeConverter(&getContext());
  converter.addConversion(
      [&](SigType sig) { return convertSigType(sig, converter); });
  converter.addConversion(
      [&](TimeType time) { return convertTimeType(time, converter); });
  converter.addConversion(
      [&](PtrType ptr) { return convertPtrType(ptr, converter); });

  // Apply a partial conversion first, lowering only the instances, to generate
  // the init function.
  patterns.insert<InstOpConversion>(&getContext(), converter);

  LLVMConversionTarget target(getContext());
  target.addIllegalOp<InstOp>();
  target.addLegalOp<LLVM::DialectCastOp>();

  // Apply the partial conversion.
  if (failed(applyPartialConversion(getOperation(), target, patterns)))
    signalPassFailure();

  // Setup the full conversion.
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateLLHDToLLVMConversionPatterns(converter, patterns);

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  target.addIllegalOp<LLVM::DialectCastOp>();

  // Apply the full conversion.
  if (failed(applyFullConversion(getOperation(), target, patterns)))
    signalPassFailure();
}

/// Create an LLHD to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>>
circt::llhd::createConvertLLHDToLLVMPass() {
  return std::make_unique<LLHDToLLVMLoweringPass>();
}

/// Register the LLHD to LLVM convesion pass.
namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Conversion/LLHDToLLVM/Passes.h.inc"
} // namespace

void llhd::initLLHDToLLVMPass() { registerPasses(); }
