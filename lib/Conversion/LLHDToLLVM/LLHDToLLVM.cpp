//===- LLHDToLLVM.cpp - LLHD to LLVM Conversion Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main LLHD to LLVM Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LLHDToLLVM.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/CombToLLVM.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTLLHDTOLLVM
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::llhd;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Get an existing global string.
static Value getGlobalString(Location loc, OpBuilder &builder,
                             const TypeConverter *typeConverter,
                             LLVM::GlobalOp &str) {
  auto voidPtrTy = LLVM::LLVMPointerType::get(builder.getContext());
  auto i32Ty = IntegerType::get(builder.getContext(), 32);

  auto addr = builder.create<LLVM::AddressOfOp>(loc, voidPtrTy, str.getName());
  auto idx = builder.create<LLVM::ConstantOp>(loc, i32Ty,
                                              builder.getI32IntegerAttr(0));
  std::array<Value, 2> idxs({idx, idx});
  return builder.create<LLVM::GEPOp>(loc, voidPtrTy, str.getType(), addr, idxs);
}

/// Looks up a symbol and inserts a new functino at the beginning of the
/// module's region in case the function does not exists. If
/// insertBodyAndTerminator is set, also adds the entry block and return
/// terminator.
static LLVM::LLVMFuncOp
getOrInsertFunction(ModuleOp &module, ConversionPatternRewriter &rewriter,
                    Location loc, std::string name, Type signature,
                    bool insertBodyAndTerminator = false) {
  auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    func = moduleBuilder.create<LLVM::LLVMFuncOp>(loc, name, signature);
    if (insertBodyAndTerminator) {
      func.addEntryBlock(moduleBuilder);
      OpBuilder b(func.getBody());
      b.create<LLVM::ReturnOp>(loc, ValueRange());
    }
  }
  return func;
}

/// Return the LLVM type used to represent a signal. It corresponds to a struct
/// with the format: {valuePtr, bitOffset, instanceIndex, globalIndex}.
static Type getLLVMSigType(LLVM::LLVMDialect *dialect) {
  auto voidPtrTy = LLVM::LLVMPointerType::get(dialect->getContext());
  auto i64Ty = IntegerType::get(dialect->getContext(), 64);
  return LLVM::LLVMStructType::getLiteral(dialect->getContext(),
                                          {voidPtrTy, i64Ty, i64Ty, i64Ty});
}

/// Extract the details from the given signal struct. The details are returned
/// in the original struct order.
static std::vector<Value> getSignalDetail(ConversionPatternRewriter &rewriter,
                                          LLVM::LLVMDialect *dialect,
                                          Location loc, Value signal,
                                          bool extractIndices = false) {

  auto voidPtrTy = LLVM::LLVMPointerType::get(dialect->getContext());
  auto i64Ty = IntegerType::get(dialect->getContext(), 64);
  auto sigTy = getLLVMSigType(dialect);

  std::vector<Value> result;

  // Extract the value and offset elements.
  auto sigPtrPtr = rewriter.create<LLVM::GEPOp>(loc, voidPtrTy, sigTy, signal,
                                                ArrayRef<LLVM::GEPArg>({0, 0}));
  result.push_back(rewriter.create<LLVM::LoadOp>(loc, voidPtrTy, sigPtrPtr));

  auto offsetPtr = rewriter.create<LLVM::GEPOp>(loc, voidPtrTy, sigTy, signal,
                                                ArrayRef<LLVM::GEPArg>({0, 1}));
  result.push_back(rewriter.create<LLVM::LoadOp>(loc, i64Ty, offsetPtr));

  // Extract the instance and global indices.
  if (extractIndices) {
    auto instIndexPtr = rewriter.create<LLVM::GEPOp>(
        loc, voidPtrTy, sigTy, signal, ArrayRef<LLVM::GEPArg>({0, 2}));
    result.push_back(rewriter.create<LLVM::LoadOp>(loc, i64Ty, instIndexPtr));

    auto globalIndexPtr = rewriter.create<LLVM::GEPOp>(
        loc, voidPtrTy, sigTy, signal, ArrayRef<LLVM::GEPArg>({0, 3}));
    result.push_back(rewriter.create<LLVM::LoadOp>(loc, i64Ty, globalIndexPtr));
  }

  return result;
}

/// Create a subsignal struct.
static Value createSubSig(LLVM::LLVMDialect *dialect,
                          ConversionPatternRewriter &rewriter, Location loc,
                          std::vector<Value> originDetail, Value newPtr,
                          Value newOffset) {
  auto i32Ty = IntegerType::get(dialect->getContext(), 32);
  auto sigTy = getLLVMSigType(dialect);

  // Create signal struct.
  auto sigUndef = rewriter.create<LLVM::UndefOp>(loc, sigTy);
  auto storeSubPtr =
      rewriter.create<LLVM::InsertValueOp>(loc, sigUndef, newPtr, 0);
  auto storeSubOffset =
      rewriter.create<LLVM::InsertValueOp>(loc, storeSubPtr, newOffset, 1);
  auto storeSubInstIndex = rewriter.create<LLVM::InsertValueOp>(
      loc, storeSubOffset, originDetail[2], 2);
  auto storeSubGlobalIndex = rewriter.create<LLVM::InsertValueOp>(
      loc, storeSubInstIndex, originDetail[3], 3);

  // Allocate and store the subsignal.
  auto oneC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                rewriter.getI32IntegerAttr(1));
  auto allocaSubSig = rewriter.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(dialect->getContext()), sigTy, oneC, 4);
  rewriter.create<LLVM::StoreOp>(loc, storeSubGlobalIndex, allocaSubSig);

  return allocaSubSig;
}

/// Returns true if the given value is passed as an argument to the destination
/// block of the given WaitOp.
static bool isWaitDestArg(WaitOp op, Value val) {
  for (auto arg : op.getDestOps()) {
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
static Type getProcPersistenceTy(LLVM::LLVMDialect *dialect,
                                 const TypeConverter *converter, ProcOp &proc) {
  SmallVector<Type, 3> types = SmallVector<Type, 3>();
  proc.walk([&](Operation *op) -> void {
    if (op->isUsedOutsideOfBlock(op->getBlock()) || isWaitDestArg(op)) {
      auto ty = op->getResult(0).getType();
      auto convertedTy = converter->convertType(ty);
      types.push_back(convertedTy);
    }
  });

  // Also persist block arguments escaping their defining block.
  for (auto &block : proc.getBlocks()) {
    // Skip entry block (contains the function signature in its args).
    if (block.isEntryBlock())
      continue;

    for (auto arg : block.getArguments()) {
      if (arg.isUsedOutsideOfBlock(&block)) {
        types.push_back(converter->convertType(arg.getType()));
      }
    }
  }

  return LLVM::LLVMStructType::getLiteral(dialect->getContext(), types);
}

/// Insert a comparison block that either jumps to the trueDest block, if the
/// resume index mathces the current index, or to falseDest otherwise. If no
/// falseDest is provided, the next block is taken insead.
static void insertComparisonBlock(ConversionPatternRewriter &rewriter,
                                  LLVM::LLVMDialect *dialect, Location loc,
                                  Region *body, Value resumeIdx, int currIdx,
                                  Block *trueDest, ValueRange trueDestArgs,
                                  Block *falseDest = nullptr) {
  auto i32Ty = IntegerType::get(dialect->getContext(), 32);
  auto secondBlock = ++body->begin();
  auto newBlock = rewriter.createBlock(body, secondBlock);
  auto cmpIdx = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(currIdx));
  auto cmpRes = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                              resumeIdx, cmpIdx);

  // Default to jumping to the next block for the false case, if no explicit
  // block is provided.
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
                                 Type stateTy, int index, Value state) {
  return rewriter.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(dialect->getContext()), stateTy, state,
      ArrayRef<LLVM::GEPArg>({0, 3, index}));
}

/// Persist a `Value` by storing it into the process persistence table, and
/// substituting the uses that escape the block the operation is defined in with
/// a load from the persistence table.
static void persistValue(LLVM::LLVMDialect *dialect, Location loc,
                         const TypeConverter *converter,
                         ConversionPatternRewriter &rewriter, Type stateTy,
                         int &i, Value state, Value persist) {
  auto elemTy = cast<LLVM::LLVMStructType>(
                    cast<LLVM::LLVMStructType>(stateTy).getBody()[3])
                    .getBody()[i];

  if (auto arg = dyn_cast<BlockArgument>(persist)) {
    rewriter.setInsertionPointToStart(arg.getParentBlock());
  } else {
    rewriter.setInsertionPointAfter(persist.getDefiningOp());
  }

  Value convPersist = converter->materializeTargetConversion(
      rewriter, loc, converter->convertType(persist.getType()), {persist});

  auto gep0 = gepPersistenceState(dialect, loc, rewriter, stateTy, i, state);

  Value toStore;
  if (auto ptr = dyn_cast<PtrType>(persist.getType())) {
    // Unwrap the pointer and store it's value.
    auto elemTy = converter->convertType(ptr.getUnderlyingType());
    toStore = rewriter.create<LLVM::LoadOp>(loc, elemTy, convPersist);
  } else if (isa<SigType>(persist.getType())) {
    // Unwrap and store the signal struct.
    toStore = rewriter.create<LLVM::LoadOp>(loc, getLLVMSigType(dialect),
                                            convPersist);
  } else {
    // Store the value directly.
    toStore = convPersist;
  }

  rewriter.create<LLVM::StoreOp>(loc, toStore, gep0);

  // Load the value from the persistence table and substitute the original
  // use with it, whenever it is in a different block.
  for (auto &use : llvm::make_early_inc_range(persist.getUses())) {
    auto user = use.getOwner();
    if (isa<PtrType>(persist.getType()) && user != toStore.getDefiningOp() &&
        user != convPersist.getDefiningOp() &&
        persist.getParentBlock() == user->getBlock()) {
      // Redirect uses of the pointer in the same block to the pointer in the
      // persistence state. This ensures that stores and loads all operate on
      // the same value.
      use.set(gep0);
    } else if (persist.getParentBlock() != user->getBlock() ||
               (isa<WaitOp>(user) &&
                isWaitDestArg(cast<WaitOp>(user), persist))) {
      // The destination args of a wait op have to be loaded in the entry block
      // of the function, before jumping to the resume destination, so they can
      // be passed as block arguments by the comparison block.
      if (isa<WaitOp>(user) && isWaitDestArg(cast<WaitOp>(user), persist))
        rewriter.setInsertionPoint(
            user->getParentRegion()->front().getTerminator());
      else
        rewriter.setInsertionPointToStart(user->getBlock());

      auto gep1 =
          gepPersistenceState(dialect, loc, rewriter, stateTy, i, state);
      // Use the pointer in the state struct directly for pointer and signal
      // types.
      if (isa<PtrType, SigType>(persist.getType())) {
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
static void insertPersistence(const TypeConverter *converter,
                              ConversionPatternRewriter &rewriter,
                              LLVM::LLVMDialect *dialect, Location loc,
                              ProcOp &proc, Type &stateTy,
                              LLVM::LLVMFuncOp &converted,
                              Operation *splitEntryBefore) {
  auto i32Ty = IntegerType::get(dialect->getContext(), 32);

  auto &firstBB = converted.getBody().front();

  // Split entry block such that all the operations contained in it in the
  // original process appear after the comparison blocks.
  auto splitFirst =
      rewriter.splitBlock(&firstBB, splitEntryBefore->getIterator());

  // Insert dummy branch terminator at the new end of the function's entry
  // block.
  rewriter.setInsertionPointToEnd(&firstBB);
  rewriter.create<LLVM::BrOp>(loc, ValueRange(), splitFirst);

  // Load the resume index from the process state argument.
  rewriter.setInsertionPoint(firstBB.getTerminator());
  auto gep = rewriter.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(dialect->getContext()), i32Ty,
      converted.getArgument(1), ArrayRef<LLVM::GEPArg>({1}));

  auto larg = rewriter.create<LLVM::LoadOp>(loc, i32Ty, gep);

  auto body = &converted.getBody();

  // Insert an abort block as the last block.
  auto abortBlock = rewriter.createBlock(body, body->end());
  rewriter.create<LLVM::ReturnOp>(loc, ValueRange());

  // Redirect the entry block to a first comparison block. If on a first
  // execution, jump to the new (splitted) entry block, else the process is in
  // an illegal state and jump to the abort block.
  insertComparisonBlock(rewriter, dialect, loc, body, larg, 0, splitFirst,
                        ValueRange(), abortBlock);

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
                            wait.getDest(), wait.getDestOps());

      // Insert the resume index update at the wait operation location.
      rewriter.setInsertionPoint(op);
      auto procState = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(1);
      auto resumeIdxC = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(waitInd));
      auto resumeIdxPtr = rewriter.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(dialect->getContext()), i32Ty,
          procState, ArrayRef<LLVM::GEPArg>({1}));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), resumeIdxC, resumeIdxPtr);
    }
  });

  // Also persist argument blocks escaping their defining block.
  for (auto &block : converted.getBlocks()) {
    // Skip entry block as it contains the function signature.
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
/// that require more than one state of the trigger to infer it (i.e. `both`,
/// `rise` and `fall`).
static LLVM::LLVMStructType getRegStateTy(LLVM::LLVMDialect *dialect,
                                          Operation *entity) {
  SmallVector<Type, 4> types;
  entity->walk([&](RegOp op) {
    size_t count = 0;
    for (size_t i = 0; i < op.getModes().size(); ++i) {
      auto mode = op.getRegModeAt(i);
      if (mode == RegMode::fall || mode == RegMode::rise ||
          mode == RegMode::both)
        ++count;
    }
    if (count > 0)
      types.push_back(LLVM::LLVMArrayType::get(
          IntegerType::get(dialect->getContext(), 1), count));
  });
  return LLVM::LLVMStructType::getLiteral(dialect->getContext(), types);
}

/// Create a zext operation by one bit on the given value. This is useful when
/// passing unsigned indexes to a GEP instruction, which treats indexes as
/// signed values, to avoid unexpected "sign overflows".
static Value zextByOne(Location loc, ConversionPatternRewriter &rewriter,
                       Value value) {
  auto valueTy = value.getType();
  auto zextTy = IntegerType::get(valueTy.getContext(),
                                 valueTy.getIntOrFloatBitWidth() + 1);
  return rewriter.create<LLVM::ZExtOp>(loc, zextTy, value);
}

/// Adjust the bithwidth of value to be the same as targetTy's bitwidth.
static Value adjustBitWidth(Location loc, ConversionPatternRewriter &rewriter,
                            Type targetTy, Value value) {
  auto valueWidth = value.getType().getIntOrFloatBitWidth();
  auto targetWidth = targetTy.getIntOrFloatBitWidth();

  if (valueWidth < targetWidth)
    return rewriter.create<LLVM::ZExtOp>(loc, targetTy, value);

  if (valueWidth > targetWidth)
    return rewriter.create<LLVM::TruncOp>(loc, targetTy, value);

  return value;
}

static unsigned getIndexOfOperandResult(Operation *op, Value result) {
  for (unsigned j = 0, e = op->getNumResults(); j < e; ++j) {
    if (result == result.getDefiningOp()->getResult(j))
      return j;
  }
  llvm_unreachable(
      "no way to recurse to an operation that does not return any value");
}

/// Recursively clone the init origin of a sig operation into the init function,
/// up to the initial constant value(s). This is required to clone the
/// initialization of array and struct signals, where the init operand cannot
/// originate from a constant operation.
static Value recursiveCloneInit(OpBuilder &initBuilder, IRMapping &mapping,
                                Value init) {
  SmallVector<Value> clonedOperands;
  Operation *initOp = init.getDefiningOp();

  // If we end up at a value that we get via BlockArgument or as a result of a
  // llhd.prb op, return a nullptr to signal that something went wrong, because
  // these cases are not supported.
  if (!initOp || isa<llhd::PrbOp>(initOp))
    return nullptr;

  for (size_t i = 0, e = initOp->getNumOperands(); i < e; ++i) {
    Value operand = initOp->getOperand(i);

    // If we have some value that is used multiple times (e.g., broadcasted to
    // an array) then don't emit the ops to create this value several times,
    // but instead remember the cloned value and use it again.
    if (auto memorizedOperand = mapping.lookupOrNull(operand)) {
      clonedOperands.push_back(memorizedOperand);
      continue;
    }

    // Recursively follow operands.
    Value clonedOperand = recursiveCloneInit(initBuilder, mapping, operand);
    if (!clonedOperand)
      return nullptr;

    mapping.map(operand, clonedOperand);
    clonedOperands.push_back(clonedOperand);
  }

  Operation *clone = initOp->clone();
  clone->setOperands(clonedOperands);

  // If we have cloned an operation that returns several values, we have to
  // find the result value of the cloned operation we want to return.
  unsigned index = getIndexOfOperandResult(initOp, init);
  return initBuilder.insert(clone)->getResult(index);
}

/// Check if the given type is either of LLHD's ArrayType, StructType, or LLVM
/// array or struct type.
static bool isArrayOrStruct(Type type) {
  return isa<LLVM::LLVMArrayType, LLVM::LLVMStructType, hw::ArrayType,
             hw::StructType>(type);
}

/// Shift an integer signal pointer to obtain a view of the underlying value as
/// if it was shifted.
static std::pair<Value, Value>
shiftIntegerSigPointer(Location loc, LLVM::LLVMDialect *dialect,
                       ConversionPatternRewriter &rewriter, Value pointer,
                       Value index) {
  auto voidPtrTy = LLVM::LLVMPointerType::get(dialect->getContext());
  auto i64Ty = IntegerType::get(dialect->getContext(), 64);

  auto ptrToInt = rewriter.create<LLVM::PtrToIntOp>(loc, i64Ty, pointer);
  auto const8 = rewriter.create<LLVM::ConstantOp>(
      loc, index.getType(), rewriter.getI64IntegerAttr(8));
  auto ptrOffset = rewriter.create<LLVM::UDivOp>(loc, index, const8);
  auto shiftedPtr = rewriter.create<LLVM::AddOp>(loc, ptrToInt, ptrOffset);
  auto newPtr = rewriter.create<LLVM::IntToPtrOp>(loc, voidPtrTy, shiftedPtr);

  // Compute the new offset into the first byte.
  auto bitOffset = rewriter.create<LLVM::URemOp>(loc, index, const8);

  return std::make_pair(newPtr, bitOffset);
}

/// Shift the pointer of a structured-type (array or struct) signal, to change
/// its view as if the desired slice/element was extracted.
static Value shiftStructuredSigPointer(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Type elemTy, Value pointer,
                                       LLVM::GEPArg index) {
  // TODO: Remove unused args
  auto voidPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  return rewriter.create<LLVM::GEPOp>(loc, voidPtrTy, elemTy, pointer,
                                      ArrayRef<LLVM::GEPArg>({0, index}));
}

/// Shift the pointer of an array-typed signal, to change its view as if the
/// desired slice/element was extracted.
static Value shiftArraySigPointer(Location loc,
                                  ConversionPatternRewriter &rewriter,
                                  Type arrTy, Value pointer,
                                  LLVM::GEPArg index) {
  if (auto indexValue = dyn_cast<Value>(index))
    index = zextByOne(loc, rewriter, indexValue);
  return shiftStructuredSigPointer(loc, rewriter, arrTy, pointer, index);
}

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

static Type convertSigType(SigType type, LLVMTypeConverter &converter) {
  auto &context = converter.getContext();
  // auto i64Ty = IntegerType::get(&context, 64);
  auto voidPtrTy = LLVM::LLVMPointerType::get(&context);
  // LLVM::LLVMStructType::getLiteral(&context, {voidPtrTy, i64Ty, i64Ty,
  // i64Ty})
  return voidPtrTy;
}

static Type convertTimeType(TimeType type, LLVMTypeConverter &converter) {
  auto i64Ty = IntegerType::get(&converter.getContext(), 64);
  return LLVM::LLVMArrayType::get(i64Ty, 3);
}

static Type convertPtrType(PtrType type, LLVMTypeConverter &converter) {
  // converter.convertType(type.getUnderlyingType())
  return LLVM::LLVMPointerType::get(type.getContext());
}

//===----------------------------------------------------------------------===//
// Unit conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert an `llhd.entity` entity to LLVM dialect. The result is an
/// `llvm.func` which takes a pointer to the global simulation state, a pointer
/// to the entity's local state, and a pointer to the instance's signal table as
/// arguments.
struct EntityOpConversion : public ConvertToLLVMPattern {
  explicit EntityOpConversion(MLIRContext *ctx,
                              LLVMTypeConverter &typeConverter,
                              size_t &sigCounter, size_t &regCounter)
      : ConvertToLLVMPattern(llhd::EntityOp::getOperationName(), ctx,
                             typeConverter),
        sigCounter(sigCounter), regCounter(regCounter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get adapted operands.
    EntityOpAdaptor transformed(operands);
    // Get entity operation.
    auto entityOp = cast<EntityOp>(op);

    // Collect used llvm types.
    auto voidTy = getVoidType();
    auto voidPtrTy = getVoidPtrType();
    auto sigTy = getLLVMSigType(&getDialect());

    regCounter = 0;

    // Use an intermediate signature conversion to add the arguments for the
    // state and signal table pointer arguments.
    LLVMTypeConverter::SignatureConversion intermediate(
        entityOp.getNumArguments());
    // Add state and signal table arguments.
    intermediate.addInputs(
        std::array<Type, 3>({voidPtrTy, voidPtrTy, voidPtrTy}));
    for (size_t i = 0, e = entityOp.getNumArguments(); i < e; ++i)
      intermediate.addInputs(i, voidTy);
    rewriter.applySignatureConversion(entityOp.getBodyBlock(), intermediate,
                                      typeConverter);

    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&entityOp.getBlocks().front());
    LLVMTypeConverter::SignatureConversion final(
        intermediate.getConvertedTypes().size());
    final.addInputs(0, voidPtrTy);
    final.addInputs(1, voidPtrTy);
    final.addInputs(2, voidPtrTy);

    // The first n elements of the signal table represent the entity arguments,
    // while the remaining elements represent the entity's owned signals.
    sigCounter = entityOp.getNumArguments();
    for (size_t i = 0; i < sigCounter; ++i) {
      // Create gep operations from the signal table for each original argument.
      auto gep = bodyBuilder.create<LLVM::GEPOp>(op->getLoc(), voidPtrTy, sigTy,
                                                 entityOp.getArgument(2),
                                                 LLVM::GEPArg(i));
      // Remap i-th original argument to the gep'd signal pointer.
      final.remapInput(i + 3, gep.getResult());
    }

    rewriter.applySignatureConversion(entityOp.getBodyBlock(), final,
                                      typeConverter);

    // Get the converted entity signature.
    auto funcTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, voidPtrTy, voidPtrTy});

    // Create the a new llvm function to house the lowered entity.
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(
        op->getLoc(), entityOp.getName(), funcTy);

    // Add a return to the entity for later inclusion into the LLVM function.
    rewriter.setInsertionPointToEnd(&entityOp.getBlocks().front());
    rewriter.create<LLVM::ReturnOp>(op->getLoc(), ValueRange{});

    // Inline the entity region in the new llvm function.
    rewriter.inlineRegionBefore(entityOp.getBody(), llvmFunc.getBody(),
                                llvmFunc.end());

    // Erase the original operation.
    rewriter.eraseOp(op);

    return success();
  }

private:
  size_t &sigCounter;
  size_t &regCounter;
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
    auto voidPtrTy = getVoidPtrType();
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto stateTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(),
        {/*currentInstance=*/i32Ty, /*resumeIndex=*/i32Ty,
         /*senseFlags=*/voidPtrTy /*senseTableTy*/,
         /*persistence=*/
         getProcPersistenceTy(&getDialect(), typeConverter, procOp)});
    auto sigTy = getLLVMSigType(&getDialect());

    // Keep track of the original first operation of the process, to know where
    // to split the first block to insert comparison blocks.
    auto &firstOp = op->getRegion(0).front().front();

    // Have an intermediate signature conversion to add the arguments for the
    // state, process-specific state and signal table.
    LLVMTypeConverter::SignatureConversion intermediate(
        procOp.getNumArguments());
    // Add state, process state table and signal table arguments.
    std::array<Type, 3> procArgTys({voidPtrTy, voidPtrTy, voidPtrTy});
    intermediate.addInputs(procArgTys);
    for (size_t i = 0, e = procOp.getNumArguments(); i < e; ++i)
      intermediate.addInputs(i, voidTy);
    rewriter.applySignatureConversion(&procOp.getBlocks().front(), intermediate,
                                      typeConverter);

    // Get the final signature conversion.
    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&procOp.getBlocks().front());
    LLVMTypeConverter::SignatureConversion final(
        intermediate.getConvertedTypes().size());
    final.addInputs(0, voidPtrTy);
    final.addInputs(1, voidPtrTy);
    final.addInputs(2, voidPtrTy);

    for (size_t i = 0, e = procOp.getNumArguments(); i < e; ++i) {
      // Create gep operations from the signal table for each original argument.
      auto gep = bodyBuilder.create<LLVM::GEPOp>(op->getLoc(), voidPtrTy, sigTy,
                                                 procOp.getArgument(2),
                                                 LLVM::GEPArg(i));

      // Remap the i-th original argument to the gep'd value.
      final.remapInput(i + 3, gep.getResult());
    }

    // Get the converted process signature.
    auto funcTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, voidPtrTy, voidPtrTy});
    // Create a new llvm function to house the lowered process.
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(),
                                                      procOp.getName(), funcTy);
    llvmFunc->setAttr("llhd.argument_count",
                      rewriter.getI32IntegerAttr(procOp.getNumArguments()));

    // Inline the process region in the new llvm function.
    rewriter.inlineRegionBefore(procOp.getBody(), llvmFunc.getBody(),
                                llvmFunc.end());

    insertPersistence(typeConverter, rewriter, &getDialect(), op->getLoc(),
                      procOp, stateTy, llvmFunc, &firstOp);

    // Convert the block argument types after inserting the persistence, as this
    // would otherwise interfere with the persistence generation.
    if (failed(rewriter.convertRegionTypes(&llvmFunc.getBody(), *typeConverter,
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
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto voidPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto procState = llvmFunc.getArgument(1);

    // Get senses ptr from the process state argument.
    auto sensePtrGep =
        rewriter.create<LLVM::GEPOp>(op->getLoc(), voidPtrTy, voidPtrTy,
                                     procState, ArrayRef<LLVM::GEPArg>({2}));
    auto sensePtr =
        rewriter.create<LLVM::LoadOp>(op->getLoc(), voidPtrTy, sensePtrGep);

    // Zero out all the senses flags.
    unsigned numSenseEntries =
        llvmFunc->getAttrOfType<IntegerAttr>("llhd.argument_count")
            .getValue()
            .getZExtValue();
    auto zeroB = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                   rewriter.getBoolAttr(false));
    for (unsigned i = 0; i < numSenseEntries; ++i) {
      auto senseElemPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), voidPtrTy, i1Ty, sensePtr, ArrayRef<LLVM::GEPArg>({i}));
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
    auto voidPtrTy = getVoidPtrType();
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Get the llhdSuspend runtime function.
    auto llhdSuspendTy = LLVM::LLVMFunctionType::get(
        voidTy, {voidPtrTy, voidPtrTy, i64Ty, i64Ty, i64Ty});
    auto module = op->getParentOfType<ModuleOp>();
    auto llhdSuspendFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                               "llhdSuspend", llhdSuspendTy);

    auto statePtr = llvmFunc.getArgument(0);
    auto procState = llvmFunc.getArgument(1);

    // Get senses ptr.
    auto sensePtrGep =
        rewriter.create<LLVM::GEPOp>(op->getLoc(), voidPtrTy, voidPtrTy,
                                     procState, ArrayRef<LLVM::GEPArg>({2}));
    auto sensePtr =
        rewriter.create<LLVM::LoadOp>(op->getLoc(), voidPtrTy, sensePtrGep);

    // Reset sense table, if not all signals are observed.
    unsigned numSenseEntries =
        llvmFunc->getAttrOfType<IntegerAttr>("llhd.argument_count")
            .getValue()
            .getZExtValue();
    if (waitOp.getObs().size() < numSenseEntries) {
      auto zeroB = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getBoolAttr(false));
      for (size_t i = 0; i < numSenseEntries; ++i) {
        auto senseElemPtr =
            rewriter.create<LLVM::GEPOp>(op->getLoc(), voidPtrTy, i1Ty,
                                         sensePtr, ArrayRef<LLVM::GEPArg>({i}));
        rewriter.create<LLVM::StoreOp>(op->getLoc(), zeroB, senseElemPtr);
      }
    }

    // Set sense flags for observed signals.
    for (auto observed : transformed.getObs()) {
      auto instIndexPtr =
          rewriter.create<LLVM::GEPOp>(op->getLoc(), voidPtrTy, i64Ty, observed,
                                       ArrayRef<LLVM::GEPArg>({2}));
      auto instIndex =
          rewriter.create<LLVM::LoadOp>(op->getLoc(), i64Ty, instIndexPtr)
              .getResult();
      auto oneB = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                    rewriter.getBoolAttr(true));
      auto senseElementPtr =
          rewriter.create<LLVM::GEPOp>(op->getLoc(), voidPtrTy, i1Ty, sensePtr,
                                       ArrayRef<LLVM::GEPArg>({instIndex}));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), oneB, senseElementPtr);
    }

    // Update and store the new resume index in the process state.
    // Spawn scheduled event, if present.
    if (waitOp.getTime()) {
      auto realTime = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), transformed.getTime(), 0);
      auto delta = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), transformed.getTime(), 1);
      auto eps = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), transformed.getTime(), 2);

      std::array<Value, 5> args({statePtr, procState, realTime, delta, eps});
      rewriter.create<LLVM::CallOp>(op->getLoc(), std::nullopt,
                                    SymbolRefAttr::get(llhdSuspendFunc), args);
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange());
    return success();
  }
};
} // namespace

namespace {
/// Lower an llhd.inst operation to LLVM dialect. This generates malloc calls
/// and allocSignal calls (to store the pointer into the state) for each signal
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
    auto entity = op->getParentOfType<EntityOp>();

    auto voidTy = getVoidType();
    auto voidPtrTy = getVoidPtrType();
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Init function signature: (i8* %state) -> void.
    auto initFuncTy = LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy});
    auto initFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(), "llhd_init",
                            initFuncTy, /*insertBodyAndTerminator=*/true);

    // Get or insert the malloc function definition.
    // Malloc function signature: (i64 %size) -> i8* %pointer.
    auto mallocSigFuncTy = LLVM::LLVMFunctionType::get(voidPtrTy, {i64Ty});
    auto mallFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                        "malloc", mallocSigFuncTy);

    // Get or insert the allocSignal library call definition.
    // allocSignal function signature: (i8* %state, i8* %sig_name, i8*
    // %sig_owner, i32 %value) -> i32 %sig_index.
    auto allocSigFuncTy = LLVM::LLVMFunctionType::get(
        i32Ty, {voidPtrTy, i32Ty, voidPtrTy, voidPtrTy, i64Ty});
    auto sigFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                       "allocSignal", allocSigFuncTy);

    // Add information about the elements of an array signal to the state.
    // Signature: (i8* state, i32 signalIndex, i32 size, i32 numElements) ->
    // void
    auto addSigArrElemFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, i32Ty, i32Ty, i32Ty});
    auto addSigElemFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(),
                            "addSigArrayElements", addSigArrElemFuncTy);

    // Add information about one element of a struct signal to the state.
    // Signature: (i8* state, i32 signalIndex, i32 offset, i32 size) -> void
    auto addSigStructElemFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, i32Ty, i32Ty, i32Ty});
    auto addSigStructFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(),
                            "addSigStructElement", addSigStructElemFuncTy);

    // Get or insert allocProc library call definition.
    auto allocProcFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, voidPtrTy, voidPtrTy});
    auto allocProcFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                             "allocProc", allocProcFuncTy);

    // Get or insert allocEntity library call definition.
    auto allocEntityFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {voidPtrTy, voidPtrTy, voidPtrTy});
    auto allocEntityFunc = getOrInsertFunction(
        module, rewriter, op->getLoc(), "allocEntity", allocEntityFuncTy);

    Value initStatePtr = initFunc.getArgument(0);

    // Get a builder for the init function.
    OpBuilder initBuilder =
        OpBuilder::atBlockTerminator(&initFunc.getBody().getBlocks().front());

    // Use the instance name to retrieve the instance from the state.
    auto ownerName = entity.getName().str() + "." + instOp.getName().str();

    // Get or create owner name string
    Value owner;
    auto parentSym =
        module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName);
    if (!parentSym) {
      owner = LLVM::createGlobalString(
          op->getLoc(), initBuilder, "instance." + ownerName, ownerName + '\0',
          LLVM::Linkage::Internal);
      parentSym = module.lookupSymbol<LLVM::GlobalOp>("instance." + ownerName);
    } else {
      owner =
          getGlobalString(op->getLoc(), initBuilder, typeConverter, parentSym);
    }

    // Handle entity instantiation.
    if (auto child = module.lookupSymbol<EntityOp>(instOp.getCallee())) {
      auto regStateTy = getRegStateTy(&getDialect(), child.getOperation());

      // Get reg state size.
      auto regNull = initBuilder.create<LLVM::ZeroOp>(op->getLoc(), voidPtrTy);
      auto regGep =
          initBuilder.create<LLVM::GEPOp>(op->getLoc(), voidPtrTy, regStateTy,
                                          regNull, ArrayRef<LLVM::GEPArg>({1}));
      auto regSize =
          initBuilder.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, regGep);

      // Malloc reg state.
      auto regMall = initBuilder
                         .create<LLVM::CallOp>(op->getLoc(), voidPtrTy,
                                               SymbolRefAttr::get(mallFunc),
                                               ArrayRef<Value>({regSize}))
                         .getResult();
      auto zeroB = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getBoolAttr(false));

      // Zero-initialize reg state entries.
      for (size_t i = 0,
                  e = cast<LLVM::LLVMStructType>(regStateTy).getBody().size();
           i < e; ++i) {
        size_t f = cast<LLVM::LLVMArrayType>(
                       cast<LLVM::LLVMStructType>(regStateTy).getBody()[i])
                       .getNumElements();
        for (size_t j = 0; j < f; ++j) {
          auto regGep = initBuilder.create<LLVM::GEPOp>(
              op->getLoc(), voidPtrTy, regStateTy, regMall,
              ArrayRef<LLVM::GEPArg>({0, i, j}));
          initBuilder.create<LLVM::StoreOp>(op->getLoc(), zeroB, regGep);
        }
      }

      // Add reg state pointer to global state.
      initBuilder.create<LLVM::CallOp>(
          op->getLoc(), std::nullopt, SymbolRefAttr::get(allocEntityFunc),
          ArrayRef<Value>({initStatePtr, owner, regMall}));

      // Index of the signal in the entity's signal table.
      int initCounter = 0;
      // Walk over the entity and generate mallocs for each one of its signals.
      WalkResult sigWalkResult = child.walk([&](SigOp op) -> WalkResult {
        // if (auto sigOp = dyn_cast<SigOp>(op)) {
        auto underlyingTy = typeConverter->convertType(op.getInit().getType());
        // Get index constant of the signal in the entity's signal table.
        auto indexConst = initBuilder.create<LLVM::ConstantOp>(
            op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(initCounter));
        initCounter++;

        // Clone and insert the operation that defines the signal's init
        // operand (assmued to be a constant/array op)
        IRMapping mapping;
        Value initDef = recursiveCloneInit(initBuilder, mapping, op.getInit());

        if (!initDef)
          return WalkResult::interrupt();

        Value initDefCast = typeConverter->materializeTargetConversion(
            initBuilder, initDef.getLoc(),
            typeConverter->convertType(initDef.getType()), initDef);

        // Compute the required space to malloc.
        auto twoC = initBuilder.create<LLVM::ConstantOp>(
            op.getLoc(), i64Ty, rewriter.getI32IntegerAttr(2));
        auto nullPtr = initBuilder.create<LLVM::ZeroOp>(op.getLoc(), voidPtrTy);
        auto sizeGep = initBuilder.create<LLVM::GEPOp>(
            op.getLoc(), voidPtrTy, underlyingTy, nullPtr,
            ArrayRef<LLVM::GEPArg>({1}));
        auto size =
            initBuilder.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, sizeGep);
        // Malloc double the required space to make sure signal
        // shifts do not segfault.
        auto mallocSize =
            initBuilder.create<LLVM::MulOp>(op.getLoc(), i64Ty, size, twoC);
        std::array<Value, 1> margs({mallocSize});
        auto mall =
            initBuilder
                .create<LLVM::CallOp>(op.getLoc(), voidPtrTy,
                                      SymbolRefAttr::get(mallFunc), margs)
                .getResult();

        // Store the initial value.
        initBuilder.create<LLVM::StoreOp>(op.getLoc(), initDefCast, mall);

        // Get the amount of bytes required to represent an integer underlying
        // type. Use the whole size of the type if not an integer.
        Value passSize;
        if (auto intTy = dyn_cast<IntegerType>(underlyingTy)) {
          auto byteWidth = llvm::divideCeil(intTy.getWidth(), 8);
          passSize = initBuilder.create<LLVM::ConstantOp>(
              op.getLoc(), i64Ty, rewriter.getI64IntegerAttr(byteWidth));
        } else {
          passSize = size;
        }

        std::array<Value, 5> args(
            {initStatePtr, indexConst, owner, mall, passSize});
        auto sigIndex =
            initBuilder
                .create<LLVM::CallOp>(op.getLoc(), i32Ty,
                                      SymbolRefAttr::get(sigFunc), args)
                .getResult();

        // Add structured underlying type information.
        if (auto arrayTy = dyn_cast<LLVM::LLVMArrayType>(underlyingTy)) {
          auto numElements = initBuilder.create<LLVM::ConstantOp>(
              op.getLoc(), i32Ty,
              rewriter.getI32IntegerAttr(arrayTy.getNumElements()));

          // Get element size.
          auto null = initBuilder.create<LLVM::ZeroOp>(op.getLoc(), voidPtrTy);
          auto gepFirst = initBuilder.create<LLVM::GEPOp>(
              op.getLoc(), voidPtrTy, arrayTy, null,
              ArrayRef<LLVM::GEPArg>({0, 1}));
          auto toInt = initBuilder.create<LLVM::PtrToIntOp>(op.getLoc(), i32Ty,
                                                            gepFirst);

          // Add information to the state.
          initBuilder.create<LLVM::CallOp>(
              op.getLoc(), std::nullopt, SymbolRefAttr::get(addSigElemFunc),
              ArrayRef<Value>({initStatePtr, sigIndex, toInt, numElements}));
        } else if (auto structTy =
                       dyn_cast<LLVM::LLVMStructType>(underlyingTy)) {
          auto null = initBuilder.create<LLVM::ZeroOp>(op.getLoc(), voidPtrTy);
          for (size_t i = 0, e = structTy.getBody().size(); i < e; ++i) {
            // Get pointer offset.
            auto gepElem = initBuilder.create<LLVM::GEPOp>(
                op.getLoc(), voidPtrTy, structTy, null,
                ArrayRef<LLVM::GEPArg>({0, i}));
            auto elemToInt = initBuilder.create<LLVM::PtrToIntOp>(
                op.getLoc(), i32Ty, gepElem);

            // Get element size.
            auto elemNull =
                initBuilder.create<LLVM::ZeroOp>(op.getLoc(), voidPtrTy);
            auto gepElemSize = initBuilder.create<LLVM::GEPOp>(
                op.getLoc(), voidPtrTy, structTy.getBody()[i], elemNull,
                ArrayRef<LLVM::GEPArg>({1}));
            auto elemSizeToInt = initBuilder.create<LLVM::PtrToIntOp>(
                op.getLoc(), i32Ty, gepElemSize);

            // Add information to the state.
            initBuilder.create<LLVM::CallOp>(
                op.getLoc(), std::nullopt, SymbolRefAttr::get(addSigStructFunc),
                ArrayRef<Value>(
                    {initStatePtr, sigIndex, elemToInt, elemSizeToInt}));
          }
        }
        return WalkResult::advance();
      });

      if (sigWalkResult.wasInterrupted())
        return failure();

    } else if (auto proc = module.lookupSymbol<ProcOp>(instOp.getCallee())) {
      // Handle process instantiation.
      auto sensesTy = LLVM::LLVMArrayType::get(i1Ty, proc.getNumArguments());
      auto procStateTy = LLVM::LLVMStructType::getLiteral(
          rewriter.getContext(),
          {i32Ty, i32Ty, voidPtrTy /*ptr(sensesTy)*/,
           getProcPersistenceTy(&getDialect(), typeConverter, proc)});

      auto zeroC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));

      // Malloc space for the process state.
      auto procStateNullPtr =
          initBuilder.create<LLVM::ZeroOp>(op->getLoc(), voidPtrTy);
      auto procStateGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), voidPtrTy, procStateTy, procStateNullPtr,
          ArrayRef<LLVM::GEPArg>({1}));
      auto procStateSize = initBuilder.create<LLVM::PtrToIntOp>(
          op->getLoc(), i64Ty, procStateGep);
      std::array<Value, 1> procStateMArgs({procStateSize});
      auto procStateMall = initBuilder
                               .create<LLVM::CallOp>(
                                   op->getLoc(), voidPtrTy,
                                   SymbolRefAttr::get(mallFunc), procStateMArgs)
                               .getResult();

      // Store the initial resume index.
      auto resumeGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), voidPtrTy, procStateTy, procStateMall,
          ArrayRef<LLVM::GEPArg>({0, 1}));
      initBuilder.create<LLVM::StoreOp>(op->getLoc(), zeroC, resumeGep);

      // Malloc space for the senses table.
      auto sensesNullPtr =
          initBuilder.create<LLVM::ZeroOp>(op->getLoc(), voidPtrTy);
      auto sensesGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), voidPtrTy, sensesTy, sensesNullPtr,
          ArrayRef<LLVM::GEPArg>({1}));
      auto sensesSize =
          initBuilder.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, sensesGep);
      std::array<Value, 1> senseMArgs({sensesSize});
      auto sensesMall =
          initBuilder
              .create<LLVM::CallOp>(op->getLoc(), voidPtrTy,
                                    SymbolRefAttr::get(mallFunc), senseMArgs)
              .getResult();

      // Set all initial senses to 1.
      auto oneB = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getBoolAttr(true));
      for (size_t i = 0, e = sensesTy.getNumElements(); i < e; ++i) {
        auto senseGep = initBuilder.create<LLVM::GEPOp>(
            op->getLoc(), voidPtrTy, i1Ty, sensesMall,
            ArrayRef<LLVM::GEPArg>({i}));
        initBuilder.create<LLVM::StoreOp>(op->getLoc(), oneB, senseGep);
      }

      // Store the senses pointer in the process state.
      auto procStateSensesPtr = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), voidPtrTy, procStateTy, procStateMall,
          ArrayRef<LLVM::GEPArg>({0, 2}));
      initBuilder.create<LLVM::StoreOp>(op->getLoc(), sensesMall,
                                        procStateSensesPtr);

      std::array<Value, 3> allocProcArgs({initStatePtr, owner, procStateMall});
      initBuilder.create<LLVM::CallOp>(op->getLoc(), std::nullopt,
                                       SymbolRefAttr::get(allocProcFunc),
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
  explicit SigOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter,
                           size_t &sigCounter)
      : ConvertToLLVMPattern(llhd::SigOp::getOperationName(), ctx,
                             typeConverter),
        sigCounter(sigCounter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the adapted opreands.
    SigOpAdaptor transformed(operands);

    // Collect the used llvm types.
    auto voidPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto sigTy = getLLVMSigType(&getDialect());

    // Get the signal table pointer from the arguments.
    Value sigTablePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(2);

    // Get the index in the signal table and increase counter.
    // Insert a gep to the signal index in the signal table argument.
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, voidPtrTy, sigTy, sigTablePtr,
                                             LLVM::GEPArg(sigCounter));
    ++sigCounter;

    return success();
  }

private:
  size_t &sigCounter;
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
    auto resTy = prbOp.getType();
    auto finalTy = typeConverter->convertType(resTy);

    // Get the signal details from the signal struct.
    auto sigDetail = getSignalDetail(rewriter, &getDialect(), op->getLoc(),
                                     transformed.getSignal());

    if (isa<IntegerType>(resTy)) {
      // Get the amount of bytes to load. An extra byte is always loaded to
      // cover the case where a subsignal spans halfway in the last byte.
      int resWidth = resTy.getIntOrFloatBitWidth();
      int loadWidth = (llvm::divideCeil(resWidth, 8) + 1) * 8;
      auto loadTy = IntegerType::get(rewriter.getContext(), loadWidth);

      auto loadSig =
          rewriter.create<LLVM::LoadOp>(op->getLoc(), loadTy, sigDetail[0]);

      // Shift the loaded value by the offset and truncate to the final width.
      auto trOff = adjustBitWidth(op->getLoc(), rewriter, loadTy, sigDetail[1]);
      auto shifted =
          rewriter.create<LLVM::LShrOp>(op->getLoc(), loadTy, loadSig, trOff);
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, finalTy, shifted);

      return success();
    }

    if (isa<hw::ArrayType, hw::StructType>(resTy)) {
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, finalTy, sigDetail[0]);

      return success();
    }

    return failure();
  }
};
} // namespace

namespace {
/// Convert an `llhd.drv` operation to LLVM dialect. The result is a library
/// call to the
/// `@driveSignal` function, which declaration is inserted at the beginning of
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
    auto voidPtrTy = getVoidPtrType();
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Get or insert the drive library call.
    auto drvFuncTy = LLVM::LLVMFunctionType::get(
        voidTy, {voidPtrTy, voidPtrTy, voidPtrTy, i64Ty, i64Ty, i64Ty, i64Ty});
    auto drvFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                       "driveSignal", drvFuncTy);

    // Get the state pointer from the function arguments.
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);

    // Get signal width.
    Value sigWidth;
    auto underlyingTy = drvOp.getValue().getType();
    if (isArrayOrStruct(underlyingTy)) {
      auto underlyingTyConv = typeConverter->convertType(underlyingTy);
      auto eightC = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i64Ty, rewriter.getI64IntegerAttr(8));
      auto nullPtr = rewriter.create<LLVM::ZeroOp>(op->getLoc(), voidPtrTy);
      auto gepOne = rewriter.create<LLVM::GEPOp>(op->getLoc(), voidPtrTy,
                                                 underlyingTyConv, nullPtr,
                                                 ArrayRef<LLVM::GEPArg>({1}));
      auto toInt =
          rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, gepOne);
      sigWidth = rewriter.create<LLVM::MulOp>(op->getLoc(), toInt, eightC);
    } else {
      sigWidth = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i64Ty,
          rewriter.getI64IntegerAttr(underlyingTy.getIntOrFloatBitWidth()));
    }

    // Insert enable comparison. Skip if the enable operand is 0.
    if (auto gate = drvOp.getEnable()) {
      auto block = op->getBlock();
      auto continueBlock =
          rewriter.splitBlock(rewriter.getInsertionBlock(), op->getIterator());
      auto drvBlock = rewriter.createBlock(continueBlock);
      rewriter.setInsertionPointToEnd(drvBlock);
      rewriter.create<LLVM::BrOp>(op->getLoc(), ValueRange(), continueBlock);

      rewriter.setInsertionPointToEnd(block);
      auto oneC = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getI16IntegerAttr(1));
      auto cmp = rewriter.create<LLVM::ICmpOp>(
          op->getLoc(), LLVM::ICmpPredicate::eq, transformed.getEnable(), oneC);
      rewriter.create<LLVM::CondBrOp>(op->getLoc(), cmp, drvBlock,
                                      continueBlock);

      rewriter.setInsertionPointToStart(drvBlock);
    }

    Type valTy = typeConverter->convertType(transformed.getValue().getType());
    Value castVal = typeConverter->materializeTargetConversion(
        rewriter, transformed.getValue().getLoc(), valTy,
        transformed.getValue());

    auto oneConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));

    // This assumes that alloca does always allocate full bytes (round up to a
    // multiple of 8 bits).
    auto alloca = rewriter.create<LLVM::AllocaOp>(op->getLoc(), voidPtrTy,
                                                  valTy, oneConst, 4);
    rewriter.create<LLVM::StoreOp>(op->getLoc(), castVal, alloca);

    // Get the time values.
    auto realTime = rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), transformed.getTime(), 0);
    auto delta = rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), transformed.getTime(), 1);
    auto eps = rewriter.create<LLVM::ExtractValueOp>(op->getLoc(),
                                                     transformed.getTime(), 2);

    // Define the driveSignal library call arguments.
    std::array<Value, 7> args({statePtr, transformed.getSignal(), alloca,
                               sigWidth, realTime, delta, eps});
    // Create the library call.
    rewriter.create<LLVM::CallOp>(op->getLoc(), std::nullopt,
                                  SymbolRefAttr::get(drvFunc), args);

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

namespace {
/// Convert an `llhd.reg` operation to LLVM dialect. This generates a series of
/// comparisons (blocks) that end up driving the signal with the arguments of
/// the first matching trigger from the trigger list.
struct RegOpConversion : public ConvertToLLVMPattern {
  explicit RegOpConversion(MLIRContext *ctx, LLVMTypeConverter &typeConverter,
                           size_t &regCounter)
      : ConvertToLLVMPattern(RegOp::getOperationName(), ctx, typeConverter),
        regCounter(regCounter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto regOp = cast<RegOp>(op);
    RegOpAdaptor transformed(operands, op->getAttrDictionary());

    auto voidPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();

    // Retrieve and update previous trigger values for rising/falling edge
    // detection.
    size_t triggerIndex = 0;
    SmallVector<Value, 4> prevTriggers;
    for (int i = 0, e = regOp.getValues().size(); i < e; ++i) {
      auto mode = regOp.getRegModeAt(i);
      if (mode == RegMode::both || mode == RegMode::fall ||
          mode == RegMode::rise) {
        auto gep = rewriter.create<LLVM::GEPOp>(
            op->getLoc(), voidPtrTy, i1Ty, func.getArgument(1),
            ArrayRef<LLVM::GEPArg>({0, regCounter, triggerIndex++}));
        prevTriggers.push_back(
            rewriter.create<LLVM::LoadOp>(op->getLoc(), i1Ty, gep));
        rewriter.create<LLVM::StoreOp>(op->getLoc(),
                                       transformed.getTriggers()[i], gep);
      }
    }

    // Create blocks for drive and continue.
    auto block = op->getBlock();
    auto continueBlock = block->splitBlock(op);

    auto drvBlock = rewriter.createBlock(continueBlock);
    auto valArg = drvBlock->addArgument(transformed.getValues()[0].getType(),
                                        transformed.getValues()[0].getLoc());
    auto delayArg = drvBlock->addArgument(transformed.getDelays()[0].getType(),
                                          transformed.getDelays()[0].getLoc());
    auto gateArg = drvBlock->addArgument(i1Ty, rewriter.getUnknownLoc());

    // Create a drive with the block arguments.
    rewriter.setInsertionPointToStart(drvBlock);
    rewriter.create<DrvOp>(op->getLoc(), regOp.getSignal(), valArg, delayArg,
                           gateArg);
    rewriter.create<LLVM::BrOp>(op->getLoc(), ValueRange(), continueBlock);

    int j = prevTriggers.size() - 1;
    // Create a comparison block for each of the reg tuples.
    for (int i = regOp.getValues().size() - 1, e = i; i >= 0; --i) {
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
          {transformed.getValues()[i], transformed.getDelays()[i], gate});

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
                                          transformed.getTriggers()[i], rhs);

      // Create comparison for modes needing more than one state of the trigger.
      Value brCond;
      if (mode == RegMode::rise || mode == RegMode::fall ||
          mode == RegMode::both) {

        auto cmpPrev = rewriter.create<LLVM::ICmpOp>(
            op->getLoc(), LLVM::ICmpPredicate::ne, transformed.getTriggers()[i],
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

private:
  size_t &regCounter;
}; // namespace
} // namespace

//===----------------------------------------------------------------------===//
// Value creation conversions
//===----------------------------------------------------------------------===//

namespace {
/// Lower an LLHD constant operation to an equivalent LLVM dialect constant
/// operation.
struct ConstantTimeOpConversion : public ConvertToLLVMPattern {
  explicit ConstantTimeOpConversion(MLIRContext *ctx,
                                    LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(llhd::ConstantTimeOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operand,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the ConstOp.
    auto constOp = cast<ConstantTimeOp>(op);
    // Get the constant's attribute.
    TimeAttr timeAttr = constOp.getValueAttr();
    // Handle the time const special case: create a new array containing the
    // three time values.
    auto timeTy = typeConverter->convertType(constOp.getResult().getType());

    // Convert real-time element to ps.
    llvm::StringMap<uint64_t> map = {
        {"s", 12}, {"ms", 9}, {"us", 6}, {"ns", 3}, {"ps", 0}};
    uint64_t adjusted =
        std::pow(10, map[timeAttr.getTimeUnit()]) * timeAttr.getTime();

    // Get sub-steps.
    uint64_t delta = timeAttr.getDelta();
    uint64_t eps = timeAttr.getEpsilon();

    // Create time constant.
    auto denseAttr =
        DenseElementsAttr::get(RankedTensorType::get(3, rewriter.getI64Type()),
                               {adjusted, delta, eps});
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, timeTy, denseAttr);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Extraction operation conversions
//===----------------------------------------------------------------------===//

namespace {
/// Convert a DynExtractSliceOp to LLVM dialect.
struct SigArraySliceOpConversion
    : public ConvertOpToLLVMPattern<llhd::SigArraySliceOp> {
  using ConvertOpToLLVMPattern<llhd::SigArraySliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(llhd::SigArraySliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type llvmArrTy = typeConverter->convertType(op.getInputArrayType());
    Type inputTy = typeConverter->convertType(op.getInput().getType());
    Type lowIndexTy = typeConverter->convertType(op.getLowIndex().getType());
    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), inputTy, op.getInput());
    Value castLowIndex = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), lowIndexTy, op.getLowIndex());

    auto sigDetail = getSignalDetail(rewriter, &getDialect(), op->getLoc(),
                                     castInput, /*extractIndices=*/true);

    auto adjustedPtr = shiftArraySigPointer(op->getLoc(), rewriter, llvmArrTy,
                                            sigDetail[0], castLowIndex);
    rewriter.replaceOp(op, createSubSig(&getDialect(), rewriter, op->getLoc(),
                                        sigDetail, adjustedPtr, sigDetail[1]));
    return success();
  }
};
} // namespace

namespace {
/// Convert a DynExtractSliceOp to LLVM dialect.
struct SigExtractOpConversion
    : public ConvertOpToLLVMPattern<llhd::SigExtractOp> {
  using ConvertOpToLLVMPattern<llhd::SigExtractOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(llhd::SigExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type inputTy = typeConverter->convertType(op.getInput().getType());
    Type lowBitTy = typeConverter->convertType(op.getLowBit().getType());
    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), inputTy, op.getInput());
    Value castLowBit = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), lowBitTy, op.getLowBit());

    auto sigDetail = getSignalDetail(rewriter, &getDialect(), op->getLoc(),
                                     castInput, /*extractIndices=*/true);

    auto zextStart = adjustBitWidth(op->getLoc(), rewriter,
                                    rewriter.getI64Type(), castLowBit);
    // Adjust the slice starting point by the signal's offset.
    auto adjustedStart =
        rewriter.create<LLVM::AddOp>(op->getLoc(), sigDetail[1], zextStart);

    auto adjusted = shiftIntegerSigPointer(
        op->getLoc(), &getDialect(), rewriter, sigDetail[0], adjustedStart);
    // Create a new subsignal with the new pointer and offset.
    rewriter.replaceOp(op, createSubSig(&getDialect(), rewriter, op->getLoc(),
                                        sigDetail, adjusted.first,
                                        adjusted.second));
    return success();
  }
};
} // namespace

namespace {
/// Convert
struct SigStructExtractOpConversion
    : public ConvertOpToLLVMPattern<llhd::SigStructExtractOp> {
  using ConvertOpToLLVMPattern<
      llhd::SigStructExtractOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(llhd::SigStructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type llvmStructTy = typeConverter->convertType(op.getStructType());
    Type inputTy = typeConverter->convertType(op.getInput().getType());
    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), inputTy, op.getInput());

    std::vector<Value> sigDetail =
        getSignalDetail(rewriter, &getDialect(), op->getLoc(), castInput,
                        /*extractIndices=*/true);

    uint32_t index = HWToLLVMEndianessConverter::llvmIndexOfStructField(
        op.getStructType(), op.getField());

    Value adjusted = shiftStructuredSigPointer(
        op->getLoc(), rewriter, llvmStructTy, sigDetail[0], index);

    rewriter.replaceOp(op, createSubSig(&getDialect(), rewriter, op->getLoc(),
                                        sigDetail, adjusted, sigDetail[1]));

    return success();
  }
};
} // namespace

namespace {
/// Convert a DynExtractElementOp to LLVM dialect.
struct SigArrayGetOpConversion
    : public ConvertOpToLLVMPattern<llhd::SigArrayGetOp> {
  using ConvertOpToLLVMPattern<llhd::SigArrayGetOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(llhd::SigArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto llvmArrTy = typeConverter->convertType(op.getArrayType());
    Type inputTy = typeConverter->convertType(op.getInput().getType());
    Type indexTy = typeConverter->convertType(op.getIndex().getType());
    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), inputTy, op.getInput());
    Value castIndex = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), indexTy, op.getIndex());

    auto sigDetail = getSignalDetail(rewriter, &getDialect(), op->getLoc(),
                                     castInput, /*extractIndices=*/true);

    auto adjustedPtr = shiftArraySigPointer(op->getLoc(), rewriter, llvmArrTy,
                                            sigDetail[0], castIndex);
    rewriter.replaceOp(op, createSubSig(&getDialect(), rewriter, op->getLoc(),
                                        sigDetail, adjustedPtr, sigDetail[1]));

    return success();
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

    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    Type initTy = typeConverter->convertType(transformed.getInit().getType());

    auto oneC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
    auto alloca = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(rewriter.getContext()), initTy,
        oneC, 4);
    rewriter.create<LLVM::StoreOp>(op->getLoc(), transformed.getInit(), alloca);
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

    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, transformed.getValue(),
                                               transformed.getPointer());

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
    : public circt::impl::ConvertLLHDToLLVMBase<LLHDToLLVMLoweringPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateLLHDToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 size_t &sigCounter,
                                                 size_t &regCounter) {
  MLIRContext *ctx = converter.getDialect()->getContext();

  // Value creation conversion patterns.
  patterns.add<ConstantTimeOpConversion>(ctx, converter);

  // Extract conversion patterns.
  patterns.add<SigExtractOpConversion, SigArraySliceOpConversion,
               SigArrayGetOpConversion, SigStructExtractOpConversion>(
      converter);

  // Unit conversion patterns.
  patterns.add<ProcOpConversion, WaitOpConversion, HaltOpConversion>(ctx,
                                                                     converter);
  patterns.add<EntityOpConversion>(ctx, converter, sigCounter, regCounter);

  // Signal conversion patterns.
  patterns.add<PrbOpConversion, DrvOpConversion>(ctx, converter);
  patterns.add<SigOpConversion>(ctx, converter, sigCounter);
  patterns.add<RegOpConversion>(ctx, converter, regCounter);

  // Memory conversion patterns.
  patterns.add<VarOpConversion, StoreOpConversion>(ctx, converter);
  patterns.add<LoadOpConversion>(converter);
}

void circt::populateLLHDToLLVMTypeConversions(LLVMTypeConverter &converter) {
  converter.addConversion(
      [&](SigType sig) { return convertSigType(sig, converter); });
  converter.addConversion(
      [&](TimeType time) { return convertTimeType(time, converter); });
  converter.addConversion(
      [&](PtrType ptr) { return convertPtrType(ptr, converter); });
}

void LLHDToLLVMLoweringPass::runOnOperation() {
  Namespace globals;
  SymbolCache cache;
  cache.addDefinitions(getOperation());
  globals.add(cache);

  // Keep a counter to infer a signal's index in his entity's signal table.
  size_t sigCounter = 0;

  // Keep a counter to infer a reg's index in his entity.
  size_t regCounter = 0;

  RewritePatternSet patterns(&getContext());
  auto converter = mlir::LLVMTypeConverter(&getContext());
  populateLLHDToLLVMTypeConversions(converter);

  // Also populate with HW type conversions
  populateHWToLLVMTypeConversions(converter);

  // Apply a partial conversion first, lowering only the instances, to generate
  // the init function.
  patterns.add<InstOpConversion>(&getContext(), converter);

  LLVMConversionTarget target(getContext());
  target.addIllegalOp<InstOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);

  // Apply the partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
  patterns.clear();

  // Setup the full conversion.
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateLLHDToLLVMConversionPatterns(converter, patterns, sigCounter,
                                       regCounter);

  // Populate with HW and Comb conversion patterns
  DenseMap<std::pair<Type, ArrayAttr>, LLVM::GlobalOp> constAggregateGlobalsMap;
  populateHWToLLVMConversionPatterns(converter, patterns, globals,
                                     constAggregateGlobalsMap);
  populateCombToLLVMConversionPatterns(converter, patterns);
  populateCombToArithConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();

  // Apply a full conversion to remove unrealized conversion casts.
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();

  patterns.clear();

  mlir::populateReconcileUnrealizedCastsPatterns(patterns);
  target.addIllegalOp<UnrealizedConversionCastOp>();

  // Apply the full conversion.
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create an LLHD to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertLLHDToLLVMPass() {
  return std::make_unique<LLHDToLLVMLoweringPass>();
}
