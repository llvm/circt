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
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace circt::llhd;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

uint32_t convertToLLVMEndianess(Type type, uint32_t index) {
  // This is hardcoded for little endian machines for now.
  return TypeSwitch<Type, uint32_t>(type)
      .Case<hw::ArrayType>(
          [&](hw::ArrayType ty) { return ty.getSize() - index - 1; })
      .Case<hw::StructType>([&](hw::StructType ty) {
        return ty.getElements().size() - index - 1;
      });
}

uint32_t llvmIndexOfStructField(hw::StructType type, StringRef fieldName) {
  auto fieldIter = type.getElements();
  size_t index = 0;

  for (const auto *iter = fieldIter.begin(); iter != fieldIter.end(); ++iter) {
    if (iter->name.equals(fieldName)) {
      return convertToLLVMEndianess(type, index);
    }
    ++index;
  }

  // Verifier of StructExtractOp has to ensure that the field name is indeed
  // present.
  llvm_unreachable("Field name attribute of hw::StructExtractOp invalid");
  return 0;
}

/// Get an existing global string.
static Value getGlobalString(Location loc, OpBuilder &builder,
                             TypeConverter *typeConverter,
                             LLVM::GlobalOp &str) {
  auto i8PtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8));
  auto i32Ty = IntegerType::get(builder.getContext(), 32);

  auto addr = builder.create<LLVM::AddressOfOp>(
      loc, LLVM::LLVMPointerType::get(str.getType()), str.getName());
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
                    Location loc, std::string name, Type signature,
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
static Type getLLVMSigType(LLVM::LLVMDialect *dialect) {
  auto i8PtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(dialect->getContext(), 8));
  auto i64Ty = IntegerType::get(dialect->getContext(), 64);
  return LLVM::LLVMStructType::getLiteral(dialect->getContext(),
                                          {i8PtrTy, i64Ty, i64Ty, i64Ty});
}

/// Extract the details from the given signal struct. The details are returned
/// in the original struct order.
static std::vector<Value> getSignalDetail(ConversionPatternRewriter &rewriter,
                                          LLVM::LLVMDialect *dialect,
                                          Location loc, Value signal,
                                          bool extractIndices = false) {

  auto i8PtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(dialect->getContext(), 8));
  auto i32Ty = IntegerType::get(dialect->getContext(), 32);
  auto i64Ty = IntegerType::get(dialect->getContext(), 64);

  std::vector<Value> result;

  // Extract the value and offset elements.
  auto zeroC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto oneC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                rewriter.getI32IntegerAttr(1));

  auto sigPtrPtr =
      rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(i8PtrTy),
                                   signal, ArrayRef<Value>({zeroC, zeroC}));
  result.push_back(rewriter.create<LLVM::LoadOp>(loc, i8PtrTy, sigPtrPtr));

  auto offsetPtr =
      rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(i64Ty),
                                   signal, ArrayRef<Value>({zeroC, oneC}));
  result.push_back(rewriter.create<LLVM::LoadOp>(loc, i64Ty, offsetPtr));

  // Extract the instance and global indices.
  if (extractIndices) {
    auto twoC = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(2));
    auto threeC = rewriter.create<LLVM::ConstantOp>(
        loc, i32Ty, rewriter.getI32IntegerAttr(3));

    auto instIndexPtr =
        rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(i64Ty),
                                     signal, ArrayRef<Value>({zeroC, twoC}));
    result.push_back(rewriter.create<LLVM::LoadOp>(loc, i64Ty, instIndexPtr));

    auto globalIndexPtr =
        rewriter.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(i64Ty),
                                     signal, ArrayRef<Value>({zeroC, threeC}));
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
  auto allocaSubSig = rewriter.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(sigTy), oneC, 4);
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

/// Unwrap the given LLVM pointer type, returning its element value.
static Type unwrapLLVMPtr(Type ty) {
  auto castTy = ty.cast<LLVM::LLVMPointerType>();
  return castTy.getElementType();
}

/// Gather the types of values that are used outside of the block they're
/// defined in. An LLVMType structure containing those types, in order of
/// appearance, is returned.
static Type getProcPersistenceTy(LLVM::LLVMDialect *dialect,
                                 TypeConverter *converter, ProcOp &proc) {
  SmallVector<Type, 3> types = SmallVector<Type, 3>();
  proc.walk([&](Operation *op) -> void {
    if (op->isUsedOutsideOfBlock(op->getBlock()) || isWaitDestArg(op)) {
      auto ty = op->getResult(0).getType();
      auto convertedTy = converter->convertType(ty);
      if (ty.isa<PtrType, SigType>()) {
        // Persist the unwrapped value.
        types.push_back(unwrapLLVMPtr(convertedTy));
      } else {
        // Persist the value as is.
        types.push_back(convertedTy);
      }
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
                                 Type elementTy, int index, Value state) {
  auto i32Ty = IntegerType::get(dialect->getContext(), 32);
  auto zeroC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto threeC = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(3));
  auto indC = rewriter.create<LLVM::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(index));
  return rewriter.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(elementTy), state,
      ArrayRef<Value>({zeroC, threeC, indC}));
}

/// Persist a `Value` by storing it into the process persistence table, and
/// substituting the uses that escape the block the operation is defined in with
/// a load from the persistence table.
static void persistValue(LLVM::LLVMDialect *dialect, Location loc,
                         TypeConverter *converter,
                         ConversionPatternRewriter &rewriter, Type stateTy,
                         int &i, Value state, Value persist) {
  auto elemTy = stateTy.cast<LLVM::LLVMStructType>()
                    .getBody()[3]
                    .cast<LLVM::LLVMStructType>()
                    .getBody()[i];

  if (auto arg = persist.dyn_cast<BlockArgument>()) {
    rewriter.setInsertionPointToStart(arg.getParentBlock());
  } else {
    rewriter.setInsertionPointAfter(persist.getDefiningOp());
  }

  Value convPersist = converter->materializeTargetConversion(
      rewriter, loc, converter->convertType(persist.getType()), {persist});

  auto gep0 = gepPersistenceState(dialect, loc, rewriter, elemTy, i, state);

  Value toStore;
  if (auto ptr = persist.getType().dyn_cast<PtrType>()) {
    // Unwrap the pointer and store it's value.
    auto elemTy = converter->convertType(ptr.getUnderlyingType());
    toStore = rewriter.create<LLVM::LoadOp>(loc, elemTy, convPersist);
  } else if (persist.getType().isa<SigType>()) {
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
    if (persist.getType().isa<PtrType>() && user != toStore.getDefiningOp() &&
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
static void insertPersistence(TypeConverter *converter,
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
  auto zeroC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto oneC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                rewriter.getI32IntegerAttr(1));
  auto gep = rewriter.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(i32Ty), converted.getArgument(1),
      ArrayRef<Value>({zeroC, oneC}));

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
                            wait.dest(), wait.destOps());

      // Insert the resume index update at the wait operation location.
      rewriter.setInsertionPoint(op);
      auto procState = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(1);
      auto resumeIdxC = rewriter.create<LLVM::ConstantOp>(
          loc, i32Ty, rewriter.getI32IntegerAttr(waitInd));
      auto resumeIdxPtr = rewriter.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(i32Ty), procState,
          ArrayRef<Value>({zeroC, oneC}));
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
    for (size_t i = 0; i < op.modes().size(); ++i) {
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

/// Recursively clone the init origin of a sig operation into the init function,
/// up to the initial constant value(s). This is required to clone the
/// initialization of array and struct signals, where the init operant cannot
/// originate from a constant operation. Integer constants are currently assumed
/// to come from a constant operation.
static Operation *recursiveCloneInit(OpBuilder &initBuilder, Operation *op) {
  if (auto arrayCreateOp = dyn_cast<hw::ArrayCreateOp>(op)) {
    auto def =
        cast<hw::ArrayCreateOp>(initBuilder.insert(arrayCreateOp.clone()));
    initBuilder.setInsertionPoint(def.getOperation());
    for (size_t i = 0, e = def.inputs().size(); i < e; ++i) {
      auto clone =
          recursiveCloneInit(initBuilder, def.inputs()[i].getDefiningOp());
      def.setOperand(i, clone->getResult(0));
    }
    initBuilder.setInsertionPointAfter(def.getOperation());
    return def;
  }

  if (auto structCreateOp = dyn_cast<hw::StructCreateOp>(op)) {
    auto def =
        cast<hw::StructCreateOp>(initBuilder.insert(structCreateOp.clone()));
    initBuilder.setInsertionPoint(def.getOperation());
    for (size_t i = 0, e = def.input().size(); i < e; ++i) {
      auto clone =
          recursiveCloneInit(initBuilder, def.input()[i].getDefiningOp());
      def.setOperand(i, clone->getResult(0));
    }
    initBuilder.setInsertionPointAfter(def.getOperation());
    return def;
  }

  return initBuilder.insert(op->clone());
}

/// Check if the given type is either of LLHD's ArrayType, StructType, or LLVM
/// array or struct type.
static bool isArrayOrStruct(Type type) {
  return type.isa<LLVM::LLVMArrayType, LLVM::LLVMStructType, hw::ArrayType,
                  hw::StructType>();
}

/// Shift an integer signal pointer to obtain a view of the underlying value as
/// if it was shifted.
static std::pair<Value, Value>
shiftIntegerSigPointer(Location loc, LLVM::LLVMDialect *dialect,
                       ConversionPatternRewriter &rewriter, Value pointer,
                       Value index) {
  auto i8PtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(dialect->getContext(), 8));
  auto i64Ty = IntegerType::get(dialect->getContext(), 64);

  auto ptrToInt = rewriter.create<LLVM::PtrToIntOp>(loc, i64Ty, pointer);
  auto const8 = rewriter.create<LLVM::ConstantOp>(
      loc, index.getType(), rewriter.getI64IntegerAttr(8));
  auto ptrOffset = rewriter.create<LLVM::UDivOp>(loc, index, const8);
  auto shiftedPtr = rewriter.create<LLVM::AddOp>(loc, ptrToInt, ptrOffset);
  auto newPtr = rewriter.create<LLVM::IntToPtrOp>(loc, i8PtrTy, shiftedPtr);

  // Compute the new offset into the first byte.
  auto bitOffset = rewriter.create<LLVM::URemOp>(loc, index, const8);

  return std::make_pair(newPtr, bitOffset);
}

/// Shift the pointer of a structured-type (array or struct) signal, to change
/// its view as if the desired slice/element was extracted.
static Value shiftStructuredSigPointer(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       Type structTy, Type elemPtrTy,
                                       Value pointer, Value index) {
  auto dialect = &structTy.getDialect();
  auto i8PtrTy =
      LLVM::LLVMPointerType::get(IntegerType::get(dialect->getContext(), 8));
  auto i32Ty = IntegerType::get(dialect->getContext(), 32);

  auto zeroC = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(0));
  auto bitcastToArr = rewriter.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMPointerType::get(structTy), pointer);
  auto gep = rewriter.create<LLVM::GEPOp>(loc, elemPtrTy, bitcastToArr,
                                          ArrayRef<Value>({zeroC, index}));
  return rewriter.create<LLVM::BitcastOp>(loc, i8PtrTy, gep);
}

/// Shift the pointer of an array-typed signal, to change its view as if the
/// desired slice/element was extracted.
static Value shiftArraySigPointer(Location loc,
                                  ConversionPatternRewriter &rewriter,
                                  Type arrTy, Value pointer, Value index) {
  auto elemPtrTy = LLVM::LLVMPointerType::get(
      arrTy.cast<LLVM::LLVMArrayType>().getElementType());
  auto zextIndex = zextByOne(loc, rewriter, index);
  return shiftStructuredSigPointer(loc, rewriter, arrTy, elemPtrTy, pointer,
                                   zextIndex);
}

//===----------------------------------------------------------------------===//
// Type conversions
//===----------------------------------------------------------------------===//

static Type convertSigType(SigType type, LLVMTypeConverter &converter) {
  auto &context = converter.getContext();
  auto i64Ty = IntegerType::get(&context, 64);
  auto i8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(&context, 8));
  return LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getLiteral(
      &context, {i8PtrTy, i64Ty, i64Ty, i64Ty}));
}

static Type convertTimeType(TimeType type, LLVMTypeConverter &converter) {
  auto i64Ty = IntegerType::get(&converter.getContext(), 64);
  return LLVM::LLVMArrayType::get(i64Ty, 3);
}

static Type convertPtrType(PtrType type, LLVMTypeConverter &converter) {
  return LLVM::LLVMPointerType::get(
      converter.convertType(type.getUnderlyingType()));
}

static Type convertArrayType(hw::ArrayType type, LLVMTypeConverter &converter) {
  auto elementTy = converter.convertType(type.getElementType());
  return LLVM::LLVMArrayType::get(elementTy, type.getSize());
}

static Type convertStructType(hw::StructType type,
                              LLVMTypeConverter &converter) {
  llvm::SmallVector<Type, 8> elements;
  mlir::SmallVector<mlir::Type> types;
  type.getInnerTypes(types);

  for (int i = 0, e = types.size(); i < e; ++i)
    elements.push_back(
        converter.convertType(types[convertToLLVMEndianess(type, i)]));

  return LLVM::LLVMStructType::getLiteral(&converter.getContext(), elements);
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
    auto i8PtrTy = getVoidPtrType();
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto sigTy = getLLVMSigType(&getDialect());
    auto entityStatePtrTy =
        LLVM::LLVMPointerType::get(getRegStateTy(&getDialect(), op));

    regCounter = 0;

    // Use an intermediate signature conversion to add the arguments for the
    // state and signal table pointer arguments.
    LLVMTypeConverter::SignatureConversion intermediate(
        entityOp.getNumArguments());
    // Add state and signal table arguments.
    intermediate.addInputs(std::array<Type, 3>(
        {i8PtrTy, entityStatePtrTy, LLVM::LLVMPointerType::get(sigTy)}));
    for (size_t i = 0, e = entityOp.getNumArguments(); i < e; ++i)
      intermediate.addInputs(i, voidTy);
    rewriter.applySignatureConversion(&entityOp.getBody(), intermediate,
                                      typeConverter);

    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&entityOp.getBlocks().front());
    LLVMTypeConverter::SignatureConversion final(
        intermediate.getConvertedTypes().size());
    final.addInputs(0, i8PtrTy);
    final.addInputs(1, entityStatePtrTy);
    final.addInputs(2, LLVM::LLVMPointerType::get(sigTy));

    // The first n elements of the signal table represent the entity arguments,
    // while the remaining elements represent the entity's owned signals.
    sigCounter = entityOp.getNumArguments();
    for (size_t i = 0; i < sigCounter; ++i) {
      // Create gep operations from the signal table for each original argument.
      auto index = bodyBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto gep = bodyBuilder.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(sigTy),
          entityOp.getArgument(2), ArrayRef<Value>(index));
      // Remap i-th original argument to the gep'd signal pointer.
      final.remapInput(i + 3, gep.getResult());
    }

    rewriter.applySignatureConversion(&entityOp.getBody(), final,
                                      typeConverter);

    // Get the converted entity signature.
    auto funcTy = LLVM::LLVMFunctionType::get(
        voidTy, {i8PtrTy, entityStatePtrTy, LLVM::LLVMPointerType::get(sigTy)});

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

private:
  size_t &sigCounter;
  size_t &regCounter;
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
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto senseTableTy = LLVM::LLVMPointerType::get(
        LLVM::LLVMArrayType::get(i1Ty, procOp.getNumArguments()));
    auto stateTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(),
        {/* current instance  */ i32Ty, /* resume index */ i32Ty,
         /* sense flags */ senseTableTy, /* persistent types */
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
    std::array<Type, 3> procArgTys({i8PtrTy,
                                    LLVM::LLVMPointerType::get(stateTy),
                                    LLVM::LLVMPointerType::get(sigTy)});
    intermediate.addInputs(procArgTys);
    for (size_t i = 0, e = procOp.getNumArguments(); i < e; ++i)
      intermediate.addInputs(i, voidTy);
    rewriter.applySignatureConversion(&procOp.getBody(), intermediate,
                                      typeConverter);

    // Get the final signature conversion.
    OpBuilder bodyBuilder =
        OpBuilder::atBlockBegin(&procOp.getBlocks().front());
    LLVMTypeConverter::SignatureConversion final(
        intermediate.getConvertedTypes().size());
    final.addInputs(0, i8PtrTy);
    final.addInputs(1, LLVM::LLVMPointerType::get(stateTy));
    final.addInputs(2, LLVM::LLVMPointerType::get(sigTy));

    for (size_t i = 0, e = procOp.getNumArguments(); i < e; ++i) {
      // Create gep operations from the signal table for each original argument.
      auto index = bodyBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto gep = bodyBuilder.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(sigTy),
          procOp.getArgument(2), ArrayRef<Value>({index}));

      // Remap the i-th original argument to the gep'd value.
      final.remapInput(i + 3, gep.getResult());
    }

    // Get the converted process signature.
    auto funcTy = LLVM::LLVMFunctionType::get(
        voidTy, {i8PtrTy, LLVM::LLVMPointerType::get(stateTy),
                 LLVM::LLVMPointerType::get(sigTy)});
    // Create a new llvm function to house the lowered process.
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(),
                                                      procOp.getName(), funcTy);

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
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);

    auto llvmFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto procState = llvmFunc.getArgument(1);
    auto senseTableTy = procState.getType()
                            .cast<LLVM::LLVMPointerType>()
                            .getElementType()
                            .cast<LLVM::LLVMStructType>()
                            .getBody()[2]
                            .cast<LLVM::LLVMPointerType>()
                            .getElementType()
                            .cast<LLVM::LLVMArrayType>();

    // Get senses ptr from the process state argument.
    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
    auto twoC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(2));
    auto sensePtrGep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(),
        LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(senseTableTy)),
        procState, ArrayRef<Value>({zeroC, twoC}));
    auto sensePtr = rewriter.create<LLVM::LoadOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(senseTableTy), sensePtrGep);

    // Zero out all the senses flags.
    for (size_t i = 0, e = senseTableTy.getNumElements(); i < e; ++i) {
      auto indC = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
      auto zeroB = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getI32IntegerAttr(0));
      auto senseElemPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), sensePtr,
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
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Get the llhdSuspend runtime function.
    auto llhdSuspendTy = LLVM::LLVMFunctionType::get(
        voidTy, {i8PtrTy, i8PtrTy, i64Ty, i64Ty, i64Ty});
    auto module = op->getParentOfType<ModuleOp>();
    auto llhdSuspendFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                               "llhdSuspend", llhdSuspendTy);

    auto statePtr = llvmFunc.getArgument(0);
    auto procState = llvmFunc.getArgument(1);
    auto procStateTy = procState.getType();
    auto senseTableTy = procStateTy.cast<LLVM::LLVMPointerType>()
                            .getElementType()
                            .cast<LLVM::LLVMStructType>()
                            .getBody()[2]
                            .cast<LLVM::LLVMPointerType>()
                            .getElementType();

    // Get senses ptr.
    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));
    auto twoC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(2));
    auto sensePtrGep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(),
        LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(senseTableTy)),
        procState, ArrayRef<Value>({zeroC, twoC}));
    auto sensePtr = rewriter.create<LLVM::LoadOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(senseTableTy), sensePtrGep);

    // Reset sense table, if not all signals are observed.
    if (waitOp.obs().size() <
        senseTableTy.cast<LLVM::LLVMArrayType>().getNumElements()) {
      auto zeroB = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getBoolAttr(false));
      for (size_t i = 0,
                  e = senseTableTy.cast<LLVM::LLVMArrayType>().getNumElements();
           i < e; ++i) {
        auto indC = rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
        auto senseElemPtr = rewriter.create<LLVM::GEPOp>(
            op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), sensePtr,
            ArrayRef<Value>({zeroC, indC}));
        rewriter.create<LLVM::StoreOp>(op->getLoc(), zeroB, senseElemPtr);
      }
    }

    // Set sense flags for observed signals.
    for (auto observed : transformed.obs()) {
      auto instIndexPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(i64Ty), observed,
          ArrayRef<Value>({zeroC, twoC}));
      auto instIndex =
          rewriter.create<LLVM::LoadOp>(op->getLoc(), i64Ty, instIndexPtr);
      auto oneB = rewriter.create<LLVM::ConstantOp>(op->getLoc(), i1Ty,
                                                    rewriter.getBoolAttr(true));
      auto senseElementPtr = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), sensePtr,
          ArrayRef<Value>({zeroC, instIndex}));
      rewriter.create<LLVM::StoreOp>(op->getLoc(), oneB, senseElementPtr);
    }

    // Update and store the new resume index in the process state.
    auto procStateBC =
        rewriter.create<LLVM::BitcastOp>(op->getLoc(), i8PtrTy, procState);

    // Spawn scheduled event, if present.
    if (waitOp.time()) {
      auto realTime = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), i64Ty, transformed.time(), rewriter.getI32ArrayAttr(0));
      auto delta = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), i64Ty, transformed.time(), rewriter.getI32ArrayAttr(1));
      auto eps = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), i64Ty, transformed.time(), rewriter.getI32ArrayAttr(2));

      std::array<Value, 5> args({statePtr, procStateBC, realTime, delta, eps});
      rewriter.create<LLVM::CallOp>(op->getLoc(), llvm::None,
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
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Init function signature: (i8* %state) -> void.
    auto initFuncTy = LLVM::LLVMFunctionType::get(voidTy, {i8PtrTy});
    auto initFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(), "llhd_init",
                            initFuncTy, /*insertBodyAndTerminator=*/true);

    // Get or insert the malloc function definition.
    // Malloc function signature: (i64 %size) -> i8* %pointer.
    auto mallocSigFuncTy = LLVM::LLVMFunctionType::get(i8PtrTy, {i64Ty});
    auto mallFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                        "malloc", mallocSigFuncTy);

    // Get or insert the allocSignal library call definition.
    // allocSignal function signature: (i8* %state, i8* %sig_name, i8*
    // %sig_owner, i32 %value) -> i32 %sig_index.
    auto allocSigFuncTy = LLVM::LLVMFunctionType::get(
        i32Ty, {i8PtrTy, i32Ty, i8PtrTy, i8PtrTy, i64Ty});
    auto sigFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                       "allocSignal", allocSigFuncTy);

    // Add information about the elements of an array signal to the state.
    // Signature: (i8* state, i32 signalIndex, i32 size, i32 numElements) ->
    // void
    auto addSigArrElemFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {i8PtrTy, i32Ty, i32Ty, i32Ty});
    auto addSigElemFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(),
                            "addSigArrayElements", addSigArrElemFuncTy);

    // Add information about one element of a struct signal to the state.
    // Signature: (i8* state, i32 signalIndex, i32 offset, i32 size) -> void
    auto addSigStructElemFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {i8PtrTy, i32Ty, i32Ty, i32Ty});
    auto addSigStructFunc =
        getOrInsertFunction(module, rewriter, op->getLoc(),
                            "addSigStructElement", addSigStructElemFuncTy);

    // Get or insert allocProc library call definition.
    auto allocProcFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {i8PtrTy, i8PtrTy, i8PtrTy});
    auto allocProcFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                             "allocProc", allocProcFuncTy);

    // Get or insert allocEntity library call definition.
    auto allocEntityFuncTy =
        LLVM::LLVMFunctionType::get(voidTy, {i8PtrTy, i8PtrTy, i8PtrTy});
    auto allocEntityFunc = getOrInsertFunction(
        module, rewriter, op->getLoc(), "allocEntity", allocEntityFuncTy);

    Value initStatePtr = initFunc.getArgument(0);

    // Get a builder for the init function.
    OpBuilder initBuilder =
        OpBuilder::atBlockTerminator(&initFunc.getBody().getBlocks().front());

    // Use the instance name to retrieve the instance from the state.
    auto ownerName = entity.getName().str() + "." + instOp.name().str();

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
    if (auto child = module.lookupSymbol<EntityOp>(instOp.callee())) {
      auto regStateTy = getRegStateTy(&getDialect(), child.getOperation());
      auto regStatePtrTy = LLVM::LLVMPointerType::get(regStateTy);

      // Get reg state size.
      auto oneC = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
      auto regNull =
          initBuilder.create<LLVM::NullOp>(op->getLoc(), regStatePtrTy);
      auto regGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), regStatePtrTy, regNull, ArrayRef<Value>({oneC}));
      auto regSize =
          initBuilder.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, regGep);

      // Malloc reg state.
      auto regMall = initBuilder
                         .create<LLVM::CallOp>(op->getLoc(), i8PtrTy,
                                               SymbolRefAttr::get(mallFunc),
                                               ArrayRef<Value>({regSize}))
                         .getResult(0);
      auto regMallBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), regStatePtrTy, regMall);
      auto zeroB = initBuilder.create<LLVM::ConstantOp>(
          op->getLoc(), i1Ty, rewriter.getBoolAttr(false));

      // Zero-initialize reg state entries.
      for (size_t i = 0,
                  e = regStateTy.cast<LLVM::LLVMStructType>().getBody().size();
           i < e; ++i) {
        size_t f = regStateTy.cast<LLVM::LLVMStructType>()
                       .getBody()[i]
                       .cast<LLVM::LLVMArrayType>()
                       .getNumElements();
        for (size_t j = 0; j < f; ++j) {
          auto regIndexC = initBuilder.create<LLVM::ConstantOp>(
              op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
          auto triggerIndexC = initBuilder.create<LLVM::ConstantOp>(
              op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(j));
          auto regGep = initBuilder.create<LLVM::GEPOp>(
              op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), regMallBC,
              ArrayRef<Value>({zeroB, regIndexC, triggerIndexC}));
          initBuilder.create<LLVM::StoreOp>(op->getLoc(), zeroB, regGep);
        }
      }

      // Add reg state pointer to global state.
      initBuilder.create<LLVM::CallOp>(
          op->getLoc(), llvm::None, SymbolRefAttr::get(allocEntityFunc),
          ArrayRef<Value>({initStatePtr, owner, regMall}));

      // Index of the signal in the entity's signal table.
      int initCounter = 0;
      // Walk over the entity and generate mallocs for each one of its signals.
      child.walk([&](SigOp op) -> void {
        // if (auto sigOp = dyn_cast<SigOp>(op)) {
        auto underlyingTy = typeConverter->convertType(op.init().getType());
        // Get index constant of the signal in the entity's signal table.
        auto indexConst = initBuilder.create<LLVM::ConstantOp>(
            op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(initCounter));
        initCounter++;

        // Clone and insert the operation that defines the signal's init
        // operand (assmued to be a constant/array op)
        auto defOp = op.init().getDefiningOp();
        auto initDef = recursiveCloneInit(initBuilder, defOp)->getResult(0);
        Value initDefCast = typeConverter->materializeTargetConversion(
            initBuilder, initDef.getLoc(),
            typeConverter->convertType(initDef.getType()), initDef);

        // Compute the required space to malloc.
        auto oneC = initBuilder.create<LLVM::ConstantOp>(
            op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
        auto twoC = initBuilder.create<LLVM::ConstantOp>(
            op.getLoc(), i64Ty, rewriter.getI32IntegerAttr(2));
        auto nullPtr = initBuilder.create<LLVM::NullOp>(
            op.getLoc(), LLVM::LLVMPointerType::get(underlyingTy));
        auto sizeGep = initBuilder.create<LLVM::GEPOp>(
            op.getLoc(), LLVM::LLVMPointerType::get(underlyingTy), nullPtr,
            ArrayRef<Value>(oneC));
        auto size =
            initBuilder.create<LLVM::PtrToIntOp>(op.getLoc(), i64Ty, sizeGep);
        // Malloc double the required space to make sure signal
        // shifts do not segfault.
        auto mallocSize =
            initBuilder.create<LLVM::MulOp>(op.getLoc(), i64Ty, size, twoC);
        std::array<Value, 1> margs({mallocSize});
        auto mall =
            initBuilder
                .create<LLVM::CallOp>(op.getLoc(), i8PtrTy,
                                      SymbolRefAttr::get(mallFunc), margs)
                .getResult(0);

        // Store the initial value.
        auto bitcast = initBuilder.create<LLVM::BitcastOp>(
            op.getLoc(), LLVM::LLVMPointerType::get(underlyingTy), mall);

        initBuilder.create<LLVM::StoreOp>(op.getLoc(), initDefCast, bitcast);

        // Get the amount of bytes required to represent an integer underlying
        // type. Use the whole size of the type if not an integer.
        Value passSize;
        if (auto intTy = underlyingTy.dyn_cast<IntegerType>()) {
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
                .getResult(0);

        // Add structured underlying type information.
        if (auto arrayTy = underlyingTy.dyn_cast<LLVM::LLVMArrayType>()) {
          auto zeroC = initBuilder.create<LLVM::ConstantOp>(
              op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));

          auto numElements = initBuilder.create<LLVM::ConstantOp>(
              op.getLoc(), i32Ty,
              rewriter.getI32IntegerAttr(arrayTy.getNumElements()));

          // Get element size.
          auto null = initBuilder.create<LLVM::NullOp>(
              op.getLoc(), LLVM::LLVMPointerType::get(arrayTy));
          auto gepFirst = initBuilder.create<LLVM::GEPOp>(
              op.getLoc(), LLVM::LLVMPointerType::get(arrayTy.getElementType()),
              null, ArrayRef<Value>({zeroC, oneC}));
          auto toInt = initBuilder.create<LLVM::PtrToIntOp>(op.getLoc(), i32Ty,
                                                            gepFirst);

          // Add information to the state.
          initBuilder.create<LLVM::CallOp>(
              op.getLoc(), llvm::None, SymbolRefAttr::get(addSigElemFunc),
              ArrayRef<Value>({initStatePtr, sigIndex, toInt, numElements}));
        } else if (auto structTy =
                       underlyingTy.dyn_cast<LLVM::LLVMStructType>()) {
          auto zeroC = initBuilder.create<LLVM::ConstantOp>(
              op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(0));

          auto null = initBuilder.create<LLVM::NullOp>(
              op.getLoc(), LLVM::LLVMPointerType::get(structTy));
          for (size_t i = 0, e = structTy.getBody().size(); i < e; ++i) {
            auto oneC = initBuilder.create<LLVM::ConstantOp>(
                op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
            auto indexC = initBuilder.create<LLVM::ConstantOp>(
                op.getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));

            // Get pointer offset.
            auto gepElem = initBuilder.create<LLVM::GEPOp>(
                op.getLoc(), LLVM::LLVMPointerType::get(structTy.getBody()[i]),
                null, ArrayRef<Value>({zeroC, indexC}));
            auto elemToInt = initBuilder.create<LLVM::PtrToIntOp>(
                op.getLoc(), i32Ty, gepElem);

            // Get element size.
            auto elemNull = initBuilder.create<LLVM::NullOp>(
                op.getLoc(), LLVM::LLVMPointerType::get(structTy.getBody()[i]));
            auto gepElemSize = initBuilder.create<LLVM::GEPOp>(
                op.getLoc(), LLVM::LLVMPointerType::get(structTy.getBody()[i]),
                elemNull, ArrayRef<Value>({oneC}));
            auto elemSizeToInt = initBuilder.create<LLVM::PtrToIntOp>(
                op.getLoc(), i32Ty, gepElemSize);

            // Add information to the state.
            initBuilder.create<LLVM::CallOp>(
                op.getLoc(), llvm::None, SymbolRefAttr::get(addSigStructFunc),
                ArrayRef<Value>(
                    {initStatePtr, sigIndex, elemToInt, elemSizeToInt}));
          }
        }
      });
    } else if (auto proc = module.lookupSymbol<ProcOp>(instOp.callee())) {
      // Handle process instantiation.
      auto sensesPtrTy = LLVM::LLVMPointerType::get(
          LLVM::LLVMArrayType::get(i1Ty, proc.getNumArguments()));
      auto procStatePtrTy =
          LLVM::LLVMPointerType::get(LLVM::LLVMStructType::getLiteral(
              rewriter.getContext(),
              {i32Ty, i32Ty, sensesPtrTy,
               getProcPersistenceTy(&getDialect(), typeConverter, proc)}));

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
      auto procStateMall = initBuilder
                               .create<LLVM::CallOp>(
                                   op->getLoc(), i8PtrTy,
                                   SymbolRefAttr::get(mallFunc), procStateMArgs)
                               .getResult(0);

      auto procStateBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), procStatePtrTy, procStateMall);

      // Store the initial resume index.
      auto resumeGep = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(i32Ty), procStateBC,
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
      auto sensesMall =
          initBuilder
              .create<LLVM::CallOp>(op->getLoc(), i8PtrTy,
                                    SymbolRefAttr::get(mallFunc), senseMArgs)
              .getResult(0);

      auto sensesBC = initBuilder.create<LLVM::BitcastOp>(
          op->getLoc(), sensesPtrTy, sensesMall);

      // Set all initial senses to 1.
      for (size_t i = 0, e = sensesPtrTy.cast<LLVM::LLVMPointerType>()
                                 .getElementType()
                                 .cast<LLVM::LLVMArrayType>()
                                 .getNumElements();
           i < e; ++i) {
        auto oneB = initBuilder.create<LLVM::ConstantOp>(
            op->getLoc(), i1Ty, rewriter.getBoolAttr(true));
        auto gepInd = initBuilder.create<LLVM::ConstantOp>(
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(i));
        auto senseGep = initBuilder.create<LLVM::GEPOp>(
            op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), sensesBC,
            ArrayRef<Value>({zeroC, gepInd}));
        initBuilder.create<LLVM::StoreOp>(op->getLoc(), oneB, senseGep);
      }

      // Store the senses pointer in the process state.
      auto procStateSensesPtr = initBuilder.create<LLVM::GEPOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(sensesPtrTy), procStateBC,
          ArrayRef<Value>({zeroC, twoC}));
      initBuilder.create<LLVM::StoreOp>(op->getLoc(), sensesBC,
                                        procStateSensesPtr);

      std::array<Value, 3> allocProcArgs({initStatePtr, owner, procStateMall});
      initBuilder.create<LLVM::CallOp>(op->getLoc(), llvm::None,
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
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto sigTy = getLLVMSigType(&getDialect());

    // Get the signal table pointer from the arguments.
    Value sigTablePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(2);

    // Get the index in the signal table and increase counter.
    auto indexConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(sigCounter));
    ++sigCounter;

    // Insert a gep to the signal index in the signal table argument.
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        op, LLVM::LLVMPointerType::get(sigTy), sigTablePtr,
        ArrayRef<Value>(indexConst));

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
                                     transformed.signal());

    if (resTy.isa<IntegerType>()) {
      // Get the amount of bytes to load. An extra byte is always loaded to
      // cover the case where a subsignal spans halfway in the last byte.
      int resWidth = resTy.getIntOrFloatBitWidth();
      int loadWidth = (llvm::divideCeil(resWidth, 8) + 1) * 8;
      auto loadTy = IntegerType::get(rewriter.getContext(), loadWidth);

      auto bitcast = rewriter.create<LLVM::BitcastOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(loadTy), sigDetail[0]);
      auto loadSig =
          rewriter.create<LLVM::LoadOp>(op->getLoc(), loadTy, bitcast);

      // Shift the loaded value by the offset and truncate to the final width.
      auto trOff = adjustBitWidth(op->getLoc(), rewriter, loadTy, sigDetail[1]);
      auto shifted =
          rewriter.create<LLVM::LShrOp>(op->getLoc(), loadTy, loadSig, trOff);
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, finalTy, shifted);

      return success();
    }

    if (resTy.isa<hw::ArrayType, hw::StructType>()) {
      auto bitcast = rewriter.create<LLVM::BitcastOp>(
          op->getLoc(), LLVM::LLVMPointerType::get(finalTy), sigDetail[0]);
      rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, finalTy, bitcast);

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
    auto i8PtrTy = getVoidPtrType();
    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);
    auto sigTy = getLLVMSigType(&getDialect());

    // Get or insert the drive library call.
    auto drvFuncTy = LLVM::LLVMFunctionType::get(
        voidTy, {i8PtrTy, LLVM::LLVMPointerType::get(sigTy), i8PtrTy, i64Ty,
                 i64Ty, i64Ty, i64Ty});
    auto drvFunc = getOrInsertFunction(module, rewriter, op->getLoc(),
                                       "driveSignal", drvFuncTy);

    // Get the state pointer from the function arguments.
    Value statePtr = op->getParentOfType<LLVM::LLVMFuncOp>().getArgument(0);

    // Get signal width.
    Value sigWidth;
    auto underlyingTy = drvOp.value().getType();
    if (isArrayOrStruct(underlyingTy)) {
      auto llvmPtrTy =
          LLVM::LLVMPointerType::get(typeConverter->convertType(underlyingTy));
      auto oneC = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
      auto eightC = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i64Ty, rewriter.getI64IntegerAttr(8));
      auto nullPtr = rewriter.create<LLVM::NullOp>(op->getLoc(), llvmPtrTy);
      auto gepOne = rewriter.create<LLVM::GEPOp>(
          op->getLoc(), llvmPtrTy, nullPtr, ArrayRef<Value>(oneC));
      auto toInt =
          rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), i64Ty, gepOne);
      sigWidth = rewriter.create<LLVM::MulOp>(op->getLoc(), toInt, eightC);
    } else {
      sigWidth = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), i64Ty,
          rewriter.getI64IntegerAttr(underlyingTy.getIntOrFloatBitWidth()));
    }

    // Insert enable comparison. Skip if the enable operand is 0.
    if (auto gate = drvOp.enable()) {
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
          op->getLoc(), LLVM::ICmpPredicate::eq, transformed.enable(), oneC);
      rewriter.create<LLVM::CondBrOp>(op->getLoc(), cmp, drvBlock,
                                      continueBlock);

      rewriter.setInsertionPointToStart(drvBlock);
    }

    Type valTy = typeConverter->convertType(transformed.value().getType());
    Value castVal = typeConverter->materializeTargetConversion(
        rewriter, transformed.value().getLoc(), valTy, transformed.value());

    auto oneConst = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
    auto alloca = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(valTy), oneConst, 4);
    rewriter.create<LLVM::StoreOp>(op->getLoc(), castVal, alloca);
    auto bc = rewriter.create<LLVM::BitcastOp>(op->getLoc(), i8PtrTy, alloca);

    // Get the time values.
    auto realTime = rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), i64Ty, transformed.time(), rewriter.getI32ArrayAttr(0));
    auto delta = rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), i64Ty, transformed.time(), rewriter.getI32ArrayAttr(1));
    auto eps = rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), i64Ty, transformed.time(), rewriter.getI32ArrayAttr(2));

    // Define the driveSignal library call arguments.
    std::array<Value, 7> args(
        {statePtr, transformed.signal(), bc, sigWidth, realTime, delta, eps});
    // Create the library call.
    rewriter.create<LLVM::CallOp>(op->getLoc(), llvm::None,
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

    auto i1Ty = IntegerType::get(rewriter.getContext(), 1);
    auto i32Ty = IntegerType::get(rewriter.getContext(), 32);

    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();

    // Retrieve and update previous trigger values for rising/falling edge
    // detection.
    size_t triggerIndex = 0;
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
            op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(triggerIndex++));
        auto gep = rewriter.create<LLVM::GEPOp>(
            op->getLoc(), LLVM::LLVMPointerType::get(i1Ty), func.getArgument(1),
            ArrayRef<Value>({zeroC, regIndexC, triggerIndexC}));
        prevTriggers.push_back(
            rewriter.create<LLVM::LoadOp>(op->getLoc(), gep));
        rewriter.create<LLVM::StoreOp>(op->getLoc(), transformed.triggers()[i],
                                       gep);
      }
    }

    // Create blocks for drive and continue.
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

private:
  size_t &regCounter;
}; // namespace
} // namespace

//===----------------------------------------------------------------------===//
// Bitwise conversions
//===----------------------------------------------------------------------===//

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
      auto baseWidth = shrOp.getType().getIntOrFloatBitWidth();
      auto hdnWidth = shrOp.hidden().getType().getIntOrFloatBitWidth();
      auto full = baseWidth + hdnWidth;

      auto tmpTy = IntegerType::get(rewriter.getContext(), full);

      // Extend all operands the combined width.
      auto baseZext =
          adjustBitWidth(op->getLoc(), rewriter, tmpTy, transformed.base());
      auto hdnZext =
          adjustBitWidth(op->getLoc(), rewriter, tmpTy, transformed.hidden());
      auto amntZext =
          adjustBitWidth(op->getLoc(), rewriter, tmpTy, transformed.amount());

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
    }
    if (auto arrTy = shrOp.result().getType().dyn_cast<hw::ArrayType>()) {

      auto combined = rewriter.create<hw::ArrayConcatOp>(
          op->getLoc(), ValueRange({shrOp.hidden(), shrOp.base()}));
      rewriter.replaceOpWithNewOp<hw::ArraySliceOp>(op, arrTy, combined,
                                                    transformed.amount());

      return success();
    }

    return failure();
  }
};
} // namespace

namespace {
/// Convert an `llhd.shl` operation to LLVM dialect. All the operands are
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
    auto baseWidth = shlOp.getType().getIntOrFloatBitWidth();
    auto hdnWidth = shlOp.hidden().getType().getIntOrFloatBitWidth();
    auto full = baseWidth + hdnWidth;

    auto tmpTy = IntegerType::get(rewriter.getContext(), full);

    // Extend all operands to the combined width.
    auto baseZext =
        adjustBitWidth(op->getLoc(), rewriter, tmpTy, transformed.base());
    auto hdnZext =
        adjustBitWidth(op->getLoc(), rewriter, tmpTy, transformed.hidden());
    auto amntZext =
        adjustBitWidth(op->getLoc(), rewriter, tmpTy, transformed.amount());

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

namespace {
template <typename SourceOp, typename TargetOp>
class VariadicOpConversion : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using Super = VariadicOpConversion<SourceOp, TargetOp>;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    size_t numOperands = op.getOperands().size();
    // All operands have the same type.
    Type type = op.getOperandTypes().front();
    auto replacement = op.getOperand(0);

    for (unsigned i = 1; i < numOperands; i++) {
      replacement = rewriter.create<TargetOp>(op.getLoc(), type, replacement,
                                              op.getOperand(i));
    }

    rewriter.replaceOp(op, replacement);

    return success();
  }
};

using AndOpConversion = VariadicOpConversion<comb::AndOp, LLVM::AndOp>;
using OrOpConversion = VariadicOpConversion<comb::OrOp, LLVM::OrOp>;
using XorOpConversion = VariadicOpConversion<comb::XorOp, LLVM::XOrOp>;

using CombShlOpConversion =
    OneToOneConvertToLLVMPattern<comb::ShlOp, LLVM::ShlOp>;
using CombShrUOpConversion =
    OneToOneConvertToLLVMPattern<comb::ShrUOp, LLVM::LShrOp>;
using CombShrSOpConversion =
    OneToOneConvertToLLVMPattern<comb::ShrSOp, LLVM::AShrOp>;

} // namespace

namespace {
/// Lower an ArrayConcatOp operation to the LLVM dialect.
/// Pattern: hw.bitcast(input) ==> load(bitcast_ptr(store(input, alloca)))
/// This is necessary because we cannot bitcast aggregate types directly in
/// LLVMIR.
struct BitcastOpConversion : public ConvertOpToLLVMPattern<hw::BitcastOp> {
  using ConvertOpToLLVMPattern<hw::BitcastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type resultTy = typeConverter->convertType(op.result().getType());
    Type inputTy = typeConverter->convertType(op.input().getType());

    Value castInput = getTypeConverter()->materializeTargetConversion(
        rewriter, op.input().getLoc(), inputTy, op.input());
    auto oneC = rewriter.createOrFold<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

    auto ptr = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(inputTy), oneC,
        /*alignment=*/4);

    rewriter.create<LLVM::StoreOp>(op->getLoc(), castInput, ptr);

    auto cast = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), LLVM::LLVMPointerType::get(resultTy), ptr);

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, resultTy, cast);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Arithmetic conversions
//===----------------------------------------------------------------------===//

namespace {

using CombAddOpConversion = VariadicOpConversion<comb::AddOp, LLVM::AddOp>;
using CombMulOpConversion = VariadicOpConversion<comb::MulOp, LLVM::MulOp>;
using CombSubOpConversion =
    OneToOneConvertToLLVMPattern<comb::SubOp, LLVM::SubOp>;

using CombDivUOpConversion =
    OneToOneConvertToLLVMPattern<comb::DivUOp, LLVM::UDivOp>;
using CombDivSOpConversion =
    OneToOneConvertToLLVMPattern<comb::DivSOp, LLVM::SDivOp>;

using CombModUOpConversion =
    OneToOneConvertToLLVMPattern<comb::ModUOp, LLVM::URemOp>;
using CombModSOpConversion =
    OneToOneConvertToLLVMPattern<comb::ModSOp, LLVM::SRemOp>;

using CombICmpOpConversion =
    OneToOneConvertToLLVMPattern<comb::ICmpOp, LLVM::ICmpOp>;

using CombSExtOpConversion =
    OneToOneConvertToLLVMPattern<comb::SExtOp, LLVM::SExtOp>;

// comb.mux supports any type thus this conversion relies on the type converter
// to be able to convert the type of the operands and result to an LLVM_Type
using CombMuxOpConversion =
    OneToOneConvertToLLVMPattern<comb::MuxOp, LLVM::SelectOp>;

} // namespace

namespace {
/// Convert a comb::ParityOp to the LLVM dialect.
struct CombParityOpConversion : public ConvertToLLVMPattern {
  explicit CombParityOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(comb::ParityOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto parityOp = cast<comb::ParityOp>(op);

    auto popCount =
        rewriter.create<LLVM::CtPopOp>(op->getLoc(), parityOp.input());
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(
        op, IntegerType::get(rewriter.getContext(), 1), popCount);

    return success();
  }
};

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
    TimeAttr timeAttr = constOp.valueAttr();
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

namespace {
struct HWConstantOpConversion : public ConvertToLLVMPattern {
  explicit HWConstantOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(hw::ConstantOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operand,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the ConstOp.
    auto constOp = cast<hw::ConstantOp>(op);
    // Get the converted llvm type.
    auto intType = typeConverter->convertType(constOp.valueAttr().getType());
    // Replace the operation with an llvm constant op.
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, intType,
                                                  constOp.valueAttr());

    return success();
  }
};
} // namespace

namespace {
/// Convert an ArrayOp operation to the LLVM dialect. An equivalent and
/// initialized llvm dialect array type is generated.
struct HWArrayCreateOpConversion
    : public ConvertOpToLLVMPattern<hw::ArrayCreateOp> {
  using ConvertOpToLLVMPattern<hw::ArrayCreateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto arrayTy = typeConverter->convertType(op->getResult(0).getType());

    Value arr = rewriter.create<LLVM::UndefOp>(op->getLoc(), arrayTy);
    for (size_t i = 0, e = op.inputs().size(); i < e; ++i) {
      Value input =
          op.inputs()[convertToLLVMEndianess(op.result().getType(), i)];
      Value castInput = typeConverter->materializeTargetConversion(
          rewriter, op->getLoc(), typeConverter->convertType(input.getType()),
          input);

      arr = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), arrayTy, arr, castInput, rewriter.getI32ArrayAttr(i));
    }

    rewriter.replaceOp(op, arr);
    return success();
  }
};
} // namespace

namespace {
/// Convert a StructCreateOp operation to the LLVM dialect. An equivalent and
/// initialized llvm dialect struct type is generated.
struct HWStructCreateOpConversion
    : public ConvertOpToLLVMPattern<hw::StructCreateOp> {
  using ConvertOpToLLVMPattern<hw::StructCreateOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resTy = typeConverter->convertType(op.result().getType());

    Value tup = rewriter.create<LLVM::UndefOp>(op->getLoc(), resTy);
    for (size_t i = 0, e = resTy.cast<LLVM::LLVMStructType>().getBody().size();
         i < e; ++i) {
      Value input =
          op.input()[convertToLLVMEndianess(op.result().getType(), i)];
      Value castInput = typeConverter->materializeTargetConversion(
          rewriter, op->getLoc(), typeConverter->convertType(input.getType()),
          input);
      tup = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), resTy, tup, castInput, rewriter.getI32ArrayAttr(i));
    }

    rewriter.replaceOp(op, tup);
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
    Type inputTy = typeConverter->convertType(op.input().getType());
    Type lowIndexTy = typeConverter->convertType(op.lowIndex().getType());
    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), inputTy, op.input());
    Value castLowIndex = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), lowIndexTy, op.lowIndex());

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

    Type inputTy = typeConverter->convertType(op.input().getType());
    Type lowBitTy = typeConverter->convertType(op.lowBit().getType());
    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), inputTy, op.input());
    Value castLowBit = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), lowBitTy, op.lowBit());

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
    Type inputTy = typeConverter->convertType(op.input().getType());
    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), inputTy, op.input());

    std::vector<Value> sigDetail =
        getSignalDetail(rewriter, &getDialect(), op->getLoc(), castInput,
                        /*extractIndices=*/true);

    uint32_t index = llvmIndexOfStructField(op.getStructType(), op.field());

    auto indexC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(index));

    auto elemPtrTy = LLVM::LLVMPointerType::get(
        llvmStructTy.cast<LLVM::LLVMStructType>().getBody()[index]);
    Value adjusted = shiftStructuredSigPointer(
        op->getLoc(), rewriter, llvmStructTy, elemPtrTy, sigDetail[0], indexC);

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
    Type inputTy = typeConverter->convertType(op.input().getType());
    Type indexTy = typeConverter->convertType(op.index().getType());
    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), inputTy, op.input());
    Value castIndex = typeConverter->materializeTargetConversion(
        rewriter, op->getLoc(), indexTy, op.index());

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

namespace {
/// Convert a comb::ExtractOp to LLVM dialect.
struct CombExtractOpConversion : public ConvertToLLVMPattern {
  explicit CombExtractOpConversion(MLIRContext *ctx,
                                   LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(comb::ExtractOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto extractOp = cast<comb::ExtractOp>(op);
    mlir::Value valueToTrunc = extractOp.input();
    mlir::Type type = extractOp.input().getType();

    if (extractOp.lowBit() != 0) {
      mlir::Value amt = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), type, extractOp.lowBitAttr());
      valueToTrunc = rewriter.create<LLVM::LShrOp>(op->getLoc(), type,
                                                   extractOp.input(), amt);
    }

    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, extractOp.result().getType(),
                                               valueToTrunc);
    return success();
  }
};
} // namespace

namespace {
/// Convert a StructExtractOp to LLVM dialect.
/// Pattern: struct_extract(input, fieldname) =>
///   extractvalue(input, fieldname_to_index(fieldname))
struct StructExtractOpConversion
    : public ConvertOpToLLVMPattern<hw::StructExtractOp> {
  using ConvertOpToLLVMPattern<hw::StructExtractOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type inputTy = typeConverter->convertType(op.input().getType());
    Type resultTy = typeConverter->convertType(op.result().getType());

    uint32_t fieldIndex = llvmIndexOfStructField(
        op.input().getType().cast<hw::StructType>(), op.field());
    IntegerAttr indexAttr = rewriter.getI32IntegerAttr(fieldIndex);

    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op.input().getLoc(), inputTy, op.input());

    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, resultTy, castInput, rewriter.getArrayAttr(indexAttr));

    return success();
  }
};
} // namespace

namespace {
/// Convert an ArrayGetOp to the LLVM dialect.
/// Pattern: array_get(input, index) =>
///   load(gep(store(input, alloca), zext(index)))
struct ArrayGetOpConversion : public ConvertOpToLLVMPattern<hw::ArrayGetOp> {
  using ConvertOpToLLVMPattern<hw::ArrayGetOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto elemTy = typeConverter->convertType(op.result().getType());
    auto inputTy = typeConverter->convertType(op.input().getType());

    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), IntegerType::get(rewriter.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    auto oneC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), IntegerType::get(rewriter.getContext(), 32),
        rewriter.getI32IntegerAttr(1));
    auto arrPtr = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(inputTy), oneC,
        /*alignment=*/4);
    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op.input().getLoc(), inputTy, op.input());

    rewriter.create<LLVM::StoreOp>(op->getLoc(), castInput, arrPtr);
    auto zextIndex = zextByOne(op->getLoc(), rewriter, op.index());
    auto gep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(elemTy), arrPtr,
        ArrayRef<Value>({zeroC, zextIndex}));
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elemTy, gep);

    return success();
  }
};
} // namespace

namespace {
/// Convert an ArraySliceOp to the LLVM dialect.
/// Pattern: array_slice(input, lowIndex) =>
///   load(bitcast(gep(store(input, alloca), zext(lowIndex))))
struct ArraySliceOpConversion
    : public ConvertOpToLLVMPattern<hw::ArraySliceOp> {
  using ConvertOpToLLVMPattern<hw::ArraySliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArraySliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto dstTy = typeConverter->convertType(op.dst().getType());
    auto elemTy = typeConverter->convertType(
        op.dst().getType().cast<hw::ArrayType>().getElementType());
    auto inputTy = typeConverter->convertType(op.input().getType());

    auto zeroC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto oneC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op.input().getLoc(), inputTy, op.input());

    auto arrPtr = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(inputTy), oneC,
        /*alignment=*/4);

    rewriter.create<LLVM::StoreOp>(op->getLoc(), castInput, arrPtr);

    auto zextIndex = zextByOne(op->getLoc(), rewriter, op.lowIndex());

    auto gep = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(elemTy), arrPtr,
        ArrayRef<Value>({zeroC, zextIndex}));

    auto cast = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(dstTy), gep);

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dstTy, cast);

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Insertion operations conversion
//===----------------------------------------------------------------------===//

namespace {
/// Convert a StructInjectOp to LLVM dialect.
/// Pattern: struct_inject(input, index, value) =>
///   insertvalue(input, value, index)
struct StructInjectOpConversion
    : public ConvertOpToLLVMPattern<hw::StructInjectOp> {
  using ConvertOpToLLVMPattern<hw::StructInjectOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::StructInjectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type inputTy = typeConverter->convertType(op.input().getType());
    Type resultTy = typeConverter->convertType(op.result().getType());

    uint32_t fieldIndex = llvmIndexOfStructField(
        op.input().getType().cast<hw::StructType>(), op.fieldAttr().getValue());
    IntegerAttr indexAttr = rewriter.getI32IntegerAttr(fieldIndex);

    Value castInput = typeConverter->materializeTargetConversion(
        rewriter, op.input().getLoc(), inputTy, op.input());

    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, resultTy, castInput, op.newValue(),
        rewriter.getArrayAttr(indexAttr));

    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Concat operations conversion
//===----------------------------------------------------------------------===//

namespace {
/// Lower an ArrayConcatOp operation to the LLVM dialect.
struct ArrayConcatOpConversion
    : public ConvertOpToLLVMPattern<hw::ArrayConcatOp> {
  using ConvertOpToLLVMPattern<hw::ArrayConcatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(hw::ArrayConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    hw::ArrayType arrTy = op.result().getType().cast<hw::ArrayType>();
    Type elemTy = typeConverter->convertType(arrTy.getElementType());
    Type resultTy = typeConverter->convertType(arrTy);

    Value arr = rewriter.create<LLVM::UndefOp>(op->getLoc(), resultTy);

    // Attention: j is hardcoded for little endian machines.
    size_t j = op.inputs().size() - 1, k = 0;

    for (size_t i = 0, e = arrTy.getSize(); i < e; ++i) {
      Type inputTy = typeConverter->convertType(op.inputs()[j].getType());
      Value castInput = typeConverter->materializeTargetConversion(
          rewriter, op.inputs()[j].getLoc(), inputTy, op.inputs()[j]);

      Value element = rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), elemTy, castInput, rewriter.getI32ArrayAttr(k));
      arr = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), resultTy, arr, element, rewriter.getI32ArrayAttr(i));

      ++k;
      if (k >= op.inputs()[j].getType().cast<hw::ArrayType>().getSize()) {
        k = 0;
        --j;
      }
    }

    rewriter.replaceOp(op, arr);
    return success();
  }
};
} // namespace

namespace {
/// Convert a comb::ConcatOp to the LLVM dialect.
struct CombConcatOpConversion : public ConvertToLLVMPattern {
  explicit CombConcatOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(comb::ConcatOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto concatOp = cast<comb::ConcatOp>(op);
    auto numOperands = concatOp->getNumOperands();
    mlir::Type type = concatOp.result().getType();

    unsigned nextInsertion = type.getIntOrFloatBitWidth();
    auto aggregate = rewriter
                         .create<LLVM::ConstantOp>(op->getLoc(), type,
                                                   IntegerAttr::get(type, 0))
                         .res();

    for (unsigned i = 0; i < numOperands; i++) {
      nextInsertion -=
          concatOp->getOperand(i).getType().getIntOrFloatBitWidth();

      auto nextInsValue = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), type, IntegerAttr::get(type, nextInsertion));
      auto extended = rewriter.create<LLVM::ZExtOp>(op->getLoc(), type,
                                                    concatOp->getOperand(i));
      auto shifted = rewriter.create<LLVM::ShlOp>(op->getLoc(), type, extended,
                                                  nextInsValue);
      aggregate =
          rewriter.create<LLVM::OrOp>(op->getLoc(), type, aggregate, shifted)
              .res();
    }

    rewriter.replaceOp(op, aggregate);
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
    Type initTy = typeConverter->convertType(transformed.init().getType());

    auto oneC = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), i32Ty, rewriter.getI32IntegerAttr(1));
    auto alloca = rewriter.create<LLVM::AllocaOp>(
        op->getLoc(), LLVM::LLVMPointerType::get(initTy), oneC, 4);
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

void circt::populateLLHDToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 size_t &sigCounter,
                                                 size_t &regCounter) {
  MLIRContext *ctx = converter.getDialect()->getContext();

  // Value creation conversion patterns.
  patterns.add<ConstantTimeOpConversion, HWConstantOpConversion>(ctx,
                                                                 converter);
  patterns.add<HWArrayCreateOpConversion, HWStructCreateOpConversion>(
      converter);

  // Extract conversion patterns.
  patterns.add<SigExtractOpConversion, SigArraySliceOpConversion,
               SigArrayGetOpConversion, SigStructExtractOpConversion>(
      converter);

  patterns.add<CombExtractOpConversion, CombConcatOpConversion>(ctx, converter);

  // Bitwise conversion patterns.
  patterns.add<ShrOpConversion, ShlOpConversion, CombParityOpConversion>(
      ctx, converter);
  patterns.add<AndOpConversion, OrOpConversion, XorOpConversion>(converter);
  patterns.add<CombShlOpConversion, CombShrUOpConversion, CombShrSOpConversion,
               BitcastOpConversion>(converter);

  // Arithmetic conversion patterns.
  patterns.add<CombAddOpConversion, CombSubOpConversion, CombMulOpConversion,
               CombDivUOpConversion, CombDivSOpConversion, CombModUOpConversion,
               CombModSOpConversion, CombICmpOpConversion, CombSExtOpConversion,
               CombMuxOpConversion>(converter);

  // Unit conversion patterns.
  patterns.add<TerminatorOpConversion, ProcOpConversion, WaitOpConversion,
               HaltOpConversion>(ctx, converter);
  patterns.add<EntityOpConversion>(ctx, converter, sigCounter, regCounter);

  // Signal conversion patterns.
  patterns.add<PrbOpConversion, DrvOpConversion>(ctx, converter);
  patterns.add<SigOpConversion>(ctx, converter, sigCounter);
  patterns.add<RegOpConversion>(ctx, converter, regCounter);

  // Memory conversion patterns.
  patterns.add<VarOpConversion, StoreOpConversion>(ctx, converter);
  patterns.add<LoadOpConversion>(converter);

  patterns.add<ArrayGetOpConversion, ArraySliceOpConversion,
               ArrayConcatOpConversion, StructExtractOpConversion,
               StructInjectOpConversion>(converter);
}

void LLHDToLLVMLoweringPass::runOnOperation() {
  // Keep a counter to infer a signal's index in his entity's signal table.
  size_t sigCounter = 0;

  // Keep a counter to infer a reg's index in his entity.
  size_t regCounter = 0;

  RewritePatternSet patterns(&getContext());
  auto converter = mlir::LLVMTypeConverter(&getContext());
  converter.addConversion(
      [&](SigType sig) { return convertSigType(sig, converter); });
  converter.addConversion(
      [&](TimeType time) { return convertTimeType(time, converter); });
  converter.addConversion(
      [&](PtrType ptr) { return convertPtrType(ptr, converter); });
  converter.addConversion(
      [&](hw::ArrayType arr) { return convertArrayType(arr, converter); });
  converter.addConversion(
      [&](hw::StructType tup) { return convertStructType(tup, converter); });

  // Apply a partial conversion first, lowering only the instances, to generate
  // the init function.
  patterns.add<InstOpConversion>(&getContext(), converter);

  LLVMConversionTarget target(getContext());
  target.addIllegalOp<InstOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Apply the partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
  patterns.clear();

  // Setup the full conversion.
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateLLHDToLLVMConversionPatterns(converter, patterns, sigCounter,
                                       regCounter);

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
