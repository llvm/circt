//===- RegOfVecToMem.cpp - Convert Register Arrays to Memories -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts register arrays that follow memory access
// patterns to seq.firmem operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "reg-of-vec-to-mem"

using namespace circt;
using namespace seq;
using namespace hw;

namespace circt {
namespace seq {
#define GEN_PASS_DEF_REGOFVECTOMEM
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

namespace {

struct WriteDataPiece {
  unsigned lowBit;
  Value value;
};

struct WriteLane {
  unsigned lowBit;
  unsigned width;
  Value condition;
  SmallVector<WriteDataPiece> dataPieces;
};

struct MemoryPattern {
  FirRegOp memReg;           // The register array representing memory
  Value clock;               // Clock signal
  Value readAddr;            // Read address
  Value writeAddr;           // Write address
  Value writeEnable;         // Write enable
  Value writeCondition;      // Condition for an unmasked write
  Value writeData;           // Full-word write data
  hw::ArrayGetOp readAccess; // Optional array read operation
  SmallVector<WriteLane> writeLanes;
  SmallVector<Operation *> opsToErase;
  unsigned maskWidth = 1;
};

class RegOfVecToMemPass : public impl::RegOfVecToMemBase<RegOfVecToMemPass> {
public:
  void runOnOperation() override;

private:
  bool analyzeMemoryPattern(FirRegOp reg, MemoryPattern &pattern);
  bool analyzeUpdatedWord(Value value, Value oldWord, Value memValue,
                          unsigned lowBit,
                          SmallVectorImpl<WriteDataPiece> &dataPieces,
                          llvm::SmallPtrSetImpl<Operation *> &matchedOps);
  bool matchWriteChain(Value value, Value memValue, MemoryPattern &pattern,
                       llvm::SmallPtrSetImpl<Operation *> &matchedOps);
  bool createFirMemory(MemoryPattern &pattern);
  bool isArrayType(Type type);
  std::optional<std::pair<uint64_t, uint64_t>> getArrayDimensions(Type type);
  bool valueDependsOn(Value value, Value dependency);

  SmallVector<Operation *> opsToErase;
};

} // end anonymous namespace

bool RegOfVecToMemPass::isArrayType(Type type) {
  return isa<hw::ArrayType, hw::UnpackedArrayType>(type);
}

std::optional<std::pair<uint64_t, uint64_t>>
RegOfVecToMemPass::getArrayDimensions(Type type) {
  if (auto arrayType = dyn_cast<hw::ArrayType>(type)) {
    auto elemType = arrayType.getElementType();
    if (auto intType = dyn_cast<IntegerType>(elemType)) {
      return std::make_pair(arrayType.getNumElements(), intType.getWidth());
    }
  }
  return std::nullopt;
}

bool RegOfVecToMemPass::analyzeMemoryPattern(FirRegOp reg,
                                             MemoryPattern &pattern) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing register: " << reg << "\n");

  // Check if register has array type
  if (!isArrayType(reg.getType()))
    return false;

  auto dims = getArrayDimensions(reg.getType());
  if (!dims)
    return false;
  unsigned wordWidth = dims->second;

  pattern.memReg = reg;
  pattern.clock = reg.getClk();

  llvm::SmallPtrSet<Operation *, 32> matchedOps;
  Value nextValue = reg.getNext();
  if (!matchWriteChain(nextValue, reg.getResult(), pattern, matchedOps)) {
    pattern = MemoryPattern{};
    pattern.memReg = reg;
    pattern.clock = reg.getClk();
    matchedOps.clear();

    // Slang emits an additional mux around a chain of masked writes. Treat
    // this as a global write enable if its false value is the memory itself.
    auto enableMux = nextValue.getDefiningOp<comb::MuxOp>();
    if (!enableMux || enableMux.getFalseValue() != reg.getResult() ||
        !matchWriteChain(enableMux.getTrueValue(), reg.getResult(), pattern,
                         matchedOps))
      return false;
    pattern.writeEnable = enableMux.getCond();
    matchedOps.insert(enableMux);
  }

  if (pattern.writeLanes.empty())
    return false;

  // A single full-word lane is the original unmasked memory pattern.
  if (pattern.writeLanes.size() == 1 &&
      pattern.writeLanes.front().lowBit == 0 &&
      pattern.writeLanes.front().width == wordWidth) {
    auto &lane = pattern.writeLanes.front();
    if (lane.dataPieces.size() != 1 || lane.dataPieces.front().lowBit != 0)
      return false;
    pattern.writeData = lane.dataPieces.front().value;
    pattern.writeCondition = lane.condition;
    pattern.writeLanes.clear();
  } else {
    unsigned laneWidth = pattern.writeLanes.front().width;
    if (laneWidth == 0 || wordWidth % laneWidth != 0)
      return false;
    pattern.maskWidth = wordWidth / laneWidth;

    llvm::DenseSet<unsigned> occupiedLanes;
    for (auto &lane : pattern.writeLanes) {
      if (lane.width != laneWidth || lane.lowBit % laneWidth != 0 ||
          lane.lowBit + lane.width > wordWidth ||
          !occupiedLanes.insert(lane.lowBit / laneWidth).second)
        return false;
    }
  }

  // Users which are not part of the write chain may only be a single read
  // access. A read is optional, which also allows write-only memories.
  for (auto *user : reg.getResult().getUsers()) {
    LLVM_DEBUG(llvm::dbgs() << "  Register user: " << *user << "\n");
    if (matchedOps.contains(user)) {
      // An array_get used to preserve the first updated word may also be the
      // externally visible read when the read and write addresses are equal.
      auto arrayGet = dyn_cast<hw::ArrayGetOp>(user);
      if (arrayGet &&
          llvm::any_of(arrayGet.getResult().getUses(), [&](auto &use) {
            return !matchedOps.contains(use.getOwner());
          })) {
        if (pattern.readAccess)
          return false;
        pattern.readAccess = arrayGet;
        pattern.readAddr = arrayGet.getIndex();
      }
      continue;
    }
    auto arrayGet = dyn_cast<hw::ArrayGetOp>(user);
    if (!arrayGet || pattern.readAccess)
      return false;
    pattern.readAccess = arrayGet;
    pattern.readAddr = arrayGet.getIndex();
    matchedOps.insert(arrayGet);
  }

  matchedOps.insert(reg);

  // Except for the external read result, the matched graph must be closed.
  // This makes it safe to break the register feedback cycle during cleanup.
  for (auto *op : matchedOps) {
    if (pattern.readAccess && op == pattern.readAccess.getOperation())
      continue;
    for (Value result : op->getResults())
      for (auto &use : result.getUses())
        if (!matchedOps.contains(use.getOwner()))
          return false;
  }
  pattern.opsToErase.assign(matchedOps.begin(), matchedOps.end());

  bool success = pattern.writeData || !pattern.writeLanes.empty();
  LLVM_DEBUG(llvm::dbgs() << "  Pattern analysis "
                          << (success ? "succeeded" : "failed") << "\n");
  return success;
}

bool RegOfVecToMemPass::valueDependsOn(Value value, Value dependency) {
  SmallVector<Value> worklist{value};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (current == dependency)
      return true;
    if (!visited.insert(current).second)
      continue;
    if (auto *definingOp = current.getDefiningOp())
      llvm::append_range(worklist, definingOp->getOperands());
  }
  return false;
}

bool RegOfVecToMemPass::analyzeUpdatedWord(
    Value value, Value oldWord, Value memValue, unsigned lowBit,
    SmallVectorImpl<WriteDataPiece> &dataPieces,
    llvm::SmallPtrSetImpl<Operation *> &matchedOps) {
  unsigned width = cast<IntegerType>(value.getType()).getWidth();

  if (value == oldWord)
    return lowBit == 0 &&
           width == cast<IntegerType>(oldWord.getType()).getWidth();

  if (auto extract = value.getDefiningOp<comb::ExtractOp>()) {
    if (extract.getInput() == oldWord && extract.getLowBit() == lowBit) {
      matchedOps.insert(extract);
      return true;
    }
  }

  if (auto concat = value.getDefiningOp<comb::ConcatOp>()) {
    unsigned operandLowBit = lowBit + width;
    for (Value input : concat.getInputs()) {
      unsigned inputWidth = cast<IntegerType>(input.getType()).getWidth();
      operandLowBit -= inputWidth;
      if (!analyzeUpdatedWord(input, oldWord, memValue, operandLowBit,
                              dataPieces, matchedOps))
        return false;
    }
    matchedOps.insert(concat);
    return true;
  }

  // The replacement data must be independent of the memory being converted.
  if (valueDependsOn(value, memValue))
    return false;
  dataPieces.push_back({lowBit, value});
  return true;
}

bool RegOfVecToMemPass::matchWriteChain(
    Value value, Value memValue, MemoryPattern &pattern,
    llvm::SmallPtrSetImpl<Operation *> &matchedOps) {
  if (value == memValue)
    return true;

  auto mux = value.getDefiningOp<comb::MuxOp>();
  if (!mux)
    return false;
  Value base = mux.getFalseValue();
  auto inject = mux.getTrueValue().getDefiningOp<hw::ArrayInjectOp>();
  if (!inject || inject.getInput() != base)
    return false;

  if (!matchWriteChain(base, memValue, pattern, matchedOps))
    return false;

  if (pattern.writeAddr && pattern.writeAddr != inject.getIndex())
    return false;
  pattern.writeAddr = inject.getIndex();

  WriteLane lane;
  lane.condition = mux.getCond();
  unsigned wordWidth =
      cast<IntegerType>(inject.getElement().getType()).getWidth();

  if (!valueDependsOn(inject.getElement(), memValue)) {
    lane.lowBit = 0;
    lane.width = wordWidth;
    lane.dataPieces.push_back({0, inject.getElement()});
  } else {
    // Find the word read from the exact array value updated by this chain
    // element. This read provides all preserved slices of the reconstructed
    // word.
    SmallVector<hw::ArrayGetOp> arrayGets;
    SmallVector<Value> worklist{inject.getElement()};
    llvm::SmallPtrSet<Operation *, 16> visited;
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      auto *definingOp = current.getDefiningOp();
      if (!definingOp || !visited.insert(definingOp).second)
        continue;
      if (auto arrayGet = dyn_cast<hw::ArrayGetOp>(definingOp)) {
        if (arrayGet.getInput() == base &&
            arrayGet.getIndex() == inject.getIndex())
          arrayGets.push_back(arrayGet);
        continue;
      }
      llvm::append_range(worklist, definingOp->getOperands());
    }
    if (arrayGets.size() != 1)
      return false;

    auto oldWord = arrayGets.front();
    if (!analyzeUpdatedWord(inject.getElement(), oldWord, memValue, 0,
                            lane.dataPieces, matchedOps) ||
        lane.dataPieces.empty())
      return false;

    llvm::sort(lane.dataPieces,
               [](const WriteDataPiece &lhs, const WriteDataPiece &rhs) {
                 return lhs.lowBit < rhs.lowBit;
               });
    lane.lowBit = lane.dataPieces.front().lowBit;
    unsigned nextBit = lane.lowBit;
    for (auto piece : lane.dataPieces) {
      if (piece.lowBit != nextBit)
        return false;
      nextBit += cast<IntegerType>(piece.value.getType()).getWidth();
    }
    lane.width = nextBit - lane.lowBit;
    matchedOps.insert(oldWord);
  }

  pattern.writeLanes.push_back(std::move(lane));
  matchedOps.insert(inject);
  matchedOps.insert(mux);
  return true;
}

bool RegOfVecToMemPass::createFirMemory(MemoryPattern &pattern) {
  LLVM_DEBUG(llvm::dbgs() << "Creating FirMemory for pattern\n");

  auto dims = getArrayDimensions(pattern.memReg.getType());
  if (!dims)
    return false;

  uint64_t depth = dims->first;
  uint64_t width = dims->second;

  LLVM_DEBUG(llvm::dbgs() << "  Memory dimensions: " << depth << " x " << width
                          << "\n");

  ImplicitLocOpBuilder builder(pattern.memReg.getLoc(), pattern.memReg);

  // Create FirMem
  auto memType = FirMemType::get(builder.getContext(), depth, width,
                                 /*maskWidth=*/pattern.maskWidth);
  auto firMem = seq::FirMemOp::create(
      builder, memType, /*readLatency=*/0, /*writeLatency=*/1,
      /*readUnderWrite=*/seq::RUW::Undefined,
      /*writeUnderWrite=*/seq::WUW::Undefined,
      /*name=*/builder.getStringAttr("mem"), /*innerSym=*/hw::InnerSymAttr{},
      /*init=*/seq::FirMemInitAttr{}, /*prefix=*/StringAttr{},
      /*outputFile=*/Attribute{});

  // FIRRTL currently uses a 1-bit address for a single element memory,
  // however HW arrays use 0-bit addresses. To bridge this gap, create a 1-bit
  // address equal to 0 if our address is 0-bit.
  auto fixZeroWidthAddr = [&](Value addr) -> Value {
    if (addr.getType().getIntOrFloatBitWidth() == 0) {
      return hw::ConstantOp::create(builder,
                                    mlir::IntegerType::get(&getContext(), 1), 0)
          .getResult();
    }
    return addr;
  };

  if (pattern.readAccess) {
    auto readAddr = fixZeroWidthAddr(pattern.readAddr);
    Value readData = FirMemReadOp::create(
        builder, firMem, readAddr, pattern.clock,
        /*enable=*/hw::ConstantOp::create(builder, builder.getI1Type(), 1));
    pattern.readAccess.getResult().replaceAllUsesWith(readData);
    LLVM_DEBUG(llvm::dbgs() << "  Created read port\n"
                            << firMem << "\n " << readData);
  }

  Value writeData = pattern.writeData;
  Value mask;
  if (!pattern.writeLanes.empty()) {
    unsigned laneWidth = width / pattern.maskWidth;
    SmallVector<Value> laneData(pattern.maskWidth);
    SmallVector<Value> laneMask(pattern.maskWidth);
    for (auto &lane : pattern.writeLanes) {
      unsigned laneIndex = lane.lowBit / laneWidth;
      SmallVector<Value> pieces;
      for (auto piece : llvm::reverse(lane.dataPieces))
        pieces.push_back(piece.value);
      laneData[laneIndex] = pieces.size() == 1
                                ? pieces.front()
                                : comb::ConcatOp::create(builder, pieces);
      laneMask[laneIndex] = lane.condition;
    }

    for (unsigned i = 0; i < pattern.maskWidth; ++i) {
      if (!laneData[i])
        laneData[i] = hw::ConstantOp::create(
            builder, IntegerType::get(builder.getContext(), laneWidth), 0);
      if (!laneMask[i])
        laneMask[i] =
            hw::ConstantOp::create(builder, builder.getI1Type(), false);
    }

    SmallVector<Value> concatData(llvm::reverse(laneData));
    SmallVector<Value> concatMask(llvm::reverse(laneMask));
    writeData = comb::ConcatOp::create(builder, concatData);
    mask = comb::ConcatOp::create(builder, concatMask);
    if (!pattern.writeEnable)
      pattern.writeEnable =
          hw::ConstantOp::create(builder, builder.getI1Type(), true);
  } else if (pattern.writeEnable) {
    pattern.writeEnable = comb::AndOp::create(
        builder, ValueRange{pattern.writeEnable, pattern.writeCondition},
        /*twoState=*/false);
  } else {
    pattern.writeEnable = pattern.writeCondition;
  }

  // Create write port
  auto writeAddr = fixZeroWidthAddr(pattern.writeAddr);
  FirMemWriteOp::create(builder, firMem, writeAddr, pattern.clock,
                        pattern.writeEnable, writeData, mask);

  LLVM_DEBUG(llvm::dbgs() << "  Created write port\n");

  llvm::append_range(opsToErase, pattern.opsToErase);

  return true;
}

void RegOfVecToMemPass::runOnOperation() {
  auto module = getOperation();

  SmallVector<FirRegOp> arrayRegs;

  // Collect all FirRegOp with array types
  module.walk([&](FirRegOp reg) {
    if (isArrayType(reg.getType())) {
      arrayRegs.push_back(reg);
    }
  });

  // Analyze each array register for memory patterns
  for (auto reg : arrayRegs) {
    MemoryPattern pattern;
    if (analyzeMemoryPattern(reg, pattern)) {
      createFirMemory(pattern);
    }
  }

  // Break the closed feedback graphs, then erase all matched operations.
  for (auto *op : opsToErase)
    op->dropAllUses();
  for (auto *op : llvm::reverse(opsToErase)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Erasing operation: " << *op << " number of uses:"
               << "\n");
    op->erase();
  }
  opsToErase.clear();
}
