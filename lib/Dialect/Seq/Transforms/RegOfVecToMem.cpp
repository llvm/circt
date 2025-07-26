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
#include "llvm/ADT/DenseMap.h"
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

struct MemoryPattern {
  FirRegOp memReg;               // The register array representing memory
  FirRegOp outputReg;            // Optional output register
  Value clock;                   // Clock signal
  Value readAddr;                // Read address
  Value writeAddr;               // Write address
  Value writeData;               // Write data
  Value writeEnable;             // Write enable
  Value readEnable;              // Read enable (optional)
  comb::MuxOp writeMux;          // Mux selecting between old/new memory state
  comb::MuxOp readMux;           // Mux for read data
  hw::ArrayGetOp readAccess;     // Array read operation
  hw::ArrayInjectOp writeAccess; // Array write operation
};

class RegOfVecToMemPass : public impl::RegOfVecToMemBase<RegOfVecToMemPass> {
public:
  void runOnOperation() override;

private:
  bool analyzeMemoryPattern(FirRegOp reg, MemoryPattern &pattern);
  bool createFirMemory(MemoryPattern &pattern);
  bool isArrayType(Type type);
  std::optional<std::pair<uint64_t, uint64_t>> getArrayDimensions(Type type);

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

  ArrayGetOp readAccess;
  ArrayInjectOp writeAccess;
  comb::MuxOp writeMux;
  for (auto *user : reg.getResult().getUsers()) {
    LLVM_DEBUG(llvm::dbgs() << "  Register user: " << *user << "\n");
    if (auto arrayGet = dyn_cast<hw::ArrayGetOp>(user); !readAccess && arrayGet)
      readAccess = arrayGet;
    else if (auto arrayInject = dyn_cast<hw::ArrayInjectOp>(user);
             !writeAccess && arrayInject)
      writeAccess = arrayInject;
    else if (auto mux = dyn_cast<comb::MuxOp>(user); !writeMux && mux)
      writeMux = mux;
    else
      return false;
  }
  if (!readAccess || !writeAccess || !writeMux)
    return false;

  pattern.memReg = reg;
  pattern.clock = reg.getClk();

  // Find the mux that drives this register
  auto nextValue = reg.getNext();
  auto mux = nextValue.getDefiningOp<comb::MuxOp>();
  if (!mux)
    return false;

  LLVM_DEBUG(llvm::dbgs() << "  Found driving mux: " << mux << "\n");
  pattern.writeMux = mux;

  // Check that the mux is only used by this register (safety check)
  if (!mux.getResult().hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs() << "  Mux has multiple uses, cannot transform\n");
    return false;
  }

  // Analyze mux inputs: sel ? write_result : current_memory
  Value writeResult = mux.getTrueValue();
  Value currentMemory = mux.getFalseValue();

  // Check if false value is the current register (feedback)
  if (currentMemory != reg.getResult())
    return false;

  // Look for array_inject operation in write path
  auto arrayInject = writeResult.getDefiningOp<hw::ArrayInjectOp>();
  if (!arrayInject)
    return false;

  LLVM_DEBUG(llvm::dbgs() << "  Found array_inject: " << arrayInject << "\n");
  pattern.writeAccess = arrayInject;
  pattern.writeAddr = arrayInject.getIndex();
  pattern.writeData = arrayInject.getElement();
  pattern.writeEnable = mux.getCond();

  // Look for read pattern - find array_get users
  auto arrayGet = readAccess;
  LLVM_DEBUG(llvm::dbgs() << "  Found array_get: " << arrayGet << "\n");
  pattern.readAccess = arrayGet;
  pattern.readAddr = arrayGet.getIndex();

  // Check if read goes through output register
  for (auto *readUser : arrayGet.getResult().getUsers()) {
    if (auto outputReg = dyn_cast<FirRegOp>(readUser)) {
      if (outputReg.getClk() == pattern.clock) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  Found output register: " << outputReg << "\n");
        pattern.outputReg = outputReg;
        break;
      }
    }
  }

  bool success = pattern.readAccess != nullptr;
  LLVM_DEBUG(llvm::dbgs() << "  Pattern analysis "
                          << (success ? "succeeded" : "failed") << "\n");
  return success;
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
  auto memType =
      FirMemType::get(builder.getContext(), depth, width, /*maskWidth=*/1);
  auto firMem = seq::FirMemOp::create(
      builder, memType, /*readLatency=*/0, /*writeLatency=*/1,
      /*readUnderWrite=*/seq::RUW::Undefined,
      /*writeUnderWrite=*/seq::WUW::Undefined,
      /*name=*/builder.getStringAttr("mem"), /*innerSym=*/hw::InnerSymAttr{},
      /*init=*/seq::FirMemInitAttr{}, /*prefix=*/StringAttr{},
      /*outputFile=*/Attribute{});

  // Create read port
  Value readData = FirMemReadOp::create(
      builder, firMem, pattern.readAddr, pattern.clock,
      /*enable=*/hw::ConstantOp::create(builder, builder.getI1Type(), 1));

  LLVM_DEBUG(llvm::dbgs() << "  Created read port\n"
                          << firMem << "\n " << readData);

  Value mask;
  // Create write port
  FirMemWriteOp::create(builder, firMem, pattern.writeAddr, pattern.clock,
                        pattern.writeEnable, pattern.writeData, mask);

  LLVM_DEBUG(llvm::dbgs() << "  Created write port\n");

  // Replace read access
  if (pattern.outputReg)
    // If there's an output register, replace its input
    pattern.outputReg.getNext().replaceAllUsesWith(readData);
  else
    // Replace direct read access
    pattern.readAccess.getResult().replaceAllUsesWith(readData);

  // Mark old operations for removal
  opsToErase.push_back(pattern.memReg);
  if (pattern.readAccess)
    opsToErase.push_back(pattern.readAccess);
  if (pattern.writeAccess)
    opsToErase.push_back(pattern.writeAccess);
  if (pattern.writeMux)
    opsToErase.push_back(pattern.writeMux);

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

  // Erase all marked operations
  for (auto *op : opsToErase) {
    LLVM_DEBUG(llvm::dbgs()
               << "Erasing operation: " << *op << " number of uses:"
               << "\n");
    op->dropAllUses();
    op->erase();
  }
  opsToErase.clear();
}
