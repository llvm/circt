//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_MEMORYALLOCATIONPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::rtg;

//===----------------------------------------------------------------------===//
// Memory Allocation Pass
//===----------------------------------------------------------------------===//

namespace {
struct AllocationInfo {
  APInt nextFree;
  APInt maxAddr;
};

/// Helper function to adjust APInt width and check for truncation errors.
LogicalResult adjustAPIntWidth(APInt &value, unsigned targetBitWidth,
                               Location loc) {
  if (value.getBitWidth() > targetBitWidth && !value.isIntN(targetBitWidth))
    return mlir::emitError(
        loc, "cannot truncate APInt because value is too big to fit");

  if (value.getBitWidth() < targetBitWidth) {
    value = value.zext(targetBitWidth);
    return success();
  }

  value = value.trunc(targetBitWidth);
  return success();
}

struct MemoryAllocationPass
    : public rtg::impl::MemoryAllocationPassBase<MemoryAllocationPass> {
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void MemoryAllocationPass::runOnOperation() {
  auto testOp = getOperation();
  DenseMap<Value, AllocationInfo> nextFreeMap;

  if (!useImmediates) {
    testOp->emitError("label mode not yet supported");
    return signalPassFailure();
  }

  // Collect memory block declarations in target.
  auto target = testOp.getTargetAttr();
  if (!target)
    return;

  SymbolTable table(testOp->getParentOfType<ModuleOp>());
  auto targetOp = table.lookupNearestSymbolFrom<TargetOp>(testOp, target);

  for (auto &op : *targetOp.getBody()) {
    auto memBlock = dyn_cast<MemoryBlockDeclareOp>(&op);
    if (!memBlock)
      continue;

    auto &slot = nextFreeMap[memBlock.getResult()];
    slot.nextFree = memBlock.getBaseAddress();
    slot.maxAddr = memBlock.getEndAddress();
  }

  // Propagate memory block declarations from target to test.
  auto targetYields = targetOp.getBody()->getTerminator()->getOperands();
  auto targetEntries = targetOp.getTarget().getEntries();
  auto testEntries = testOp.getTargetType().getEntries();
  auto testArgs = testOp.getBody()->getArguments();

  size_t targetIdx = 0;
  for (auto [testEntry, testArg] : llvm::zip(testEntries, testArgs)) {
    while (targetIdx < targetEntries.size() &&
           targetEntries[targetIdx].name.getValue() < testEntry.name.getValue())
      targetIdx++;

    if (targetIdx < targetEntries.size() &&
        targetEntries[targetIdx].name.getValue() == testEntry.name.getValue()) {
      auto targetYield = targetYields[targetIdx];
      auto it = nextFreeMap.find(targetYield);
      if (it != nextFreeMap.end())
        nextFreeMap[testArg] = it->second;
    }
  }

  // Iterate through the test and allocate memory for each 'memory_alloc'
  // operation.
  for (auto &op : llvm::make_early_inc_range(*testOp.getBody())) {
    auto mem = dyn_cast<MemoryAllocOp>(&op);
    if (!mem)
      continue;

    auto iter = nextFreeMap.find(mem.getMemoryBlock());
    if (iter == nextFreeMap.end()) {
      mem->emitError("memory block not found");
      return signalPassFailure();
    }

    APInt size;
    if (!matchPattern(mem.getSize(), m_ConstantInt(&size))) {
      mem->emitError("could not determine memory allocation size");
      return signalPassFailure();
    }

    APInt alignment;
    if (!matchPattern(mem.getAlignment(), m_ConstantInt(&alignment))) {
      mem->emitError("could not determine memory allocation alignment");
      return signalPassFailure();
    }

    if (size.isZero()) {
      mem->emitError(
          "memory allocation size must be greater than zero (was 0)");
      return signalPassFailure();
    }

    if (!alignment.isPowerOf2()) {
      mem->emitError("memory allocation alignment must be a power of two (was ")
          << alignment.getZExtValue() << ")";
      return signalPassFailure();
    }

    auto &memBlock = iter->getSecond();
    APInt nextFree = memBlock.nextFree;
    unsigned bitWidth = nextFree.getBitWidth();

    if (failed(adjustAPIntWidth(size, bitWidth, mem.getLoc())) ||
        failed(adjustAPIntWidth(alignment, bitWidth, mem.getLoc())))
      return signalPassFailure();

    // Calculate aligned address
    APInt bias(bitWidth, !nextFree.isZero());
    APInt ceilDiv = (nextFree - bias).udiv(alignment) + bias;
    APInt nextFreeAligned = ceilDiv * alignment;

    memBlock.nextFree = nextFreeAligned + size;
    if (memBlock.nextFree.ugt(memBlock.maxAddr)) {
      mem->emitError("memory block not large enough to fit all allocations");
      return signalPassFailure();
    }

    ++numMemoriesAllocated;

    IRRewriter builder(mem);
    builder.replaceOpWithNewOp<ConstantOp>(
        mem, ImmediateAttr::get(builder.getContext(), nextFreeAligned));
  }
}
