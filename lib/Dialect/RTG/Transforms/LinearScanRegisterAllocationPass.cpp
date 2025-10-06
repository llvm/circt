//===- LinearScanRegisterAllocationPass.cpp - Register Allocation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass allocates registers using a simple linear scan algorithm.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTG/IR/RTGAttributes.h"
#include "circt/Dialect/RTG/IR/RTGISAAssemblyOpInterfaces.h"
#include "circt/Dialect/RTG/IR/RTGOps.h"
#include "circt/Dialect/RTG/Transforms/RTGPasses.h"
#include "circt/Support/UnusedOpPruner.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_LINEARSCANREGISTERALLOCATIONPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;

#define DEBUG_TYPE "rtg-linear-scan-register-allocation"


// Follow all users of a value transitively until reaching an operation that
// implements RegisterAllocationOpInterface. Store the operand values at those
// interface operations in the output vector.
static LogicalResult collectTransitiveRegisterAllocationOperands(
    Value value, rtg::RegisterAttrInterface reg, SetVector<std::pair<Value, rtg::RegisterAttrInterface>> &operands) {
  LLVM_DEBUG(llvm::dbgs() << "Collecting transitive register allocation operands for value: "
                          << value << " with register: " << reg << "\n");

  SmallVector<std::pair<Value, Attribute>> worklist;
  DenseMap<Value, Attribute> visited;
  worklist.push_back({value, reg});
  visited.insert({value, reg});

  while (!worklist.empty()) {
    auto [current, currentAttr] = worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs() << "  Processing current value: " << current
                            << " with attribute: " << currentAttr << "\n");
    
    for (Operation *user : current.getUsers()) {
      if (auto regAllocOp = dyn_cast<rtg::RegisterAllocationOpInterface>(user)) {
        // Found a RegisterAllocationOpInterface - store the operand and don't
        // follow results further
        LLVM_DEBUG(llvm::dbgs() << "    Found RegisterAllocationOpInterface: "
                                << *user << "\n");
        operands.insert({current, cast<rtg::RegisterAttrInterface>(currentAttr)});
      } else {
        // Not a RegisterAllocationOpInterface - continue following results
        LLVM_DEBUG(llvm::dbgs() << "    Following non-RegisterAllocationOpInterface: "
                                << *user << "\n");
        
        // Check that all additional operands are provided by constant-like operations
        SmallVector<Attribute> operandAttrs;
        for (auto operand : user->getOperands()) {
          if (operand == current) {
            operandAttrs.push_back(currentAttr);
            continue;
          }
          auto *defOp = operand.getDefiningOp();
          if (!defOp || !defOp->hasTrait<OpTrait::ConstantLike>())
            return operand.getDefiningOp()->emitError(
                "operand not provided by a constant-like operation");
          
          SmallVector<OpFoldResult> operandFoldResults;
          if (failed(operand.getDefiningOp()->fold(operandFoldResults)))
            return operand.getDefiningOp()->emitError(
                "folding constant like operation failed???");

          operandAttrs.push_back(cast<Attribute>(operandFoldResults[0]));
        }

        // Fold the operation to get the result attribute
        SmallVector<OpFoldResult> foldResults;
        if (failed(user->fold(operandAttrs, foldResults))) {
          LLVM_DEBUG(llvm::dbgs() << "    Failed to fold operation: " << *user << "\n");
          return user->emitError("operation could not be folded");
        }
        
        for (auto [result, foldResult] : llvm::zip(user->getResults(), foldResults)) {
          Attribute resultAttr = dyn_cast<Attribute>(foldResult);
          if (!resultAttr)
            return user->emitError("fold result is not an attribute");
          
          if (visited.insert({result, resultAttr}).second)
            worklist.push_back({result, resultAttr});
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Collected " << operands.size()
                          << " transitive register allocation operands\n");
  return success();
}

namespace {

/*

Assumptions of this pass:
* All sequences are inlined
* Instructions appear in program order

Overall Algorithm:
* Assign op indices to all operations for live range analysis
* Each SSA value of RegisterTypeInterface type has a live range
* Sort ranges by increasing start
* Iterate over the ranges defined by virtual_reg operations and assign the first register in the list that is available
* Check availability as follows and repeat until one is found:
  - iterate over overlapping ranges and confirm none uses it as fixed reg already.
  - Follow all register uses until they terminate in a RegisterAllocationOpInterface usage
  - For all SSA values of RegisterTypeInterface type defined along the way, check that the range is not already fixed (if it is multiple virtual regs can set it and it's probably a problem) and that the computed register value is available for its live range.

*/

/// Represents a register and its live range.
struct RegisterLiveRange {
  rtg::RegisterAttrInterface fixedReg;
  OpResult reg;
  unsigned start;
  unsigned end;
};

class LinearScanRegisterAllocationPass
    : public circt::rtg::impl::LinearScanRegisterAllocationPassBase<
          LinearScanRegisterAllocationPass> {
public:
  void runOnOperation() override;
};

} // end namespace

static void expireOldInterval(SmallVector<RegisterLiveRange *> &active,
                              RegisterLiveRange *reg) {
  LLVM_DEBUG(llvm::dbgs() << "Expiring old intervals for register starting at "
                          << reg->start << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  Active intervals before expiration: " << active.size() << "\n");

  // TODO: use a better datastructure for 'active'
  llvm::sort(active, [](auto *a, auto *b) { return a->end < b->end; });

  for (auto *iter = active.begin(); iter != active.end(); ++iter) {
    auto *a = *iter;
    if (a->end >= reg->start) {
      LLVM_DEBUG(llvm::dbgs() << "  Keeping active interval ending at " << a->end << "\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "  Expiring interval ending at " << a->end << "\n");
    active.erase(iter--);
  }

  LLVM_DEBUG(llvm::dbgs() << "  Active intervals after expiration: " << active.size() << "\n");
}

void LinearScanRegisterAllocationPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Processing "
                          << OpWithFlags(getOperation(),
                                         OpPrintingFlags().skipRegions())
                          << "\n\n");

  if (getOperation()->getNumRegions() != 1 ||
      getOperation()->getRegion(0).getBlocks().size() != 1) {
    getOperation()->emitError("expected a single region with a single block");
    return signalPassFailure();
  }

  DenseMap<Operation *, unsigned> opIndices;
  unsigned maxIdx;
  // Find Text segment and walk its operations
  // FIXME: assumes there is exactly one such segment
  rtg::SegmentOp textSeg;
  getOperation()->walk([&](rtg::SegmentOp segOp) {
    if (segOp.getKind() != rtg::SegmentKind::Text)
      return;

    LLVM_DEBUG(llvm::dbgs() << "Found text segment: " << segOp << "\n");
    textSeg = segOp;
    for (auto [i, op] : llvm::enumerate(*segOp.getBody())) {
      // TODO: ideally check that the IR is already fully elaborated
      opIndices[&op] = i;
      maxIdx = i;
      LLVM_DEBUG(llvm::dbgs() << "  Op " << i << ": " << op << "\n");
    }
    LLVM_DEBUG(llvm::dbgs() << "Total operations in text segment: " << (maxIdx + 1) << "\n");
  });

  if (!textSeg) {
    getOperation()->emitError("expected a text segment");
    return signalPassFailure();
  }

  // Collect all the register intervals we have to consider.
  LLVM_DEBUG(llvm::dbgs() << "\nCollecting register live ranges...\n");
  SmallVector<std::unique_ptr<RegisterLiveRange>> regRanges;
  SmallVector<RegisterLiveRange *> active;
  for (auto &op : *textSeg.getBody()) {
    for (auto result : op.getResults()) {
      if (!isa<rtg::RegisterTypeInterface>(result.getType()))
        continue;

      LLVM_DEBUG(llvm::dbgs() << "Processing register result: " << result
                              << " from op: " << op << "\n");
      auto &lr = regRanges.emplace_back(std::make_unique<RegisterLiveRange>());
      lr->start = maxIdx;
      lr->end = 0;
      lr->reg = result;
     
      bool hasUser = false;
      for (auto *user : op.getUsers()) {
        if (!isa<rtg::RegisterAllocationOpInterface>(user))
          continue;

        // TODO: support labels and control-flow loops (jumps in general)
        unsigned idx = opIndices.at(user);
        LLVM_DEBUG(llvm::dbgs() << "  User at index " << idx << ": " << *user << "\n");
        lr->start = std::min(lr->start, idx);
        lr->end = std::max(lr->end, idx);
        hasUser = true;
      }

      if (!hasUser) {
        LLVM_DEBUG(llvm::dbgs() << "  No RegisterAllocationOpInterface users, removing range\n");
        regRanges.pop_back();
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "  Live range: [" << lr->start << ", " << lr->end << "]\n");

      if (auto regOp = dyn_cast<rtg::ConstantOp>(&op)) {
        auto reg = dyn_cast<rtg::RegisterAttrInterface>(regOp.getValue());
        if (!reg) {
          op.emitError("expected register attribute");
          return signalPassFailure();
        }
        lr->fixedReg = reg;
        LLVM_DEBUG(llvm::dbgs() << "  Fixed register: " << reg << "\n");

        // Reserve fixed registers from the start. It will be made available again
        // past the interval end. Not reserving it from the start can lead to the
        // same register being chosen for a virtual register that overlaps with the
        // fixed register interval.
        // TODO: don't overapproximate that much
        active.push_back(lr.get());
        LLVM_DEBUG(llvm::dbgs() << "  Added fixed register to active list\n");
      }
    }
  }

  // Sort such that we can process registers by increasing interval start.
  LLVM_DEBUG(llvm::dbgs() << "\nSorting " << regRanges.size() << " register ranges by start time\n");
  llvm::sort(regRanges, [](const auto &a, const auto &b) {
    return a->start < b->start || (a->start == b->start && !isa<rtg::VirtualRegisterOp>(a->reg.getOwner()));
  });

  LLVM_DEBUG(llvm::dbgs() << "\nStarting register allocation...\n");
  for (auto &lr : regRanges) {
    LLVM_DEBUG(llvm::dbgs() << "Processing register range [" << lr->start << ", " << lr->end
                            << "] for " << lr->reg << "\n");

    // Make registers out of live range available again.
    expireOldInterval(active, lr.get());

    // Handle already fixed registers.
    auto virtualReg = dyn_cast<rtg::VirtualRegisterOp>(lr->reg.getOwner());
    if (lr->fixedReg || !virtualReg) {
      LLVM_DEBUG(llvm::dbgs() << "  Skipping already fixed or non-virtual register\n");
      continue;
    }

    // Handle virtual registers.
    LLVM_DEBUG(llvm::dbgs() << "  Processing virtual register\n");
    auto configAttr =
        cast<rtg::VirtualRegisterConfigAttr>(virtualReg.getAllowedRegsAttr());
    LLVM_DEBUG(llvm::dbgs() << "  Allowed registers: " << configAttr.getAllowedRegs().size() << "\n");
    rtg::RegisterAttrInterface availableReg;
    for (auto reg : configAttr.getAllowedRegs()) {
      LLVM_DEBUG(llvm::dbgs() << "    Trying register: " << reg << "\n");
      if (llvm::any_of(active, [&](auto *r) { return r->fixedReg == reg; })) {
        LLVM_DEBUG(llvm::dbgs() << "    Register is already active, checking conflicts\n");
        continue;
      }

      availableReg = reg;

      LLVM_DEBUG(llvm::dbgs() << "    Register is available (not in active list)\n");

      SetVector<std::pair<Value, rtg::RegisterAttrInterface>> registers;
      if (failed(collectTransitiveRegisterAllocationOperands(lr->reg, reg, registers)))
        return signalPassFailure();

      LLVM_DEBUG(llvm::dbgs() << "    Found " << registers.size()
                              << " transitive register allocation operands\n");

      DenseMap<Value, RegisterLiveRange *> registerToLiveRange;
      for (auto [reg, regAttr] : registers) {
        Value registerValue = reg;
        LLVM_DEBUG(llvm::dbgs() << "      Processing register value: " << registerValue
                                << " with attr: " << regAttr << "\n");
        auto *it = llvm::find_if(regRanges, [&](auto &lr) { return lr->reg == registerValue; });
        if (it == regRanges.end()) {
          reg.getDefiningOp()->emitError("register value not found in live ranges");
          return signalPassFailure();
        }

        RegisterLiveRange *lr = it->get();
        registerToLiveRange.insert({reg, lr});
        LLVM_DEBUG(llvm::dbgs() << "      Mapped to live range [" << lr->start << ", " << lr->end << "]\n");

        // Get all live ranges that overlap with the one defined by 'reg'
        bool hasConflict = false;
        LLVM_DEBUG(llvm::dbgs() << "      Checking for conflicts with register " << regAttr << "\n");
        for (auto &otherLr : regRanges) {
          if (lr == otherLr.get())
            continue;

          // Skip if lr and otherLr don't overlap
          if (otherLr->start > lr->end || otherLr->end < lr->start)
            continue;

          LLVM_DEBUG(llvm::dbgs() << "        Overlapping range [" << otherLr->start
                                  << ", " << otherLr->end << "] with fixed reg: "
                                  << otherLr->fixedReg << "\n");

          // Check if this overlapping range has its fixed reg set to 'regAttr'
          if (otherLr->fixedReg == regAttr) {
            LLVM_DEBUG(llvm::dbgs() << "        Conflict detected!\n");
            hasConflict = true;
            break;
          }
        }

        if (hasConflict) {
          LLVM_DEBUG(llvm::dbgs() << "      Register " << reg << " has conflicts, trying next\n");
          availableReg = rtg::RegisterAttrInterface();
          break;
        }
      }

      if (!availableReg)
        continue;
      
      for (auto [reg, regAttr] : registers) {
        if (registerToLiveRange[reg]->fixedReg) {
          reg.getDefiningOp()->emitError("register already fixed");
          return signalPassFailure();
        }

        LLVM_DEBUG(llvm::dbgs() << "      Assigning register " << regAttr
                                << " to value " << reg << "\n");
        registerToLiveRange[reg]->fixedReg = regAttr;
      }

      break;
    }

    if (!availableReg) {
      LLVM_DEBUG(llvm::dbgs() << "  No available register found, need to spill\n");
      ++numRegistersSpilled;
      virtualReg.emitError(
          "need to spill this register, but not supported yet");
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "  Assigned register " << availableReg
                            << " to virtual register\n");
    // lr->fixedReg = availableReg;
    active.push_back(lr.get());
  }

  LLVM_DEBUG({
    for (auto &regRange : regRanges) {
      llvm::dbgs() << "Start: " << regRange->start << ", End: " << regRange->end
                   << ", Selected: " << regRange->fixedReg << "\n";
    }
    llvm::dbgs() << "\n";
  });

  LLVM_DEBUG(llvm::dbgs() << "\nReplacing virtual registers with fixed registers...\n");
  circt::UnusedOpPruner operationPruner;
  for (auto &reg : regRanges) {
    // No need to fix already fixed registers.
    if (isa<rtg::ConstantOp>(reg->reg.getOwner())) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping already fixed register: " << reg->fixedReg << "\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Replacing virtual register " << reg->reg
                            << " with fixed register " << reg->fixedReg << "\n");
    OpBuilder builder(reg->reg.getOwner());
    auto newFixedReg = rtg::ConstantOp::create(builder, reg->reg.getLoc(), reg->fixedReg);
    reg->reg.replaceAllUsesWith(newFixedReg);
    operationPruner.eraseNow(reg->reg.getOwner());
  }

  // Erase any operations that became unused after erasing the virtual register operations
  operationPruner.eraseNow();

  LLVM_DEBUG(llvm::dbgs() << "Register allocation completed successfully!\n");
}
