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

#include <variant>

namespace circt {
namespace rtg {
#define GEN_PASS_DEF_LINEARSCANREGISTERALLOCATIONPASS
#include "circt/Dialect/RTG/Transforms/RTGPasses.h.inc"
} // namespace rtg
} // namespace circt

using namespace mlir;
using namespace circt;

#define DEBUG_TYPE "rtg-linear-scan-register-allocation"

namespace {

struct Unit {};

/// This class provides support for representing a failure result, or a valid
/// value of type `T`. This allows for integrating with LogicalResult, while
/// also providing a value on the success path.
template <typename S, typename F>
class [[nodiscard]] Result {
public:
  /// Allow constructing from a LogicalResult. The result *must* be a failure.
  /// Success results should use a proper instance of type `T`.
  Result(S &&val) : resultValue(std::forward<S>(val)) {}
  Result(F &&val) : resultValue(std::forward<F>(val)) {}
  Result(const S &val) : resultValue(val) {}
  Result(const F &val) : resultValue(val) {}

  operator LogicalResult() const { return std::holds_alternative<S>(resultValue) ? success() : failure(); }

  F error() const {
    assert(!std::holds_alternative<S>(resultValue) && "no failure reason");
    return std::get<F>(resultValue);
  }

  S value() const {
    assert(std::holds_alternative<S>(resultValue) && "no value");
    return std::get<S>(resultValue);
  }

private:
  std::variant<S, F> resultValue;
};

template<typename S, typename F>
inline bool succeeded(const Result<S, F> &result) {
  return succeeded(LogicalResult(result));
}

template<typename S, typename F>
inline bool failed(const Result<S, F> &result) {
  return failed(LogicalResult(result));
}

enum class RegisterAvailabilityFailure {
  Fatal,
  InUse,
  ConstraintViolation,
};

using RegisterAvailabilityResult = Result<Unit, RegisterAvailabilityFailure>;

} // namespace

// Follow all users of a value transitively until reaching an operation that
// implements RegisterAllocationOpInterface. Store the operand values at those
// interface operations in the output vector.
static RegisterAvailabilityResult collectTransitiveRegisterAllocationOperands(
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
        operands.insert({current, cast_or_null<rtg::RegisterAttrInterface>(currentAttr)});
      } else if (auto constraintOp = dyn_cast<rtg::ConstraintOp>(user)) {
        if (reg) {
          auto condAttr = dyn_cast<IntegerAttr>(currentAttr);
          if (!condAttr || condAttr.getInt() != 1) {
            LLVM_DEBUG(llvm::dbgs() << "    Constraint could not be satisfied\n");
            return RegisterAvailabilityFailure::ConstraintViolation;
          }
        }
      } else {
        // Not a RegisterAllocationOpInterface - continue following results
        LLVM_DEBUG(llvm::dbgs() << "    Following non-RegisterAllocationOpInterface: "
                                << *user << "\n");
        
        if (reg) {
          // Check that all additional operands are provided by constant-like operations
          SmallVector<Attribute> operandAttrs;
          for (auto operand : user->getOperands()) {
            if (operand == current) {
              operandAttrs.push_back(currentAttr);
              continue;
            }
            auto *defOp = operand.getDefiningOp();
            if (!defOp || !defOp->hasTrait<OpTrait::ConstantLike>()) {
              // This is most likely because we are following a constraint that is too complex.
              return RegisterAvailabilityFailure::ConstraintViolation;
            }
            
            SmallVector<OpFoldResult> operandFoldResults;
            if (failed(operand.getDefiningOp()->fold(operandFoldResults))) {
              operand.getDefiningOp()->emitError(
                  "folding constant like operation failed???");
              return RegisterAvailabilityFailure::Fatal;
            }

            operandAttrs.push_back(cast<Attribute>(operandFoldResults[0]));
          }

          // Fold the operation to get the result attribute
          SmallVector<OpFoldResult> foldResults;
          if (failed(user->fold(operandAttrs, foldResults))) {
            LLVM_DEBUG(llvm::dbgs() << "    Failed to fold operation: " << *user << "\n");
            user->emitError("operation could not be folded");
            return RegisterAvailabilityFailure::Fatal;
          }
        
          for (auto [result, foldResult] : llvm::zip(user->getResults(), foldResults)) {
            Attribute resultAttr = dyn_cast<Attribute>(foldResult);
            if (!resultAttr) {
              user->emitError("fold result is not an attribute");
              return RegisterAvailabilityFailure::Fatal;
            }
            
            if (visited.insert({result, resultAttr}).second)
              worklist.push_back({result, resultAttr});
          }
        } else {
          for (auto result : user->getResults()) {
            if (visited.insert({result, Attribute()}).second)
              worklist.push_back({result, Attribute()});
          }
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Collected " << operands.size()
                          << " transitive register allocation operands\n");
  return Unit();
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
  SmallVector<std::pair<Value, rtg::RegisterAttrInterface>> dependentRegs;
  rtg::VirtualRegisterOp virtualReg;
  rtg::RegisterAttrInterface fixedReg;
  unsigned start;
  unsigned end;
};

struct RegisterAllocator {
  RegisterAllocator(rtg::SegmentOp segOp) : segOp(segOp) {}

  LogicalResult allocate();

private:
  void computeOpIndices();
  LogicalResult computeLiveRanges();
  bool hasConflict(Value reg, rtg::RegisterAttrInterface regAttr);
  rtg::RegisterAttrInterface findAvailableRegister(rtg::VirtualRegisterOp virtualReg);
  RegisterAvailabilityResult isRegisterAvailable(rtg::VirtualRegisterOp virtualReg, rtg::RegisterAttrInterface reg);
  void computeOverlappingFixedRegisters(RegisterLiveRange *lr, DenseSet<rtg::RegisterAttrInterface> &fixedRegs);
  RegisterLiveRange *createRange(rtg::VirtualRegisterOp op);
  LogicalResult allocateRegisters();
  void cleanup();

private:
  rtg::SegmentOp segOp;
  DenseMap<Operation *, unsigned> opIndices;
  DenseMap<rtg::VirtualRegisterOp, RegisterLiveRange *> regToLiveRange;
  SmallVector<std::unique_ptr<RegisterLiveRange>> regRanges;
  SmallVector<RegisterLiveRange *> active;
  DenseSet<rtg::RegisterAttrInterface> reserved;
  SmallVector<std::pair<unsigned, rtg::RegisterAttrInterface>> reservedEndIndex;
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

  for (auto segOp : getOperation()->getRegion(0).getOps<rtg::SegmentOp>()) {
    // Perform register allocation for all text segments in isolation. There
    // should usually only be one such segment.
    if (segOp.getKind() == rtg::SegmentKind::Text) {
      RegisterAllocator allocator(segOp);
      if (failed(allocator.allocate()))
        return signalPassFailure();
    }
  }
}

LogicalResult RegisterAllocator::allocate() {
  computeOpIndices();

  if (failed(computeLiveRanges()) || failed(allocateRegisters()))
    return failure();

  cleanup();
  return success();
}

void RegisterAllocator::computeOpIndices() {
  for (auto [i, op] : llvm::enumerate(*segOp.getBody())) {
    // TODO: ideally check that the IR is already fully elaborated
    opIndices[&op] = i;
    LLVM_DEBUG(llvm::dbgs() << "  Op " << i << ": " << op << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "Total operations in text segment: " << opIndices.size() << "\n");
}

/// Creates a range if the op result has users that form a range. Returns
/// 'nullptr' if no range was created.
RegisterLiveRange *RegisterAllocator::createRange(rtg::VirtualRegisterOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Processing virtual register: " << op << "\n");

  auto &lr = regRanges.emplace_back(std::make_unique<RegisterLiveRange>());
  lr->start = opIndices.size();
  lr->end = 0;
  lr->virtualReg = op;
  
  SetVector<std::pair<Value, rtg::RegisterAttrInterface>> registers;
  if (failed(collectTransitiveRegisterAllocationOperands(op.getResult(), {}, registers)))
    return nullptr;

  bool hasUser = false;
  for (auto *user : op->getUsers()) {
    if (!isa<rtg::RegisterAllocationOpInterface>(user))
      continue;

    // TODO: support labels and control-flow loops (jumps in general)
    unsigned idx = opIndices.at(user);
    LLVM_DEBUG(llvm::dbgs() << "  User at index " << idx << ": " << *user << "\n");
    lr->start = std::min(lr->start, idx);
    lr->end = std::max(lr->end, idx);
    hasUser = true;
  }
  for (auto [reg, regAttr] : registers) {
    lr->dependentRegs.emplace_back(reg, rtg::RegisterAttrInterface());

    for (auto *user : reg.getUsers()) {
      if (!isa<rtg::RegisterAllocationOpInterface>(user))
        continue;

      // TODO: support labels and control-flow loops (jumps in general)
      unsigned idx = opIndices.at(user);
      LLVM_DEBUG(llvm::dbgs() << "  User at index " << idx << ": " << *user << "\n");
      lr->start = std::min(lr->start, idx);
      lr->end = std::max(lr->end, idx);
      hasUser = true;
    }
  }

  if (!hasUser) {
    LLVM_DEBUG(llvm::dbgs() << "  No RegisterAllocationOpInterface users, removing range\n");
    regRanges.pop_back();
    return nullptr;
  }

  LLVM_DEBUG(llvm::dbgs() << "  Live range: [" << lr->start << ", " << lr->end << "]\n");
  return lr.get();
}

LogicalResult RegisterAllocator::computeLiveRanges() {
  // Collect all the register intervals we have to consider.
  LLVM_DEBUG(llvm::dbgs() << "\nCollecting register live ranges...\n");
  for (auto op : segOp.getBody()->getOps<rtg::ConstantOp>()) {
    auto reg = dyn_cast<rtg::RegisterAttrInterface>(op.getValue());
    if (!reg)
      continue;

    reserved.insert(reg);

    unsigned maxIdx = 0;
    for (auto *user : op->getUsers())
      maxIdx = std::max(maxIdx, opIndices.at(user));
    reservedEndIndex.emplace_back(maxIdx, reg);

    LLVM_DEBUG(llvm::dbgs() << "  Added fixed register to active list: " << reg << "\n");
  }

  llvm::sort(reservedEndIndex, [](auto &a, auto &b) { return a.first < b.first; });

  for (auto op : segOp.getBody()->getOps<rtg::VirtualRegisterOp>()) {
    auto *lr = createRange(op);
    if (!lr)
      continue;

    regToLiveRange[op] = lr;

    // if (auto regOp = dyn_cast<rtg::ConstantOp>(op)) {
    //   auto reg = dyn_cast<rtg::RegisterAttrInterface>(regOp.getValue());
    //   if (!reg)
    //     return op.emitError("expected register attribute");
      
    //   lr->fixedReg = reg;
    //   LLVM_DEBUG(llvm::dbgs() << "  Fixed register: " << reg << "\n");

    //   // Reserve fixed registers from the start. It will be made available again
    //   // past the interval end. Not reserving it from the start can lead to the
    //   // same register being chosen for a virtual register that overlaps with the
    //   // fixed register interval.
    //   // TODO: don't overapproximate that much
    //   active.push_back(lr);
    //   LLVM_DEBUG(llvm::dbgs() << "  Added fixed register to active list\n");
    // }
  }

  // Sort such that we can process registers by increasing interval start.
  LLVM_DEBUG(llvm::dbgs() << "\nSorting " << regRanges.size() << " register ranges by start time\n");
  llvm::sort(regRanges, [](const auto &a, const auto &b) {
    return a->start < b->start || (a->start == b->start && !a->virtualReg);
  });

  return success();
}

bool RegisterAllocator::hasConflict(Value reg, rtg::RegisterAttrInterface regAttr) {
  // LLVM_DEBUG(llvm::dbgs() << "      Processing register value: " << reg
  //                         << " with attr: " << regAttr << "\n");
  // auto *lr = regToLiveRange[reg];
  // if (!lr) {
  //   reg.getDefiningOp()->emitError("register value not found in live ranges");
  //   return {};
  // }

  // // Get all live ranges that overlap with the one defined by 'reg'
  // LLVM_DEBUG(llvm::dbgs() << "      Checking for conflicts with register " << regAttr << "\n");
  // for (auto &otherLr : regRanges) {
  //   if (lr == otherLr.get())
  //     continue;

  //   if (otherLr->start > lr->end)
  //     break;

  //   if (otherLr->end < lr->start)
  //     continue;

  //   LLVM_DEBUG(llvm::dbgs() << "        Overlapping range [" << otherLr->start
  //                           << ", " << otherLr->end << "] with fixed reg: "
  //                           << otherLr->fixedReg << "\n");

  //   // Check if this overlapping range has its fixed reg set to 'regAttr'
  //   if (otherLr->fixedReg == regAttr) {
  //     LLVM_DEBUG(llvm::dbgs() << "      Register " << reg << " has conflicts\n");
  //     return true;
  //   }
  // }

  return false;
}

RegisterAvailabilityResult RegisterAllocator::isRegisterAvailable(rtg::VirtualRegisterOp virtualReg, rtg::RegisterAttrInterface reg) {
  LLVM_DEBUG(llvm::dbgs() << "    Trying register: " << reg << "\n");
  
  if (reserved.contains(reg)) {
    LLVM_DEBUG(llvm::dbgs() << "    Register is reserved, skipping\n");
    return RegisterAvailabilityFailure::InUse;
  }

  SetVector<std::pair<Value, rtg::RegisterAttrInterface>> registers;
  auto res = collectTransitiveRegisterAllocationOperands(virtualReg.getResult(), reg, registers);
  if (failed(res))
    return res;

  for (auto [regVal, regAttr] : registers) {
    if (reserved.contains(regAttr)) {
      LLVM_DEBUG(llvm::dbgs() << "    Register is reserved, skipping\n");
      return RegisterAvailabilityFailure::InUse;
    }
  }

  if (llvm::any_of(active, [&](auto *r) {
    if (r->fixedReg == reg)
      return true;
    for (auto [otherReg, otherRegAttr] : r->dependentRegs) {
      if (!otherRegAttr)
        break;
      if (reg == otherRegAttr)
        return true;
    }

    for (auto [reg, regAttr] : registers) {
      if (regAttr == r->fixedReg)
        return true;

      for (auto [otherReg, otherRegAttr] : r->dependentRegs) {
        if (!otherRegAttr)
          break;
        if (regAttr == otherRegAttr)
          return true;
      }
    }

    return false;
  })) {
    LLVM_DEBUG(llvm::dbgs() << "    Register is already active, checking conflicts\n");
    return RegisterAvailabilityFailure::InUse;
  }

  LLVM_DEBUG(llvm::dbgs() << "    Found " << registers.size()
                          << " transitive register allocation operands\n");
  
  if (regToLiveRange[virtualReg]->fixedReg && regToLiveRange[virtualReg]->fixedReg != reg) {
    virtualReg.emitError("register already fixed to other register value ") << regToLiveRange[virtualReg]->fixedReg << " vs " << reg << "\n";
    return RegisterAvailabilityFailure::Fatal;
  }

  LLVM_DEBUG(llvm::dbgs() << "      Assigning register " << reg
                          << " to value " << reg << "\n");
  auto *lr = regToLiveRange[virtualReg];
  lr->fixedReg = reg;
  lr->dependentRegs = SmallVector<std::pair<Value, rtg::RegisterAttrInterface>>(registers.getArrayRef());
  
  return Unit();
}

void RegisterAllocator::computeOverlappingFixedRegisters(RegisterLiveRange *lr, DenseSet<rtg::RegisterAttrInterface> &fixedRegs) {
  for (auto &otherLr : regRanges) {
    if (lr == otherLr.get())
      continue;

    // Skip if lr and otherLr don't overlap
    if (otherLr->start > lr->end || otherLr->end < lr->start)
      continue;

    fixedRegs.insert(otherLr->fixedReg);
  }
}

rtg::RegisterAttrInterface RegisterAllocator::findAvailableRegister(rtg::VirtualRegisterOp virtualReg) {
  auto configAttr =
      cast<rtg::VirtualRegisterConfigAttr>(virtualReg.getAllowedRegsAttr());
  LLVM_DEBUG(llvm::dbgs() << "  Allowed registers: " << configAttr.getAllowedRegs().size() << "\n");

  // DenseSet<rtg::RegisterAttrInterface> fixedRegs;
  // computeOverlappingFixedRegisters(valueToLiveRange[virtualReg.getResult()], fixedRegs);

  DenseSet<RegisterAvailabilityFailure> failureReasons;
  for (auto reg : configAttr.getAllowedRegs()) {
    auto res = isRegisterAvailable(virtualReg, reg);
    if (failed(res)) {
      // Fatal errors already printed an error message.
      if (res.error() == RegisterAvailabilityFailure::Fatal)
        return {};

      failureReasons.insert(res.error());
      continue;
    }

    // If we found a valid register, use it without trying any other.
    return reg;
  }

  if (failureReasons.contains(RegisterAvailabilityFailure::InUse)) {
    // If at least one register was marked InUse it could have been chosen if it weren't in use since it is not reported as any other error.
    virtualReg.emitError("all allowed registers in use; need to spill registers, but not supported yet");
  } else if (failureReasons.contains(RegisterAvailabilityFailure::ConstraintViolation)) {
    // If we reach here, all registers failed to satisfy all constraints.
    virtualReg.emitError("no register available that satisfies all constraints");
  }

  return {};
}

LogicalResult RegisterAllocator::allocateRegisters() {
  for (auto &lr : regRanges) {
    LLVM_DEBUG(llvm::dbgs() << "Processing register range [" << lr->start << ", " << lr->end
                            << "] for " << lr->virtualReg << "\n");

    // Make registers out of live range available again.
    expireOldInterval(active, lr.get());

    for (auto [idx, reg] : llvm::make_early_inc_range(reservedEndIndex)) {
      if (idx >= lr->start)
        break;

      reserved.erase(reg);
    }

    // Handle already fixed registers.
    // auto virtualReg = dyn_cast<rtg::VirtualRegisterOp>(lr->reg.getOwner());
    if (lr->fixedReg || !lr->virtualReg)
      continue;

    // Handle virtual registers.
    auto availableReg = findAvailableRegister(lr->virtualReg);
    if (!availableReg) {
      // ++numRegistersSpilled;
      return failure();
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

  return success();
}

void RegisterAllocator::cleanup() {
  LLVM_DEBUG(llvm::dbgs() << "\nReplacing virtual registers with fixed registers...\n");
  circt::UnusedOpPruner operationPruner;
  for (auto &reg : regRanges) {
    LLVM_DEBUG(llvm::dbgs() << "Replacing virtual register " << reg->virtualReg
                            << " with fixed register " << reg->fixedReg << "\n");
    OpBuilder builder(reg->virtualReg);
    auto newFixedReg = rtg::ConstantOp::create(builder, reg->virtualReg.getLoc(), reg->fixedReg);
    reg->virtualReg.getResult().replaceAllUsesWith(newFixedReg);
    operationPruner.eraseNow(reg->virtualReg);
  }

  // Erase any operations that became unused after erasing the virtual register operations
  operationPruner.eraseNow();

  LLVM_DEBUG(llvm::dbgs() << "Register allocation completed successfully!\n");
}
