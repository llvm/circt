//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// Class declarations
//===----------------------------------------------------------------------===//

namespace {

/// Base class for all live ranges. Should not be used to construct a live range
/// directly.
struct LiveRange {
  LiveRange(Location loc, unsigned start, unsigned end)
      : loc(loc), start(start), end(end) {}

  /// Location of the defining operation.
  Location loc;
  /// Operation index of the first user.
  unsigned start;
  /// Operation index of the last user.
  unsigned end;
};

/// Represents a fixed register and its live range.
struct FixedLiveRange : public LiveRange {
  FixedLiveRange(Location loc, rtg::RegisterAttrInterface fixedReg,
                 unsigned start, unsigned end)
      : LiveRange(loc, start, end), fixedReg(fixedReg) {}

  /// The concrete register as attribute.
  rtg::RegisterAttrInterface fixedReg;
};

/// Represents a dependent register and its live range.
struct DependentLiveRange : public LiveRange {
  DependentLiveRange(Value value, unsigned start, unsigned end)
      : LiveRange(value.getLoc(), start, end), value(value) {}

  /// The value that is dependent on the virtual register. It is of
  /// `rtg::RegisterTypeInterface` type.
  Value value;
};

/// Represents a virtual register and its live range.
struct VirtualLiveRange : public LiveRange {
  VirtualLiveRange(rtg::VirtualRegisterOp virtualReg, unsigned start,
                   unsigned end)
      : LiveRange(virtualReg.getLoc(), start, end), virtualReg(virtualReg) {}

  /// The virtual register operation defining this live range.
  rtg::VirtualRegisterOp virtualReg;
  /// Dependent registers that must be allocated together with this range.
  SmallVector<DependentLiveRange> dependentRegs;
};

struct RegisterAllocationResult {
  enum class Kind {
    /// The register is available for allocation.
    Available,
    /// The register is currently in use by an active live range.
    InUse,
    /// The register violates one or more constraints (any may be in use).
    ConstraintViolation,
    /// A fatal error occurred that prevents allocation from continuing.
    FatalError,
  };

private:
  explicit RegisterAllocationResult(Kind kind) : kind(kind) {}

  Kind kind;
  std::variant<rtg::ConstraintOp, LiveRange> value;

public:
  Kind getKind() const { return kind; }

  LiveRange getUser() const {
    assert(kind == Kind::InUse);
    return std::get<LiveRange>(value);
  }

  rtg::ConstraintOp getConstraint() const {
    assert(kind == Kind::ConstraintViolation);
    return std::get<rtg::ConstraintOp>(value);
  }

  bool isAvailable() const { return kind == Kind::Available; }

  static RegisterAllocationResult available() {
    return RegisterAllocationResult(Kind::Available);
  }

  static RegisterAllocationResult inUseBy(LiveRange liveRange) {
    auto res = RegisterAllocationResult(Kind::InUse);
    res.value = liveRange;
    return res;
  }

  static RegisterAllocationResult
  constraintViolation(rtg::ConstraintOp constraintOp) {
    auto res = RegisterAllocationResult(Kind::ConstraintViolation);
    res.value = constraintOp;
    return res;
  }

  static RegisterAllocationResult fatalError() {
    return RegisterAllocationResult(Kind::FatalError);
  }
};

/// Caches information about register live ranges.
struct LiveRangeCache {
  LogicalResult populate(Region &region);
  void clear();

  /// Maps each operation to its index in the segment for ordering.
  DenseMap<Operation *, unsigned> opIndices;
  /// Reverse mapping from index to operation for diagnostic purposes.
  DenseMap<unsigned, Operation *> indexToOp;
  /// Storage for all register live ranges.
  SmallVector<VirtualLiveRange> virtualRanges;
  /// All fixed live ranges (before any allocation) sorted by increasing start
  /// index.
  SmallVector<FixedLiveRange> reservedRanges;
};

class LinearScanRegisterAllocationPass
    : public circt::rtg::impl::LinearScanRegisterAllocationPassBase<
          LinearScanRegisterAllocationPass> {
public:
  void runOnOperation() override;
};

} // end namespace

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Folds an operation and adds its results to the worklist.
/// Returns Available on success, or an error status on failure.
static LogicalResult
foldAndEnqueueResults(Operation *user, Value current, Attribute currentAttr,
                      SmallVectorImpl<std::pair<Value, Attribute>> &worklist,
                      DenseMap<Value, Attribute> &visited) {
  // Check that all additional operands are provided by constant-like operations
  SmallVector<Attribute> operandAttrs;
  for (auto operand : user->getOperands()) {
    if (operand == current) {
      operandAttrs.push_back(currentAttr);
      continue;
    }
    auto *defOp = operand.getDefiningOp();
    if (!defOp || !defOp->hasTrait<OpTrait::ConstantLike>())
      return failure();

    SmallVector<OpFoldResult> operandFoldResults;
    if (failed(operand.getDefiningOp()->fold(operandFoldResults))) {
      return operand.getDefiningOp()->emitError(
          "folding constant like operation failed");
    }

    operandAttrs.push_back(cast<Attribute>(operandFoldResults[0]));
  }

  // Fold the operation to get the result attribute
  SmallVector<OpFoldResult> foldResults;
  if (failed(user->fold(operandAttrs, foldResults))) {
    LLVM_DEBUG(llvm::dbgs()
               << "    Failed to fold operation: " << *user << "\n");
    return user->emitError("operation could not be folded");
  }

  for (auto [result, foldResult] : llvm::zip(user->getResults(), foldResults)) {
    Attribute resultAttr = dyn_cast<Attribute>(foldResult);
    if (!resultAttr) {
      return user->emitError("fold result is not an attribute");
    }

    if (visited.insert({result, resultAttr}).second)
      worklist.push_back({result, resultAttr});
  }

  return success();
}

// Follow all users of a value transitively until reaching an operation that
// implements RegisterAllocationOpInterface. Store the operand values at those
// interface operations in the output vector.
static RegisterAllocationResult collectTransitiveRegisterAllocationOperands(
    Value value, rtg::RegisterAttrInterface reg,
    SetVector<std::pair<Value, rtg::RegisterAttrInterface>> &operands) {
  LLVM_DEBUG(llvm::dbgs()
             << "Collecting transitive register allocation operands for value: "
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
      if (auto regAllocOp =
              dyn_cast<rtg::RegisterAllocationOpInterface>(user)) {
        // Found a RegisterAllocationOpInterface - store the operand and don't
        // follow results further
        LLVM_DEBUG(llvm::dbgs() << "    Found RegisterAllocationOpInterface: "
                                << *user << "\n");
        operands.insert(
            {current, cast_or_null<rtg::RegisterAttrInterface>(currentAttr)});
        continue;
      }

      if (auto constraintOp = dyn_cast<rtg::ConstraintOp>(user)) {
        if (!reg)
          continue;

        auto condAttr = dyn_cast<IntegerAttr>(currentAttr);
        if (!condAttr || condAttr.getValue().isZero()) {
          LLVM_DEBUG(llvm::dbgs() << "    Constraint could not be satisfied\n");
          return RegisterAllocationResult::constraintViolation(constraintOp);
        }
        continue;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "    Following non-RegisterAllocationOpInterface: " << *user
                 << "\n");

      if (reg) {
        if (failed(foldAndEnqueueResults(user, current, currentAttr, worklist,
                                         visited)))
          return RegisterAllocationResult::fatalError();
      } else {
        for (auto result : user->getResults()) {
          if (visited.insert({result, Attribute()}).second)
            worklist.push_back({result, Attribute()});
        }
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Collected " << operands.size()
                          << " transitive register allocation operands\n");
  return RegisterAllocationResult::available();
}

static void expireOldInterval(SmallVector<FixedLiveRange> &active,
                              const VirtualLiveRange &liveRange) {
  LLVM_DEBUG(llvm::dbgs() << "Expiring old intervals for register starting at "
                          << liveRange.start << "\n");
  LLVM_DEBUG(llvm::dbgs() << "  Active intervals before expiration: "
                          << active.size() << "\n");

  // TODO: use a better datastructure for 'active'
  llvm::sort(active, [](const FixedLiveRange &a, const FixedLiveRange &b) {
    return a.end < b.end;
  });

  for (auto *iter = active.begin(); iter != active.end(); ++iter) {
    const auto &activeRange = *iter;
    if (activeRange.end >= liveRange.start) {
      LLVM_DEBUG(llvm::dbgs() << "  Keeping active interval ending at "
                              << activeRange.end << "\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "  Expiring interval ending at " << activeRange.end << "\n");
    active.erase(iter--);
  }

  LLVM_DEBUG(llvm::dbgs() << "  Active intervals after expiration: "
                          << active.size() << "\n");
}

/// Updates the live range based on users of a value. Returns true if any
/// RegisterAllocationOpInterface users were found.
static bool updateLiveRangeFromUsers(const LiveRangeCache &cache, Value value,
                                     LiveRange &liveRange) {
  bool hasUser = false;
  for (auto *user : value.getUsers()) {
    if (!isa<rtg::RegisterAllocationOpInterface>(user))
      continue;

    // TODO: support labels and control-flow loops (jumps in general)
    unsigned idx = cache.opIndices.at(user);
    LLVM_DEBUG(llvm::dbgs()
               << "  User at index " << idx << ": " << *user << "\n");
    liveRange.start = std::min(liveRange.start, idx);
    liveRange.end = std::max(liveRange.end, idx);
    hasUser = true;
  }
  return hasUser;
}

/// Creates a live range for a virtual register operation. Returns nullptr
/// if the virtual register has no users that implement
/// RegisterAllocationOpInterface.
static void createRange(LiveRangeCache &cache, rtg::VirtualRegisterOp op) {
  LLVM_DEBUG(llvm::dbgs() << "Processing virtual register: " << op << "\n");

  auto &liveRange =
      cache.virtualRanges.emplace_back(op, cache.opIndices.size(), 0);

  SetVector<std::pair<Value, rtg::RegisterAttrInterface>> registers;
  auto res = collectTransitiveRegisterAllocationOperands(op.getResult(), {},
                                                         registers);
  if (!res.isAvailable())
    return;

  bool hasUser = updateLiveRangeFromUsers(cache, op.getResult(), liveRange);

  for (auto [reg, regAttr] : registers) {
    auto &depRange =
        liveRange.dependentRegs.emplace_back(reg, cache.opIndices.size(), 0);
    updateLiveRangeFromUsers(cache, reg, depRange);
    // TODO: instead of updating the virtual register range we should
    // cache/compute that differently
    hasUser |= updateLiveRangeFromUsers(cache, reg, liveRange);
  }

  if (!hasUser) {
    LLVM_DEBUG(llvm::dbgs()
               << "  No RegisterAllocationOpInterface users, removing range\n");
    cache.virtualRanges.pop_back();
  }

  LLVM_DEBUG(llvm::dbgs() << "  Live range: [" << liveRange.start << ", "
                          << liveRange.end << "]\n");
}

/// Computes live ranges for all virtual registers and identifies reserved
/// (fixed) registers.
static LogicalResult computeLiveRanges(LiveRangeCache &cache, Region &region) {
  LLVM_DEBUG(llvm::dbgs() << "\nCollecting register live ranges...\n");

  for (auto op : region.getOps<rtg::ConstantOp>()) {
    auto reg = dyn_cast<rtg::RegisterAttrInterface>(op.getValue());
    if (!reg)
      continue;

    unsigned maxIdx = 0, minIdx = cache.opIndices.size();
    for (auto *user : op->getUsers()) {
      minIdx = std::min(minIdx, cache.opIndices.at(user));
      maxIdx = std::max(maxIdx, cache.opIndices.at(user));
    }

    cache.reservedRanges.emplace_back(op.getLoc(), reg, minIdx, maxIdx);

    LLVM_DEBUG(llvm::dbgs()
               << "  Added fixed register to active list: " << reg << "\n");
  }

  llvm::sort(cache.reservedRanges,
             [](const FixedLiveRange &a, const FixedLiveRange &b) {
               return a.start < b.start;
             });

  for (auto op : region.getOps<rtg::VirtualRegisterOp>())
    createRange(cache, op);

  // Sort such that we can process registers by increasing interval start.
  LLVM_DEBUG(llvm::dbgs() << "\nSorting " << cache.virtualRanges.size()
                          << " register ranges by start time\n");
  llvm::sort(cache.virtualRanges, [](const VirtualLiveRange &a,
                                     const VirtualLiveRange &b) {
    return a.start < b.start || (a.start == b.start && a.end > b.end);
  });

  return success();
}

/// Checks if a specific register is available for allocation to a virtual
/// register. Verifies the register is not reserved, not in use by active
/// ranges, and satisfies all constraints.
static RegisterAllocationResult isRegisterAvailable(
    const LiveRangeCache &cache, ArrayRef<FixedLiveRange> active,
    rtg::VirtualRegisterOp virtualReg, rtg::RegisterAttrInterface reg,
    DenseMap<Value, rtg::RegisterAttrInterface> &dependentRegValues) {
  LLVM_DEBUG(llvm::dbgs() << "Trying register: " << reg << "\n");

  SetVector<std::pair<Value, rtg::RegisterAttrInterface>> registers;
  auto res = collectTransitiveRegisterAllocationOperands(virtualReg.getResult(),
                                                         reg, registers);
  if (!res.isAvailable())
    return res;

  LLVM_DEBUG(
      llvm::dbgs() << "Checking live range overlap with active registers\n");

  for (auto activeRange : active) {
    if (activeRange.fixedReg == reg)
      return RegisterAllocationResult::inUseBy(activeRange);

    for (auto [candidateReg, candidateRegAttr] : registers) {
      if (candidateRegAttr == activeRange.fixedReg)
        return RegisterAllocationResult::inUseBy(activeRange);
    }
  }

  for (auto [regVal, regAttr] : registers)
    dependentRegValues[regVal] = regAttr;

  LLVM_DEBUG(llvm::dbgs() << "Register: " << reg << " is available\n");

  return RegisterAllocationResult::available();
}

/// Finds an available register for the given virtual register from its
/// allowed register set. Returns an empty RegisterAttrInterface on failure.
static rtg::RegisterAttrInterface findAvailableRegister(
    const LiveRangeCache &cache, ArrayRef<FixedLiveRange> active,
    VirtualLiveRange &liveRange,
    DenseMap<Value, rtg::RegisterAttrInterface> &dependentRegValues) {
  auto virtualReg = liveRange.virtualReg;
  auto configAttr =
      cast<rtg::VirtualRegisterConfigAttr>(virtualReg.getAllowedRegsAttr());
  LLVM_DEBUG(llvm::dbgs() << "  Allowed registers: "
                          << configAttr.getAllowedRegs().size() << "\n");

  // Start a diagnostic, if we find a register, this diagnostic will be
  // discarded.
  auto diag = virtualReg.emitError(
      "no register available for allocation within constraints");
  if (auto *startOp = cache.indexToOp.lookup(liveRange.start))
    diag.attachNote(startOp->getLoc()) << "live range starts here";
  if (auto *endOp = cache.indexToOp.lookup(liveRange.end))
    diag.attachNote(endOp->getLoc()) << "live range ends here";

  // Try all registers allowed by the virtual register configuration in
  // decreasing order of preference.
  for (auto reg : configAttr.getAllowedRegs()) {
    dependentRegValues.clear();
    auto res =
        isRegisterAvailable(cache, active, virtualReg, reg, dependentRegValues);
    switch (res.getKind()) {
    case RegisterAllocationResult::Kind::Available:
      // If we found a valid register, use it without trying any other.
      diag.abandon();
      return reg;
    case RegisterAllocationResult::Kind::InUse:
      diag.attachNote(res.getUser().loc)
          << "cannot choose '" << reg.getRegisterAssembly()
          << "' because of overlapping live-range with this register";
      continue;
    case RegisterAllocationResult::Kind::ConstraintViolation:
      diag.attachNote(res.getConstraint().getLoc())
          << "constraint would be violated when choosing '"
          << reg.getRegisterAssembly() << "'";
      continue;
    case RegisterAllocationResult::Kind::FatalError:
      // Abandon since fatal errors are already reported when the error happend.
      diag.abandon();
      return {};
    }
  }

  return {};
}

/// Replaces all virtual register operations with their allocated fixed
/// registers and removes unused operations.
static void materializeAllocation(
    const LiveRangeCache &cache,
    const DenseMap<rtg::VirtualRegisterOp, rtg::RegisterAttrInterface>
        &assignedRegisters) {
  LLVM_DEBUG(llvm::dbgs()
             << "\nReplacing virtual registers with fixed registers...\n");

  circt::UnusedOpPruner operationPruner;
  for (auto &liveRange : cache.virtualRanges) {
    auto virtualReg = liveRange.virtualReg;
    auto fixedReg = assignedRegisters.at(virtualReg);

    LLVM_DEBUG(llvm::dbgs() << "Replacing virtual register " << virtualReg
                            << " with fixed register " << fixedReg << "\n");

    OpBuilder builder(virtualReg);
    auto newFixedReg =
        rtg::ConstantOp::create(builder, virtualReg.getLoc(), fixedReg);
    virtualReg.getResult().replaceAllUsesWith(newFixedReg);
    operationPruner.eraseNow(virtualReg);
  }

  // Erase any operations that became unused after erasing the virtual register
  // operations
  operationPruner.eraseNow();

  LLVM_DEBUG(llvm::dbgs() << "Register allocation completed successfully!\n");
}

/// Performs the main register allocation using linear scan algorithm.
/// Processes live ranges in order, expires old intervals, and assigns
/// registers to virtual registers.
static LogicalResult allocateVirtualRegistersInCache(LiveRangeCache &cache) {
  SmallVector<FixedLiveRange> active;
  DenseMap<rtg::VirtualRegisterOp, rtg::RegisterAttrInterface>
      assignedRegisters;

  // TODO: would be better to only add them to active once their range actually
  // starts
  for (auto &fixedRange : cache.reservedRanges)
    active.push_back(fixedRange);

  for (auto &virtualRange : cache.virtualRanges) {
    LLVM_DEBUG(llvm::dbgs() << "Processing register range ["
                            << virtualRange.start << ", " << virtualRange.end
                            << "] for " << virtualRange.virtualReg << "\n");

    // Make registers out of live range available again.
    expireOldInterval(active, virtualRange);

    DenseMap<Value, rtg::RegisterAttrInterface> dependentRegValues;
    auto availableReg =
        findAvailableRegister(cache, active, virtualRange, dependentRegValues);
    if (!availableReg)
      return failure();

    assignedRegisters[virtualRange.virtualReg] = availableReg;

    LLVM_DEBUG(llvm::dbgs() << "  Assigned register " << availableReg
                            << " to virtual register\n");

    active.emplace_back(virtualRange.loc, availableReg, virtualRange.start,
                        virtualRange.end);

    for (auto &dependentRange : virtualRange.dependentRegs)
      active.emplace_back(dependentRange.loc,
                          dependentRegValues.at(dependentRange.value),
                          dependentRange.start, dependentRange.end);
  }

  LLVM_DEBUG({
    for (auto &virtualRange : cache.virtualRanges) {
      llvm::dbgs() << "Start: " << virtualRange.start
                   << ", End: " << virtualRange.end << ", Selected: "
                   << assignedRegisters.at(virtualRange.virtualReg) << "\n";
    }
    llvm::dbgs() << "\n";
  });

  materializeAllocation(cache, assignedRegisters);
  return success();
}

//===----------------------------------------------------------------------===//
// Class implementations
//===----------------------------------------------------------------------===//

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

  LiveRangeCache cache;
  for (auto segOp : getOperation()->getRegion(0).getOps<rtg::SegmentOp>()) {
    // Perform register allocation for all text segments in isolation. There
    // should usually only be one such segment.
    if (segOp.getKind() == rtg::SegmentKind::Text) {
      if (failed(cache.populate(segOp.getRegion())))
        return signalPassFailure();

      if (failed(allocateVirtualRegistersInCache(cache)))
        return signalPassFailure();

      cache.clear();
    }
  }
}

LogicalResult LiveRangeCache::populate(Region &region) {
  for (auto [i, op] : llvm::enumerate(region.front())) {
    // TODO: ideally check that the IR is already fully elaborated
    opIndices[&op] = i;
    indexToOp[i] = &op;
    LLVM_DEBUG(llvm::dbgs() << "  Op " << i << ": " << op << "\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "Total operations in text segment: "
                          << opIndices.size() << "\n");

  return computeLiveRanges(*this, region);
}

void LiveRangeCache::clear() {
  opIndices.clear();
  indexToOp.clear();
  virtualRanges.clear();
  reservedRanges.clear();
}
