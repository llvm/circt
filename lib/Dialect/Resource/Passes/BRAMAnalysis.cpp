#include "circt/Dialect/Resource/Passes/BRAMAnalysis.h"
#include "circt/Dialect/Resource/Interfaces/MemoryInterface.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#define DEBUG_TYPE "hls-analysis-loop-schedule"

using namespace mlir;
using circt::loopschedule::LoopSchedulePipelineOp;
using circt::loopschedule::LoopSchedulePipelineStageOp;
using namespace circt::hls_analysis;


namespace {

//===----------------------------------------------------------------------===//
// Device model (Zynq-7020 / 7-series, UG473).
//===----------------------------------------------------------------------===//
// One BRAM_18K block holds 16*1024 = 16384 *data* bits (parity excluded; this is
// the S_bram value, NOT 18*1024). A RAMB36E1 is just two of these tiled, so we
// count everything in BRAM_18K units and let width/depth tiling synthesize the
// 36K cases.
//
// The per-block *data width* depends on how many ports must be live at once:
//
//     primitive / mode        gross width   data width   "# of 9b bytes"
//     RAMB18E1 TDP (2P/T2P)       18            16             2
//     RAMB18E1 1P / SDP (S2P)     36            32             4
//     RAMB36E1 TDP                36            32             4   (= 2x 18K)
//     RAMB36E1 SDP               72            64             8   (= 2x 18K)
//
// The x16-per-port limit is a *True-Dual-Port* effect: it applies only when the
// array needs two ports live in the SAME cycle (1R+1W read-modify, 2R, or 2W),
// which forces TDP where each port maxes at x18. With only one live port there
// is nothing to force TDP, so the block is used in its wide x36 config and a
// 512x32 array fits one 18K block -- this is why a time-shared single-port array
// (RAM_1P) is 1 BRAM, not 2. A simple-dual-port (RAM_S2P: one port read-only,
// one write-only) is likewise wide x36. So wide x32 == {RAM_1P, RAM_S2P};
// narrow x16 == {RAM_2P, RAM_T2P}. That x16 -> x32 difference is the whole
// story: at x16 a 32-bit element needs two width tiles (one RAMB36E1); at x32 it
// fits one 18K block. An 8/16-bit element fits x16, so contention never doubles
// it -- which is exactly the bug in the old flat "RAM_2P -> x2".
constexpr int64_t kBitsPerBRAM = 16 * 1024;

struct PortDemand {
  unsigned reads = 0;
  unsigned writes = 0;
  unsigned total = 0;
  // True iff reads and writes are partitioned across scopes -- no single scope
  // both reads and writes this bank, and at least one scope of each exists. This
  // is the structural precondition for simple-dual-port (one port read-only, the
  // other write-only); only meaningful on the combined demand.
  bool segregated = false;
};


//===----------------------------------------------------------------------===//
// Port demand: per temporal scope, then combined across scopes.
//===----------------------------------------------------------------------===//
// Peak demand within ONE loop. A pipelined scope (ii>=1) overlaps iterations, so
// accesses contend when they share a slot (stageStart % ii). A sequential scope
// (ii==0) issues at most one memory op per cycle, so reads and writes never
// collide regardless of how many there are.
PortDemand scopeDemand(ArrayRef<const AccessSummary *> group) {
  PortDemand d;
  if (group.empty())
    return d;

  unsigned ii = group.front()->ii;
  assert(llvm::all_of(group, [&](auto *a) { return a->ii == ii; }) &&
         "scopeDemand requires a single-II group; bucket by scope first");

  // Sequential scope: the body runs to completion before the next iteration, so
  // there is no cross-iteration overlap and at most one access is in flight
  // regardless of latency.
  if (ii == 0) {
    bool anyR = false, anyW = false;
    for (const auto *a : group)
      (a->isWrite ? anyW : anyR) = true;
    d.reads = anyR ? 1u : 0u;
    d.writes = anyW ? 1u : 0u;
    d.total = (anyR || anyW) ? 1u : 0u;
    return d;
  }

  // Pipelined scope: an access starting at `stageStart` and holding its port for
  // `latency` cycles reserves phases stageStart .. stageStart+latency-1, each
  // taken mod II. Looping the cycles handles wraparound, including self-overlap
  // when latency >= II.
  std::vector<std::pair<unsigned, unsigned>> slot(ii, {0u, 0u}); // {reads,writes}
  for (const auto *a : group) {
    unsigned occ = std::max(a->latency, 1u); // 0-latency access still uses a port
    for (unsigned c = 0; c < occ; ++c) {
      unsigned phase = (a->stageStart + c) % ii;
      (a->isWrite ? slot[phase].second : slot[phase].first)++;
    }
  }

  for (const auto &cell : slot) {
    unsigned r = cell.first, w = cell.second;
    d.reads = std::max(d.reads, r);
    d.writes = std::max(d.writes, w);
    d.total = std::max(d.total, r + w);
  }
  
  return d;
}

// Combine per-loop demand into a single requirement for the shared bank.
//   sequential loops  -> element-wise MAX (the array is time-shared)
static PortDemand computePortDemand(ArrayRef<AccessSummary> accesses) {
  std::map<Operation *, SmallVector<const AccessSummary *>> byScope;
  for (const auto &a : accesses)
    byScope[a.scope].push_back(&a);

  PortDemand combined;
  bool anyRead = false, anyWrite = false, anyReadModifyScope = false;
  for (const auto &kv : byScope) {
    PortDemand local = scopeDemand(kv.second);
    anyRead |= local.reads > 0;
    anyWrite |= local.writes > 0;
    anyReadModifyScope |= (local.reads > 0 && local.writes > 0);
    combined.reads = std::max(combined.reads, local.reads);
    combined.writes = std::max(combined.writes, local.writes);
    combined.total = std::max(combined.total, local.total);
    
  }
  combined.segregated = anyRead && anyWrite && !anyReadModifyScope;
  return combined;
}

//===----------------------------------------------------------------------===//
// Kind naming (declared early; used in notes).
//===----------------------------------------------------------------------===//
const char *kindName(StorageKind k) {
  switch (k) {
  case StorageKind::RAM_1P:   return "RAM_1P";
  case StorageKind::RAM_2P:   return "RAM_2P";
  case StorageKind::RAM_S2P:  return "RAM_S2P";
  case StorageKind::RAM_T2P:  return "RAM_T2P";
  case StorageKind::RAM_1WNR: return "RAM_1WNR";
  case StorageKind::LUTRAM:   return "LUTRAM";
  case StorageKind::URAM:     return "URAM";
  }
  return "?";
}

//===----------------------------------------------------------------------===//
// Storage-kind inference from port demand (replaces the old flat multiplier).
//===----------------------------------------------------------------------===//
struct KindResult {
  StorageKind kind = StorageKind::RAM_1P;
  bool infeasible = false; // demand exceeds the 2 physical ports of one block
  std::string note;
};

// A single BRAM primitive has at most two ports. Map the per-cycle access
// pattern to the narrowest mode that serves it; that mode then fixes the
// per-block data width in blockDataWidth(). Note this is count-only and never
// returns RAM_S2P: from counts a read-modify array (narrow RAM_2P) and a
// segregated producer/consumer (wide RAM_S2P) both look like 1R+1W. The wide
// case is decided structurally by the caller (concurrent + segregated) or via
// an explicit bind_storage pragma.
static KindResult selectStorageKind(const PortDemand &d) {
  KindResult r;
  unsigned total = d.total, w = d.writes, rd = d.reads;

  if (total <= 1) {
    r.kind = StorageKind::RAM_1P;          // one live port, wide x32
    return r;
  }
  if (total == 2) {
    r.kind = (rd == 1 && w == 1) ? StorageKind::RAM_2P
                                 : StorageKind::RAM_T2P;
    return r;
  }
  
  r.kind = StorageKind::RAM_T2P;
  r.infeasible = true;
  r.note = "port demand " + std::to_string(total) + "/cycle (" +
           std::to_string(rd) + "R+" + std::to_string(w) +
           "W, multi-writer) exceeds 2 ports; needs array_partition.";
  return r;
}

//===----------------------------------------------------------------------===//
// Width-aware block count.
//===----------------------------------------------------------------------===//
// Data bits one BRAM_18K block can present per cycle for the given kind.
unsigned blockDataWidth(StorageKind k) {
  switch (k) {
  case StorageKind::RAM_1P:   // one live port -> no TDP constraint -> wide
  case StorageKind::RAM_S2P:  // simple dual port: dedicated R + W -> wide
    return 32;
  case StorageKind::RAM_2P:   // two live ports (read-modify) -> TDP x18
  case StorageKind::RAM_T2P:  // two live ports -> TDP x18
  case StorageKind::RAM_1WNR:
    return 16;
  case StorageKind::LUTRAM:
  case StorageKind::URAM:
    return 0; // not mapped to BRAM
  }
  return 16;
}

int64_t ceilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Round a data width up to a width the block can be configured at (1,2,4,8,16,
// 32...). A 24-bit element occupies an x32 slice and a 12-bit element an x16
// slice; that rounding is what wastes BRAM depth for non-power-of-two widths.
unsigned configWidth(unsigned w) {
  return w <= 1 ? 1u : static_cast<unsigned>(llvm::PowerOf2Ceil(w));
}

// BRAM_18K blocks needed for numElems x elemBits in the chosen kind.
// Width tiling handles elements wider than the per-block bus (the x16 -> x32
// promotion); depth tiling handles arrays deeper than one block at that width.
int64_t estimateBlocks(int64_t numElems, unsigned elemBits,
                              StorageKind kind) {
  unsigned wmax = blockDataWidth(kind);
  if (wmax == 0 || numElems == 0 || elemBits == 0)
    return 0;

  int64_t widthTiles;
  unsigned cfgW;
  if (elemBits <= wmax) {
    widthTiles = 1;
    cfgW = std::min(configWidth(elemBits), wmax); // narrower elements pack deeper
  } else {
    widthTiles = ceilDiv(elemBits, wmax); // promote: each tile is a full x-wmax block
    cfgW = wmax;
  }
  int64_t depthPerBlock = std::max<int64_t>(kBitsPerBRAM / cfgW, 1);
  int64_t depthTiles = ceilDiv(numElems, depthPerBlock);
  return widthTiles * depthTiles;
}

//===----------------------------------------------------------------------===//
// Access-summary collection (replaces the removed getAccessSummary interface
// method). The memref interface stays dialect-neutral; this loopschedule-aware
// analysis lives in the pass, where reaching into pipeline/stage timing is fair
// game. It reconstructs, per access to this bank:
//   isWrite     - from the op's memory effect on the bank value (dialect-neutral)
//   scope       - the enclosing pipeline (pipelined) or loop (sequential); only
//                 accesses sharing a scope contend for ports
//   ii          - the pipeline II, or 0 for a sequential scope
//   stageStart  - the enclosing stage's `start` cycle (the port-issue phase)
//   latency     - port *occupancy* in cycles; a BRAM address port is held for
//                 one cycle per access regardless of read latency
//===----------------------------------------------------------------------===//

// BRAM address-port occupancy. A synchronous read presents its address for a
// single cycle (data returns later without re-using the port), so one cycle is
// the right contention footprint. Exposed as a hook for multi-cycle/wide ops.
unsigned portOccupancy(bool /*isWrite*/) { return 1u; }

// Classify an op's effect on `mem`. Works for memref.load/store, affine
// load/store, or any op exposing memory effects; non-accessing aliasing ops
// (subview, cast, dealloc) report no read/write effect and are skipped.
bool classifyAccess(Operation *op, Value mem, bool &isWrite) {
  auto eff = dyn_cast<MemoryEffectOpInterface>(op);
  if (!eff)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  eff.getEffects(effects);
  bool reads = false, writes = false;
  for (auto &e : effects) {
    if (e.getValue() && e.getValue() != mem)
      continue;
    if (isa<MemoryEffects::Read>(e.getEffect()))
      reads = true;
    else if (isa<MemoryEffects::Write>(e.getEffect()))
      writes = true;
  }
  if (!reads && !writes)
    return false;
  isWrite = writes; // a read-modify-write op needs a write port
  return true;
}

// Nearest enclosing loop for a sequential (non-pipelined) access. Uses the
// generic LoopLikeOpInterface so scf.for / affine.for / custom loops all work
// without hard-coding a dialect; falls back to the enclosing function so that
// straight-line accesses still share one scope.
Operation *sequentialScope(Operation *op) {
  for (Operation *cur = op->getParentOp(); cur; cur = cur->getParentOp())
    if (isa<LoopLikeOpInterface>(cur))
      return cur;
  if (auto fn = op->getParentOfType<func::FuncOp>())
    return fn.getOperation();
  return nullptr;
}


SmallVector<AccessSummary>
collectAccessSummaries(Operation *allocOp) {
  SmallVector<AccessSummary> out;
  if (allocOp->getNumResults() == 0)
    return out;
  Value mem = allocOp->getResult(0); // the bank's memref

  for (Operation *user : mem.getUsers()) {
    bool isWrite = false;
    if (!classifyAccess(user, mem, isWrite))
      continue;

    AccessSummary a;
    a.op = user;
    a.isWrite = isWrite;
    a.latency = portOccupancy(isWrite);

    if (auto pipeline = user->getParentOfType<LoopSchedulePipelineOp>()) {
      auto stage = user->getParentOfType<LoopSchedulePipelineStageOp>();
      a.scope = pipeline.getOperation();
      a.ii = static_cast<unsigned>(pipeline.getII());
      a.stageStart =
          stage ? static_cast<unsigned>(stage.getStart()) : 0u;
    } else {
      // Sequential: one iteration in flight, so ii=0 collapses the scope to at
      // most one access per cycle in scopeDemand().
      a.scope = sequentialScope(user);
      a.ii = 0;
      a.stageStart = 0;
    }
    out.push_back(a);
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Top-level estimator.
//===----------------------------------------------------------------------===//
// Post-banking the memref IS one bank: no partition factor, no affine bank-id
// analysis. Geometry comes straight off the shape.
static BRAMCost estimateBRAM(MemoryResourceOpInterface iface) {
  BRAMCost cost;
  cost.kind = StorageKind::RAM_1P;
  cost.bramCount = 0;

  if (!iface.contributesToBRAMCost()) {
    cost.note = "excluded from BRAM accounting (LUTRAM / stream / zero-size)";
    return cost;
  }

  // --- Bank geometry ---
  auto shape = iface.getStaticShape();
  int64_t numElems = 1;
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim)) {
      cost.note = "dynamic shape; not estimated";
      return cost;
    }
    numElems *= dim;
  }
  unsigned elemBits = iface.getElementType().getIntOrFloatBitWidth();
  if (numElems == 0 || elemBits == 0) {
    cost.note = "zero-size";
    return cost;
  }
  // --- Port contention -> storage kind ---
  SmallVector<AccessSummary> accesses =
      collectAccessSummaries(iface.getOperation());
  PortDemand demand = computePortDemand(accesses);
  KindResult kr = selectStorageKind(demand);
  // --- Width-aware block count ---
  int64_t blocks = estimateBlocks(numElems, elemBits, kr.kind);
  cost.kind = kr.kind;
  cost.infeasible = kr.infeasible;
  cost.bramCount = static_cast<unsigned>(blocks);
  cost.note =
      !kr.note.empty()
          ? kr.note + " => " + std::to_string(blocks) + " BRAM"
          : (std::to_string(numElems) + "x" + std::to_string(elemBits) +
             "b -> " + kindName(kr.kind) + " (x" +
             std::to_string(blockDataWidth(kr.kind)) + "/block) = " +
             std::to_string(blocks) + " BRAM_18K");
  return cost;
}

//===----------------------------------------------------------------------===//
// Pass.
//===----------------------------------------------------------------------===//
struct BRAMAnalysisLoopSchedule
    : PassWrapper<BRAMAnalysisLoopSchedule, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BRAMAnalysisLoopSchedule)

  StringRef getArgument() const final { return "bram-analysis"; }
  StringRef getDescription() const final {
    return "Estimate BRAM resources for HLS allocations (post-banking).";
  }

  void runOnOperation() override {
    int64_t total = 0;
    getOperation().walk([&](Operation *op) {
      auto iface = dyn_cast<MemoryResourceOpInterface>(op);
      if (!iface)
        return;
      BRAMCost cost = estimateBRAM(iface);
      llvm::outs() << op->getName() << " @ " << op->getLoc() << "\n"
                   << "  kind=" << kindName(cost.kind)
                   << "  bram=" << cost.bramCount
                   << (cost.infeasible ? "  [INFEASIBLE]" : "") << "\n"
                   << "    " << cost.note << "\n";
      total += cost.bramCount;
    });
    llvm::outs() << "Total BRAM: " << total << "\n";
  }
};

} // namespace

namespace circt::hls_analysis {

std::unique_ptr<Pass> createBRAMAnalysisPass() {
  return std::make_unique<BRAMAnalysisLoopSchedule>();
}

void registerBRAMLoopscheduleAnalysisPass() {
  PassRegistration<BRAMAnalysisLoopSchedule>();
}
}