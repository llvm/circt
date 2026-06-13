#include "circt/Dialect/Resource/Passes/BRAMAnalysisLoopSchedule.h"
#include "circt/Dialect/Resource/Interfaces/MemoryInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <map>
#include <string>


#define DEBUG_TYPE "hls-analysis-loop-schedule"

using namespace mlir;

namespace {

constexpr int64_t kBitsPerBRAM = 16 * 1024;

struct PortDemand {
  unsigned reads = 0;
  unsigned writes = 0;
  unsigned total = 0;
};

//===----------------------------------------------------------------------===//
// Port demand: per temporal scope, then combined across scopes.
//===----------------------------------------------------------------------===//
// Peak demand within ONE loop. A pipelined scope (ii>=1) overlaps iterations,
// so accesses contend when they share slot (stageStart % ii). A sequential
// scope (ii==0) issues at most one memory op per cycle, so reads and writes
// never collide regardless of how many there are.
static PortDemand scopeDemand(ArrayRef<const hls::AccessSummary *> group) {
  PortDemand d;
  if (group.empty())
    return d;

  unsigned ii = group.front()->ii;
  assert(llvm::all_of(group, [&](auto *a) { return a->ii == ii; }) &&
         "scopeDemand requires a single-II group; bucket by scope first");

  // Sequential scope: body runs to completion before the next iteration, so
  // there is no cross-iteration overlap and at most one access is in flight
  // regardless of its latency. Latency does not create contention here.
  if (ii == 0) {
    bool anyR = false, anyW = false;
    for (const auto *a : group)
      (a->isWrite ? anyW : anyR) = true;
    d.reads  = anyR ? 1u : 0u;
    d.writes = anyW ? 1u : 0u;
    d.total  = (anyR || anyW) ? 1u : 0u;
    return d;
  }

  // Pipelined scope: an access starting at `stageStart` and occupying its
  // port for `latency` cycles reserves phases stageStart, stageStart+1, ...,
  // stageStart+latency-1, each taken mod II. Looping over the cycles handles
  // wraparound correctly, including self-overlap when latency >= II.
  std::vector<std::pair<unsigned, unsigned>> slot(ii, {0u, 0u}); // {reads,writes}
  for (const auto *a : group) {
    unsigned occ = std::max(a->latency, 1u);   // 0-latency access still uses port once
    for (unsigned c = 0; c < occ; ++c) {
      unsigned phase = (a->stageStart + c) % ii;
      (a->isWrite ? slot[phase].second : slot[phase].first)++;
    }
  }

  for (const auto &cell : slot) {
    unsigned r = cell.first, w = cell.second;
    d.reads  = std::max(d.reads, r);
    d.writes = std::max(d.writes, w);
    d.total  = std::max(d.total, r + w);
  }
  return d;
}

// Combine per-loop demand into a single requirement for the shared BRAM.
//   sequential loops  -> element-wise MAX (the array is time-shared)
//   dataflow region   -> SUM (loops run concurrently; caller also applies the
//                         ping-pong x2 capacity multiplier)
PortDemand computePortDemand(ArrayRef<hls::AccessSummary> accesses,
                             bool concurrentScopes) {
  std::map<Operation *, SmallVector<const hls::AccessSummary *>> byScope;

  for (const auto &a : accesses)
    byScope[a.scope].push_back(&a);

  PortDemand combined;
  for (const auto &kv : byScope) {
    PortDemand local = scopeDemand(kv.second);
    if (concurrentScopes) {
      combined.reads += local.reads;
      combined.writes += local.writes;
      combined.total += local.total;
    } else {
      combined.reads = std::max(combined.reads, local.reads);
      combined.writes = std::max(combined.writes, local.writes);
      combined.total = std::max(combined.total, local.total);
    }
  }
  return combined;
}

//===----------------------------------------------------------------------===//
// Storage-kind inference (UG1399 bind_storage port semantics).
//===----------------------------------------------------------------------===//
struct KindResult {
  hls::StorageKind kind = hls::StorageKind::RAM_1P;
  unsigned banks = 1; // RAM_1WNR only
  bool infeasible = false;
  std::string note;
};

KindResult inferStorageKind(const PortDemand &d) {
  KindResult r;
  if (d.writes > 2) {
    r.kind = hls::StorageKind::RAM_T2P;
    r.infeasible = true;
    r.note = "write demand " + std::to_string(d.writes) +
             "/cycle exceeds 2 write ports; requires array_partition.";
    return r;
  }
  if (d.total <= 1)
    r.kind = hls::StorageKind::RAM_1P;
  else
    r.kind = hls::StorageKind::RAM_2P;        // 1R + 1W
  return r;
}

unsigned bramMultiplier(const KindResult &k) {
  switch (k.kind) {
  case hls::StorageKind::RAM_1P:   return 1;
  case hls::StorageKind::RAM_S2P:  return 1; // SDP keeps full width
  case hls::StorageKind::RAM_2P:   return 2; // TDP halves width
  case hls::StorageKind::RAM_T2P:  return 2; // TDP halves width
  case hls::StorageKind::RAM_1WNR: return std::max(k.banks, 1u);
  case hls::StorageKind::LUTRAM:   return 0;
  case hls::StorageKind::URAM:     return 0;
  }
  return 1;
}

//===----------------------------------------------------------------------===//
// Capacity count: bits -> BRAMs for ONE single-port copy (HAPE BRE).
//===----------------------------------------------------------------------===//
int64_t breCapacity(hls::MemoryResourceOpInterface iface,
                    const hls::PartitionSpec *part) {
  auto shape = iface.getStaticShape();
  if (shape.empty())
    return 0;

  int64_t SA = 1;
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim))
      return 0;
    SA *= dim;
  }
  unsigned Bits = iface.getElementType().getIntOrFloatBitWidth();

  if (part && part->kind == hls::PartitionSpec::Complete)
    return 0;

  int64_t Pf = part ? static_cast<int64_t>(part->factor) : 1;
  if (Pf < 1)
    Pf = 1;

  double BRAMb =
      static_cast<double>(SA * Bits) / static_cast<double>(Pf * kBitsPerBRAM);
  int64_t R = static_cast<int64_t>(std::round(BRAMb));
  if (R == 0)
    return 0;
  int64_t BRAMs = Pf * R;
  if (!llvm::isPowerOf2_64(static_cast<uint64_t>(BRAMs))) {
    double l = std::log2(static_cast<double>(BRAMs));
    BRAMs = 1LL << static_cast<int64_t>(std::llround(l));
  }
  return BRAMs;
}

//===----------------------------------------------------------------------===//
// Dataflow detection hook. Returns true if the accessing loops are scheduled
// concurrently (e.g. enclosed in a dataflow region), which makes demands add
// and the array a double-buffered channel. Stubbed to false until the IR
// carries a dataflow marker; this is the entry point for your DialectInterface
// adjustCost / double-buffer multiplier.
bool accessesAreConcurrent(hls::MemoryResourceOpInterface /*iface*/) {
  return false;
}

//===----------------------------------------------------------------------===//
// Top-level estimator.
//===----------------------------------------------------------------------===//
hls::BRAMCost estimateBRAM(hls::MemoryResourceOpInterface iface) {
  hls::BRAMCost cost;
  cost.kind = hls::StorageKind::RAM_1P;
  cost.bramCount = 0;

  if (!iface.contributesToBRAMCost()) {
    cost.note = "excluded from BRAM accounting (LUTRAM / stream / zero-size)";
    return cost;
  }

  bool concurrent = accessesAreConcurrent(iface);
  PortDemand demand = computePortDemand(iface.getAccessSummary(), concurrent);

  std::optional<hls::PartitionSpec> partOpt = iface.getPartitionSpec();
  const hls::PartitionSpec *part = partOpt ? &*partOpt : nullptr;

  int64_t capacity = breCapacity(iface, part);
  if (capacity == 0) {
    cost.note = "fits in registers/LUTRAM "
                "(sub-BRAM threshold or complete partition)";
    return cost;
  }

  KindResult kr;
  if (auto pragmaKind = iface.getPragmaStorageKind()) {
    kr.kind = *pragmaKind;
    kr.banks = std::max(demand.reads, 1u);
  } else {
    kr = inferStorageKind(demand);
  }

  unsigned mult = bramMultiplier(kr);
  // Concurrent (dataflow) shared arrays are double-buffered.
  unsigned buffers = concurrent ? 2u : 1u;

  cost.kind = kr.kind;
  cost.infeasible = kr.infeasible;
  cost.bramCount = static_cast<unsigned>(capacity * mult * buffers);
  cost.note = kr.note.empty()
                  ? ("capacity=" + std::to_string(capacity) +
                     " x mult=" + std::to_string(mult) +
                     (concurrent ? " x2 (dataflow ping-pong)" : "") +
                     " across " + std::to_string(0) + " scopes")
                  : kr.note;
  return cost;
}

//===----------------------------------------------------------------------===//
// Pass.
//===----------------------------------------------------------------------===//
const char *kindName(hls::StorageKind k) {
  switch (k) {
  case hls::StorageKind::RAM_1P:   return "RAM_1P";
  case hls::StorageKind::RAM_2P:   return "RAM_2P";
  case hls::StorageKind::RAM_S2P:  return "RAM_S2P";
  case hls::StorageKind::RAM_T2P:  return "RAM_T2P";
  case hls::StorageKind::LUTRAM:   return "LUTRAM";
  case hls::StorageKind::RAM_1WNR: return "RAM_1WNR";
  case hls::StorageKind::URAM:     return "URAM";
  }
  return "?";
}

struct BRAMAnalysisLoopSchedule
    : public PassWrapper<BRAMAnalysisLoopSchedule, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BRAMAnalysisLoopSchedule)

  StringRef getArgument() const final { return "bram-analysis"; }
  StringRef getDescription() const final {
    return "Estimate BRAM resources for HLS allocations.";
  }
  

  void runOnOperation() override {
    int64_t total = 0;
    getOperation().walk([&](Operation *op) {
      auto iface = dyn_cast<hls::MemoryResourceOpInterface>(op);
      if (!iface)
        return;
      // how much each op contributes to 
      hls::BRAMCost cost = estimateBRAM(iface);
      LLVM_DEBUG({
          llvm::outs() << op->getName() << " @ " << op->getLoc() << "\n"
                       << " kind=" << kindName(cost.kind)
                       << "  bram=" << cost.bramCount
                       << (cost.infeasible ? "  [INFEASIBLE]" : "") << "\n"
                       << "    " << cost.note << "\n";
      });
      total += cost.bramCount;
    });
    llvm::outs() << "Total BRAM: " << total << "\n";
  }
};

} // namespace

std::unique_ptr<Pass> hls::createBRAMAnalysisPass() {
  return std::make_unique<BRAMAnalysisLoopSchedule>();
}

void hls::registerBRAMLoopscheduleAnalysisPass() {
  PassRegistration<BRAMAnalysisLoopSchedule>();
}
