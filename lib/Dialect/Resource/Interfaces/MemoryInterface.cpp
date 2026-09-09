//===----------------------------------------------------------------------===//
// MemoryResourceOpInterface external models.
//
// Attaches the BRAM-costing interface to the memref ops that actually declare
// storage:
//   - memref.alloca : function-local array
//   - memref.global : module-level array (materialized per memref.get_global)
//
// memref.get_global is deliberately NOT modeled: it is a use of a global, not
// a declaration of storage, and attaching there would count one array once per
// use site.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "circt/Dialect/Resource/Interfaces/MemoryInterface.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"

using namespace mlir;
using circt::hls_analysis::PartitionSpec;
using circt::hls_analysis::StorageKind;

// OpInterface dispatch tables. Include exactly once in the project.
#include "circt/Dialect/Resource/Interfaces/MemoryOpInterface.cpp.inc"

unsigned getStageStart(Operation *user) {
  if (auto stage =
          user->getParentOfType<circt::loopschedule::LoopSchedulePipelineStageOp>())
    return stage.getStart();
  return 0; // not inside a pipeline stage -> treat as cycle 0
}

//===----------------------------------------------------------------------===//
// Shared implementation.
//
// Every storage op answers these the same way; only the two geometry accessors
// (element type, static shape) differ, since one reads a result type and the
// other a symbol's type attribute. Kept as free functions rather than a CRTP
// base so the models stay plain and the logic has exactly one home.
//===----------------------------------------------------------------------===//
namespace {

int64_t computeSizeInBits(ArrayRef<int64_t> shape, Type elt) {
  int64_t n = 1;
  for (int64_t d : shape) {
    if (ShapedType::isDynamic(d))
      return 0;
    n *= d;
  }
  unsigned w = 0;
  if (elt.isIntOrFloat())
    w = elt.getIntOrFloatBitWidth();
  else if (isa<IndexType>(elt))
    w = 64; // platform-dependent; this is the convention we assume
  return n * static_cast<int64_t>(w);
}

/// Parse an `hls.array_partition` attribute into a PartitionSpec.
///
/// Only the first entry is consulted. Multi-dimensional partitioning would
/// need the caller to combine several specs; until that exists, taking the
/// first is preferable to silently merging them.
std::optional<PartitionSpec> parsePartitionSpec(Operation *op) {
  auto arr = op->getAttrOfType<ArrayAttr>("hls.array_partition");
  if (!arr || arr.empty())
    return std::nullopt; // no pragma -> unpartitioned, caller treats as Pf=1

  auto dict = dyn_cast<DictionaryAttr>(arr[0]);
  if (!dict)
    return std::nullopt;

  PartitionSpec spec;

  // kind (required to disambiguate the enum)
  auto kindAttr = dict.getAs<StringAttr>("kind");
  if (!kindAttr)
    return std::nullopt;
  StringRef k = kindAttr.getValue();
  if (k == "cyclic")
    spec.kind = PartitionSpec::Cyclic;
  else if (k == "block")
    spec.kind = PartitionSpec::Block;
  else if (k == "complete")
    spec.kind = PartitionSpec::Complete;
  else
    return std::nullopt; // unknown kind -> don't guess

  // factor (ignored for Complete, but parse defensively)
  if (auto f = dict.getAs<IntegerAttr>("factor"))
    spec.factor = static_cast<unsigned>(f.getInt());
  else
    spec.factor = 1;

  // dim -- Vitis pragma dims are 1-indexed; PartitionSpec::dim is 0-indexed.
  if (auto d = dict.getAs<IntegerAttr>("dim")) {
    int64_t pragmaDim = d.getInt();
    spec.dim = pragmaDim > 0 ? static_cast<unsigned>(pragmaDim - 1) : 0;
  } else {
    spec.dim = 0;
  }

  return spec;
}

/// Whether an array of `sizeInBits` is large enough that a tool would map it
/// to block RAM rather than leaving it in registers/LUTRAM.
///
/// TODO: most tools have a configurable threshold for BRAM promotion. This
///       number is a placeholder; make it parameterizable, either in a .td
///       file or from a device/target specification.
bool meetsBRAMPromotionThreshold(int64_t sizeInBits) {
  // Assume 32 entries x 32-bit data as the promotion point for now.
  constexpr int64_t kBRAMPromotionBits = 32 * 32;
  return sizeInBits > kBRAMPromotionBits;
}

/// Shared cost opt-out: an allocation explicitly annotated as not BRAM-backed
/// (registers, FIFO, stream) is excluded, as is anything below the promotion
/// threshold.
bool contributesToBRAMCostImpl(std::optional<StorageKind> pragmaKind,
                               int64_t sizeInBits) {
  if (pragmaKind && *pragmaKind == StorageKind::LUTRAM)
    return false;
  // URAM goes through a different resource pool; whether it is counted here
  // is a tool-level policy decision, left to the caller for now.
  return meetsBRAMPromotionThreshold(sizeInBits);
}

} // namespace


//===----------------------------------------------------------------------===//
// memref.alloca
//===----------------------------------------------------------------------===//
struct MemRefAllocaExternalModel
    : circt::hls_analysis::MemoryResourceOpInterface::ExternalModel
          <MemRefAllocaExternalModel, memref::AllocaOp> {

  Type getElementType(Operation *op) const {
    return cast<memref::AllocaOp>(op).getType().getElementType();
  }

  SmallVector<int64_t> getStaticShape(Operation *op) const {
    return llvm::to_vector(cast<memref::AllocaOp>(op).getType().getShape());
  }

  int64_t getStaticSizeInBits(Operation *op) const {
    return computeSizeInBits(getStaticShape(op), getElementType(op));
  }

  std::optional<StorageKind> getPragmaStorageKind(Operation *op) const {
    // Stub; add later (e.g. hls.bind_storage = ...).
    return std::nullopt;
  }

  std::optional<PartitionSpec> getPartitionSpec(Operation *op) const {
    return parsePartitionSpec(op);
  }

  std::optional<int64_t> getEnclosingPipelineII(Operation *op) const {
    // Stub; add later.
    return std::nullopt;
  }

  bool contributesToBRAMCost(Operation *op) const {
    return contributesToBRAMCostImpl(getPragmaStorageKind(op),
                                     getStaticSizeInBits(op));
  }
};

//===----------------------------------------------------------------------===//
// memref.global
//
// Geometry comes from the symbol's type attribute rather than a result, since
// a global declares storage without producing an SSA value. Pragma attributes
// are read from the global declaration itself, which is where a front end
// would attach them.
//===----------------------------------------------------------------------===//
struct MemRefGlobalExternalModel
    : circt::hls_analysis::MemoryResourceOpInterface::ExternalModel
          <MemRefGlobalExternalModel, memref::GlobalOp> {

  Type getElementType(Operation *op) const {
    return cast<memref::GlobalOp>(op).getType().getElementType();
  }

  SmallVector<int64_t> getStaticShape(Operation *op) const {
    return llvm::to_vector(cast<memref::GlobalOp>(op).getType().getShape());
  }

  int64_t getStaticSizeInBits(Operation *op) const {
    return computeSizeInBits(getStaticShape(op), getElementType(op));
  }

  std::optional<StorageKind> getPragmaStorageKind(Operation *op) const {
    // Stub; add later (e.g. hls.bind_storage = ...).
    return std::nullopt;
  }

  std::optional<PartitionSpec> getPartitionSpec(Operation *op) const {
    return parsePartitionSpec(op);
  }

  std::optional<int64_t> getEnclosingPipelineII(Operation *op) const {
    // A global has no enclosing pipeline; II is a property of the accesses,
    // which the analysis derives per access site.
    return std::nullopt;
  }

  bool contributesToBRAMCost(Operation *op) const {
    // TODO: a global with an initial value synthesizes as ROM rather than
    //       RAM. It still consumes BRAM, but is inherently read-only, so
    //       port-demand classification could be short-circuited to a
    //       single-port kind. Branch on
    //       cast<memref::GlobalOp>(op).getInitialValue() once the storage
    //       kinds distinguish ROM.
    return contributesToBRAMCostImpl(getPragmaStorageKind(op),
                                     getStaticSizeInBits(op));
  }
};


void circt::hls_analysis::registerBRAMInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
        memref::AllocaOp::attachInterface<MemRefAllocaExternalModel>(*ctx);
        memref::GlobalOp::attachInterface<MemRefGlobalExternalModel>(*ctx);
      });
}