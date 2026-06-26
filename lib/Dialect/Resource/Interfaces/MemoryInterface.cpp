#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "circt/Dialect/Resource/Interfaces/MemoryInterface.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"

using namespace mlir;
// OpInterface dispatch tables. Include exactly once in the project.

#include "circt/Dialect/Resource/Interfaces/MemoryOpInterface.cpp.inc"

//===----------------------------------------------------------------------===//
// External model: attach MemoryResourceOpInterface to memref allocs.
//===----------------------------------------------------------------------===//


unsigned getStageStart(Operation *user) {
  if (auto stage =
    user->getParentOfType<circt::loopschedule::LoopSchedulePipelineStageOp>())
    return stage.getStart();
  return 0;  // not inside a pipeline stage → treat as cycle 0
}


// Returns {scope, ii}; ii == 0 means a sequential (non-pipelined) scope.
static std::pair<Operation *, unsigned> getScopeAndII(Operation *user) {
  using namespace circt::loopschedule;
  if (auto pipe = user->getParentOfType<LoopSchedulePipelineOp>())
    return {pipe.getOperation(), std::max<unsigned>(pipe.getII(), 1u)};
  if (auto loop = user->getParentOfType<LoopLikeOpInterface>())
    return {loop.getOperation(), 0u};        // sequential loop
  return {user->getParentOfType<func::FuncOp>(), 0u}; // top level
}
struct MemRefLoopScheduleExternalModel :
circt::hls_analysis::MemoryResourceOpInterface::ExternalModel<MemRefLoopScheduleExternalModel,
memref::AllocaOp> {
  Type getElementType(Operation *op) const {
    return cast<memref::AllocaOp>(op).getType().getElementType();
  }

  SmallVector<int64_t> getStaticShape(Operation *op) const {
    auto mt = cast<memref::AllocaOp>(op)
    .getMemref().getType();
    return llvm::to_vector(mt.getShape());
  }

  int64_t getStaticSizeInBits(Operation *op) const {
    auto shape = getStaticShape(op);
    int64_t n = 1;
    for (int64_t d : shape) {
      if (ShapedType::isDynamicShape(d)) return 0;
      n *= d;
    }
    Type elt = getElementType(op);
    unsigned w = 0;
    if (elt.isIntOrFloat())
      w = elt.getIntOrFloatBitWidth();
    else if (auto idx = dyn_cast<IndexType>(elt))
      w = 64;  // platform-dependent; pick a convention and document it
    return n * static_cast<int64_t>(w);
  }

  std::optional<circt::hls_analysis::StorageKind> getPragmaStorageKind(Operation *op) const {
    // Stub add later (ex hls.bind_op = ...)
    return std::nullopt;
  }

  std::optional<circt::hls_analysis::PartitionSpec> getPartitionSpec(Operation *op) const {
    auto arr = op->getAttrOfType<ArrayAttr>("hls.array_partition");
    if (!arr || arr.empty())
      return std::nullopt;   // no pragma → unpartitioned, caller treats as Pf=1

    // Take the first entry. See note below on multiple entries.
    auto dict = dyn_cast<DictionaryAttr>(arr[0]);
    if (!dict)
      return std::nullopt;

    circt::hls_analysis::PartitionSpec spec;

    // kind (required to disambiguate the enum)
    auto kindAttr = dict.getAs<StringAttr>("kind");
    if (!kindAttr)
      return std::nullopt;
    StringRef k = kindAttr.getValue();
    if (k == "cyclic")        spec.kind = circt::hls_analysis::PartitionSpec::Cyclic;
    else if (k == "block")    spec.kind = circt::hls_analysis::PartitionSpec::Block;
    else if (k == "complete") spec.kind = circt::hls_analysis::PartitionSpec::Complete;
    else
      return std::nullopt;    // unknown kind → don't guess

    // factor (ignored for Complete, but parse defensively)
    if (auto f = dict.getAs<IntegerAttr>("factor"))
      spec.factor = static_cast<unsigned>(f.getInt());
    else
      spec.factor = 1;

    // dim — NOTE: Vitis pragma dims are 1-indexed; your struct is 0-indexed.
    if (auto d = dict.getAs<IntegerAttr>("dim")) {
      int64_t pragmaDim = d.getInt();
      spec.dim = pragmaDim > 0 ? static_cast<unsigned>(pragmaDim - 1) : 0;
    } else {
      spec.dim = 0;
    }

    return spec;
  }

  std::optional<int64_t> getEnclosingPipelineII(Operation *op) const {
    // Stub add later
    return std::nullopt;
  }
  
  bool contributesToBRAMCost(Operation *op) const {
    // Opt out of cost accounting if the user explicitly annotated this
    // allocation as not BRAM-backed (e.g. registers, FIFO, stream).
    if (auto kind = getPragmaStorageKind(op)) {
      if (*kind == circt::hls_analysis::StorageKind::LUTRAM) return false;
      // URAM goes through a different resource pool; up to your tool
      // whether to count it here.
    }
    // TODO: most tools have a threshold where the memory is promoted to
    //       block ram. This number is placeholder make parmaterizable
    //       either in a td file or from some specification somewhere
    // Assume 32 width * 4 byte data for threshold to promote to bram for now 
    unsigned bramPromotion = 32 * 32;
    return getStaticSizeInBits(op) > bramPromotion;
  }
};

void circt::hls_analysis::registerBRAMInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                         memref::MemRefDialect *dialect) {
    memref::AllocaOp::attachInterface<MemRefLoopScheduleExternalModel>(*ctx);
  });
}