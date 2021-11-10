#ifndef HIROpVerifier
#define HIROpVerifier
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"

namespace circt {
namespace hir {
LogicalResult verifyAllocaOp(hir::AllocaOp);
LogicalResult verifyBusTensorMapOp(hir::BusTensorMapOp);
LogicalResult verifyBusTensorInsertElementOp(hir::BusTensorInsertElementOp);
LogicalResult verifyCallOp(hir::CallOp);
LogicalResult verifyCastOp(hir::CastOp);
LogicalResult verifyDelayOp(hir::DelayOp);
LogicalResult verifyForOp(hir::ForOp);
LogicalResult verifyFuncExternOp(hir::FuncExternOp);
LogicalResult verifyFuncOp(hir::FuncOp);
LogicalResult verifyIfOp(hir::IfOp);
LogicalResult verifyLatchOp(hir::LatchOp);
LogicalResult verifyLoadOp(hir::LoadOp);
LogicalResult verifyNextIterOp(hir::NextIterOp);
LogicalResult verifyProbeOp(hir::ProbeOp);
LogicalResult verifyStoreOp(hir::StoreOp);
LogicalResult verifyTimeOp(hir::TimeOp);
LogicalResult verifyYieldOp(hir::YieldOp);
} // namespace hir
} // namespace circt

#endif
