#ifndef HIROpVerifier
#define HIROpVerifier
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"

namespace circt {
namespace hir {
LogicalResult verifyFuncOp(hir::FuncOp);
LogicalResult verifyFuncExternOp(hir::FuncExternOp);
LogicalResult verifyAllocaOp(hir::AllocaOp);
LogicalResult verifyCallOp(hir::CallOp);
LogicalResult verifyDelayOp(hir::DelayOp);
LogicalResult verifyYieldOp(hir::YieldOp);
LogicalResult verifyLatchOp(hir::LatchOp);
LogicalResult verifyTimeOp(hir::TimeOp);
LogicalResult verifyForOp(hir::ForOp);
LogicalResult verifyLoadOp(hir::LoadOp);
LogicalResult verifyStoreOp(hir::StoreOp);
LogicalResult verifyIfOp(hir::IfOp);
LogicalResult verifyNextIterOp(hir::NextIterOp);
} // namespace hir
} // namespace circt

#endif
