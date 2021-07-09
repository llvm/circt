#ifndef HIROpVerifier
#define HIROpVerifier
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"

namespace circt {
namespace hir {
LogicalResult verifyFuncOp(hir::FuncOp op);
LogicalResult verifyDelayOp(hir::DelayOp op);
LogicalResult verifyTimeOp(hir::TimeOp op);
LogicalResult verifyForOp(hir::ForOp op);
} // namespace hir
} // namespace circt

#endif
