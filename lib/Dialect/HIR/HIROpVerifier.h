#ifndef HIROpVerifier
#define HIROpVerifier
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"

namespace mlir {
namespace hir {
LogicalResult verifyFuncOp(hir::FuncOp op);
} // namespace hir
} // namespace mlir

#endif
