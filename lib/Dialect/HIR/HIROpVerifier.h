#ifndef HIROpVerifier
#define HIROpVerifier
#include "circt/Dialect/HIR/HIR.h"
#include "circt/Dialect/HIR/HIRDialect.h"

namespace mlir {
namespace hir {
LogicalResult verifySelectOp(hir::SelectOp op);
} // namespace hir
} // namespace mlir

#endif
