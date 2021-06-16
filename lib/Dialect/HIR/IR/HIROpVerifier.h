#ifndef HIROpVerifier
#define HIROpVerifier
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"

namespace mlir {
namespace hir {
LogicalResult verifyFuncOp(hir::FuncOp op);
} // namespace hir
} // namespace mlir

#endif
