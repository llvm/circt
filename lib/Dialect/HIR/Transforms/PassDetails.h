#ifndef DIALECT_HIR_TRANSFORMS_PASSDETAILS_H
#define DIALECT_HIR_TRANSFORMS_PASSDETAILS_H
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/Transforms/Passes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace hw {
class HWDialect;
} // namespace hw

namespace comb {
class CombDialect;
} // namespace comb

namespace sv {
class SVDialect;
} // namespace sv

namespace hir {
#define GEN_PASS_CLASSES
#include "circt/Dialect/HIR/Transforms/Passes.h.inc"

} // namespace hir
} // namespace circt

#endif // DIALECT_HIR_TRANSFORMS_PASSDETAILS_H
