#include "circt/Dialect/HIR/IR/HIR.h"
namespace mlir {
namespace hir {

class BusFanoutInfo {
public:
  BusFanoutInfo(Operation *);

private:
  void dispatchOp(Operation *);
  void visitOp(hir::FuncOp);
  void visitOp(hir::AllocaOp);
  void visitOp(hir::TensorExtractOp);
  void visitOp(hir::BusUnpackOp);
  void visitOp(hir::CallOp);

public:
  llvm::DenseMap<Value, SmallVector<SmallVector<Operation *, 1>, 1>>
      mapBusTensor2Uses;
  llvm::DenseMap<Value, SmallVector<Operation *, 4>> mapBus2Uses;
};

} // namespace hir
} // namespace mlir
