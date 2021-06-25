#include "circt/Dialect/HIR/IR/HIR.h"
namespace circt {
namespace hir {

class MemrefFanoutInfo {
public:
  MemrefFanoutInfo(Operation *);

private:
  void dispatchOp(Operation *);
  void visitOp(hir::FuncOp);
  void visitOp(hir::AllocaOp);
  void visitOp(hir::CallOp);
  void visitOp(hir::LoadOp);
  void visitOp(hir::StoreOp);

public:
  llvm::DenseMap<Value, SmallVector<SmallVector<SmallVector<Operation *>>>>
      mapMemref2PerPortPerBankUses;
};

} // namespace hir
} // namespace circt
