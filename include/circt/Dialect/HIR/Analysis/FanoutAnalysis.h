#include "circt/Dialect/HIR/IR/HIR.h"
namespace circt {
namespace hir {

typedef SmallVector<Operation *> ListOfUses;

class MemrefUseInfo {
public:
  MemrefUseInfo(FuncOp);

private:
  void addOpToUseList(Operation *, Value, uint64_t, uint64_t);
  void dispatchOp(Operation *);
  void visitOp(hir::LoadOp);
  void visitOp(hir::StoreOp);
  void visitOp(hir::CallOp);
  void visitOp(hir::FuncOp);
  void visitOp(hir::AllocaOp);

public:
  llvm::DenseMap<Value, SmallVector<SmallVector<ListOfUses>>>
      mapMemref2PerPortPerBankUses;
};

} // namespace hir
} // namespace circt
