#include "circt/Dialect/HIR/IR/HIR.h"
namespace circt {
namespace hir {

typedef SmallVector<Operation *> ListOfUses;

class MemrefUseInfo {
public:
  MemrefUseInfo(FuncOp);

private:
  LogicalResult addOpToUseList(Operation *, Value, uint64_t, uint64_t);
  LogicalResult dispatchOp(Operation *);
  LogicalResult visitOp(hir::LoadOp);
  LogicalResult visitOp(hir::StoreOp);
  LogicalResult visitOp(hir::FuncOp);
  LogicalResult visitOp(hir::AllocaOp);
  LogicalResult visitOp(hir::CallOp);

public:
  llvm::DenseMap<Value, SmallVector<SmallVector<ListOfUses>>>
      mapMemref2PerPortPerBankUses;
};

} // namespace hir
} // namespace circt
