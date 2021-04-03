#include "MLIRGen.h"
#include "mlir/IR/Builders.h"
int emitMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  mlir::OpBuilder builder(&context);
  auto theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  auto funcType = builder.getFunctionType(llvm::None, llvm::None);
  auto symName =
      mlir::StringAttr::get(&context, llvm::SmallString<16>("test_func"));
  auto inputDelays =
      mlir::ArrayAttr::get(&context, llvm::SmallVector<mlir::Attribute, 4>());
  auto outputDelays =
      mlir::ArrayAttr::get(&context, llvm::SmallVector<mlir::Attribute, 4>());
  auto defOp = builder.create<hir::DefOp>(
      builder.getUnknownLoc(), llvm::SmallVector<Type, 4>(),
      TypeAttr::get(funcType), symName, inputDelays, outputDelays);
  auto *entryBlock = defOp.addEntryBlock();
  entryBlock->addArgument(hir::TimeType::get(&context));
  builder.setInsertionPointToStart(entryBlock);
  theModule.push_back(defOp);
  module = theModule;
  return 0;
}
