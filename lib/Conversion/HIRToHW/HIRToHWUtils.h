#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"

using namespace circt;

class FuncToHWModulePortMap {
public:
  void addFuncInput(StringAttr name, hw::PortDirection direction, Type type);
  void addFuncResult(StringAttr name, Type type);
  void addClk(OpBuilder &);
  ArrayRef<hw::ModulePortInfo> getPortInfoList();
  const hw::ModulePortInfo getPortInfoForFuncInput(size_t inputArgNum);

private:
  size_t hwModuleInputArgNum = 0;
  size_t hwModuleResultArgNum = 0;
  SmallVector<hw::ModulePortInfo> portInfoList;
  SmallVector<hw::ModulePortInfo *> mapFuncInputToHWModulePortInfo;
};

Type convertType(Type type);

bool isRecvBus(DictionaryAttr busAttr);

std::pair<SmallVector<Value>, SmallVector<Value>>
filterCallOpArgs(hir::FuncType funcTy, OperandRange args);

FuncToHWModulePortMap getHWModulePortMap(OpBuilder &builder,
                                         hir::FuncType funcTy,
                                         ArrayAttr inputNames,
                                         ArrayAttr resultNames);

Operation *getConstantX(OpBuilder *builder, Type originalTy);
