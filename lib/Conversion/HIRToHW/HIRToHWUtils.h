#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
using namespace circt;

Type convertType(Type type);

bool isRecvBus(DictionaryAttr busAttr);

std::pair<SmallVector<Value>, SmallVector<Value>>
filterCallOpArgs(hir::FuncType funcTy, OperandRange args);

SmallVector<hw::ModulePortInfo> getHWModulePortInfoList(OpBuilder &builder,
                                                        hir::FuncType funcTy,
                                                        ArrayAttr inputNames,
                                                        ArrayAttr resultNames);

Operation *getConstantX(OpBuilder *builder, Type originalTy);
