#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
using namespace circt;

Type convertType(Type type);
Type convertBusType(hir::BusType busTy);
Type convertTensorType(mlir::TensorType tensorTy);
bool isRecvBus(DictionaryAttr busAttr);
SmallVector<Value> getHWModuleInputs(hir::FuncType funcTy,
                                     SmallVectorImpl<Value> &args,
                                     SmallVectorImpl<Value> &inputArgs);
std::pair<SmallVector<Type>, SmallVector<Type>>
getHWModuleTypes(hir::FuncType funcTy, SmallVectorImpl<Type> &inputTypes,
                 SmallVectorImpl<Type> &resultTypes);

SmallVector<hw::ModulePortInfo> getHWModulePortInfoList(OpBuilder &builder,
                                                        hir::FuncType funcTy,
                                                        ArrayAttr inputNames,
                                                        ArrayAttr resultNames);
Operation *getConstantX(OpBuilder *builder, Type originalTy);
