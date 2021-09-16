#include "HIRToHWUtils.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"

void FuncToHWModulePortMap::addFuncInput(StringAttr name,
                                         hw::PortDirection direction,
                                         Type type) {
  assert(name);
  assert(type);
  assert(direction != hw::PortDirection::INOUT);

  size_t argNum = (direction == hw::PortDirection::INPUT)
                      ? (hwModuleInputArgNum++)
                      : (hwModuleResultArgNum++);

  portInfoList.push_back(
      {.name = name, .direction = direction, .type = type, .argNum = argNum});

  mapFuncInputToHWModulePortInfo.push_back(&portInfoList.back());
}
void FuncToHWModulePortMap::addClk(OpBuilder &builder) {
  auto clkName = builder.getStringAttr("clk");
  portInfoList.push_back({.name = clkName,
                          .direction = hw::PortDirection::INPUT,
                          .type = builder.getI1Type(),
                          .argNum = hwModuleInputArgNum++});
}
void FuncToHWModulePortMap::addFuncResult(StringAttr name, Type type) {
  assert(name);
  assert(type);
  portInfoList.push_back({.name = name,
                          .direction = hw::PortDirection::OUTPUT,
                          .type = type,
                          .argNum = hwModuleResultArgNum++});
}

bool isSendBus(DictionaryAttr busAttr) {
  return helper::extractBusPortFromDict(busAttr) == "send";
}

ArrayRef<hw::ModulePortInfo> FuncToHWModulePortMap::getPortInfoList() {
  return portInfoList;
}

const hw::ModulePortInfo
FuncToHWModulePortMap::getPortInfoForFuncInput(size_t inputArgNum) {
  return *mapFuncInputToHWModulePortInfo[inputArgNum];
}

IntegerType convertToIntegerType(Type ty) {
  if (auto tupleTy = ty.dyn_cast<TupleType>()) {
    auto width = 0;
    for (auto elementTy : tupleTy.getTypes()) {
      width += convertToIntegerType(elementTy).getWidth();
    }
    return IntegerType::get(ty.getContext(), width);
  }
  auto integerTy = ty.dyn_cast<IntegerType>();
  assert(integerTy);
  return integerTy;
}

Type convertBusType(hir::BusType busTy) {
  assert(busTy.getElementTypes().size() == 1);
  return convertToIntegerType(busTy.getElementTypes()[0]);
}

Type convertToArrayType(mlir::Type elementTy, ArrayRef<int64_t> shape) {
  assert(shape.size() > 0);
  if (shape.size() > 1)
    return hw::ArrayType::get(convertToArrayType(elementTy, shape.slice(1)),
                              shape[0]);
  assert(shape[0] > 0);
  return hw::ArrayType::get(convertType(elementTy), shape[0]);
}

Type convertTensorType(mlir::TensorType tensorTy) {
  if (tensorTy.getShape().size() > 0)
    return convertToArrayType(tensorTy.getElementType(), tensorTy.getShape());
  return convertType(tensorTy.getElementType());
}

Type convertType(Type type) {
  if (auto ty = type.dyn_cast<hir::BusType>())
    return convertBusType(ty);
  if (auto ty = type.dyn_cast<mlir::TensorType>())
    return convertTensorType(ty);
  if (type.isa<hir::TimeType>())
    return IntegerType::get(type.getContext(), 1);
  return type;
}

std::pair<SmallVector<Value>, SmallVector<Value>>
filterCallOpArgs(hir::FuncType funcTy, OperandRange args) {
  SmallVector<Value> inputs;
  SmallVector<Value> results;
  for (uint64_t i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto ty = funcTy.getInputTypes()[i];
    if (helper::isBusType(ty) && isSendBus(funcTy.getInputAttrs()[i])) {
      results.push_back(args[i]);
      continue;
    }
    inputs.push_back(args[i]);
  }

  return std::make_pair(inputs, results);
}

FuncToHWModulePortMap getHWModulePortMap(OpBuilder &builder,
                                         hir::FuncType funcTy,
                                         ArrayAttr inputNames,
                                         ArrayAttr resultNames) {
  FuncToHWModulePortMap portMap;

  // filter the input and output types and names.
  uint64_t i;
  for (i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto originalTy = funcTy.getInputTypes()[i];
    auto hwTy = convertType(originalTy);
    assert(hwTy);
    auto attr = funcTy.getInputAttrs()[i];
    auto name = inputNames[i].dyn_cast<StringAttr>();
    if (helper::isBusType(originalTy) && isSendBus(attr)) {
      portMap.addFuncInput(name, hw::PortDirection::OUTPUT, hwTy);
    } else {
      portMap.addFuncInput(name, hw::PortDirection::INPUT, hwTy);
    }
  }

  // Add time input arg.
  auto timeVarName = inputNames[i].dyn_cast<StringAttr>();
  portMap.addFuncInput(timeVarName, hw::PortDirection::INPUT,
                       builder.getI1Type());

  // Add clk input arg.
  portMap.addClk(builder);

  for (uint64_t i = 0; i < funcTy.getResultTypes().size(); i++) {
    auto hwTy = convertType(funcTy.getResultTypes()[i]);
    auto name = resultNames[i].dyn_cast<StringAttr>();
    portMap.addFuncResult(name, hwTy);
  }

  return portMap;
}

Operation *getConstantX(OpBuilder *builder, Type originalTy) {
  auto hwTy = convertType(originalTy);
  if (auto ty = hwTy.dyn_cast<hw::ArrayType>()) {
    auto *element = getConstantX(builder, ty.getElementType());
    uint64_t size = ty.getSize();
    if (size == 1)
      return element;
    SmallVector<Value> elementCopies;
    for (uint64_t i = 0; i < size; i++) {
      elementCopies.push_back(element->getResult(0));
    }
    return builder->create<hw::ArrayCreateOp>(builder->getUnknownLoc(),
                                              elementCopies);
  }
  assert(hwTy.isa<IntegerType>());
  return builder->create<sv::ConstantXOp>(builder->getUnknownLoc(), hwTy);
}
