#include "HIRToHWUtils.h"
#include "circt/Dialect/HIR/IR/helper.h"

bool isRecvBus(DictionaryAttr busAttr) {
  return helper::extractBusPortFromDict(busAttr) == "recv";
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
  return type;
}

std::pair<SmallVector<Value>, SmallVector<Value>>
filterCallOpArgs(hir::FuncType funcTy, OperandRange args) {
  SmallVector<Value> inputs;
  SmallVector<Value> results;
  for (uint64_t i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto ty = funcTy.getInputTypes()[i];
    if (helper::isBusType(ty) && !isRecvBus(funcTy.getInputAttrs()[i])) {
      results.push_back(args[i]);
      continue;
    }
    inputs.push_back(args[i]);
  }

  return std::make_pair(inputs, results);
}

SmallVector<hw::ModulePortInfo> getHWModulePortInfoList(OpBuilder &builder,
                                                        hir::FuncType funcTy,
                                                        ArrayAttr inputNames,
                                                        ArrayAttr resultNames) {
  SmallVector<hw::ModulePortInfo> portInfoList;

  SmallVector<Type> hwInputTypes;
  SmallVector<Type> hwResultTypes;
  SmallVector<StringAttr> hwInputNames;
  SmallVector<StringAttr> hwResultNames;

  // filter the input and output types and names.
  uint64_t i;
  for (i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto originalTy = funcTy.getInputTypes()[i];
    auto hwTy = convertType(originalTy);
    assert(hwTy);
    auto attr = funcTy.getInputAttrs()[i];
    if (helper::isBusType(originalTy) && !isRecvBus(attr)) {
      hwResultTypes.push_back(hwTy);
      auto name = inputNames[i].dyn_cast<StringAttr>();
      assert(name);
      hwResultNames.push_back(name);
      continue;
    }
    hwInputTypes.push_back(hwTy);
    auto name = inputNames[i].dyn_cast<StringAttr>();
    assert(name);
    hwInputNames.push_back(name);
  }

  // Add time type.
  hwInputTypes.push_back(hir::TimeType::get(builder.getContext()));
  auto name = inputNames[i].dyn_cast<StringAttr>();
  assert(name);
  hwInputNames.push_back(name);

  for (uint64_t i = 0; i < funcTy.getResultTypes().size(); i++) {
    auto ty = convertType(funcTy.getResultTypes()[i]);
    assert(ty);
    hwResultTypes.push_back(ty);
    auto name = resultNames[i].dyn_cast<StringAttr>();
    assert(name);
    hwResultNames.push_back(name);
  }

  // insert input args;
  for (i = 0; i < hwInputTypes.size(); i++) {
    hw::ModulePortInfo portInfo;
    portInfo.argNum = i;
    portInfo.direction = hw::PortDirection::INPUT;
    portInfo.name = hwInputNames[i];
    portInfo.type = hwInputTypes[i];
    portInfoList.push_back(portInfo);
  }
  for (uint64_t j = 0; j < hwResultTypes.size(); j++) {
    hw::ModulePortInfo portInfo;
    portInfo.argNum = i + j;
    portInfo.direction = hw::PortDirection::OUTPUT;
    portInfo.name = hwResultNames[j];
    portInfo.type = hwResultTypes[j];
    portInfoList.push_back(portInfo);
  }
  return portInfoList;
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
