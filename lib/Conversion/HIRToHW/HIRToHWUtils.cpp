#include "HIRToHWUtils.h"
#include "circt/Dialect/HIR/IR/helper.h"

bool isInputBus(DictionaryAttr busAttr) {
  return helper::extractBusPortFromDict(busAttr) == "recv";
}

Type convertBusType(hir::BusType busTy) {
  assert(busTy.getElementTypes().size() == 1);
  return busTy.getElementTypes()[0];
}

hw::ArrayType convertTensorType(mlir::TensorType tensorTy) {
  return hw::ArrayType::get(convertType(tensorTy.getElementType()),
                            tensorTy.getNumElements());
}

Type convertType(Type type) {
  if (auto ty = type.dyn_cast<hir::BusType>())
    return convertBusType(ty);
  if (auto ty = type.dyn_cast<mlir::TensorType>())
    return convertTensorType(ty);
  return type;
}

SmallVector<Value> getHWModuleInputs(hir::FuncType funcTy,
                                     SmallVectorImpl<Value> &args) {
  SmallVector<Value> inputArgs;
  for (uint64_t i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto ty = convertType(funcTy.getInputTypes()[i]);
    auto attr = funcTy.getInputAttrs()[i];
    if (ty.isa<hir::BusType>()) {
      if (isInputBus(attr)) {
        inputArgs.push_back(args[i]);
      }
    } else {
      inputArgs.push_back(args[i]);
    }
  }
  return inputArgs;
}

std::pair<SmallVector<Type>, SmallVector<Type>>
getHWModuleTypes(hir::FuncType funcTy) {
  SmallVector<Type> inputTypes;
  SmallVector<Type> resultTypes;
  for (uint64_t i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto ty = convertType(funcTy.getInputTypes()[i]);
    auto attr = funcTy.getInputAttrs()[i];
    if (ty.isa<hir::BusType>()) {
      if (isInputBus(attr)) {
        inputTypes.push_back(ty);
      } else {
        resultTypes.push_back(ty);
      }
    } else {
      inputTypes.push_back(ty);
    }
  }
  return std::make_pair(inputTypes, resultTypes);
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
    if (helper::isBusType(originalTy) && !isInputBus(attr)) {
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
