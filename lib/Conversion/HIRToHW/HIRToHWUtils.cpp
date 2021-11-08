#include "HIRToHWUtils.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"

void FuncToHWModulePortMap::addFuncInput(StringAttr name,
                                         hw::PortDirection direction,
                                         Type type) {
  assert(name);
  assert(type);
  assert((direction == hw::PortDirection::INPUT) ||
         (direction == hw::PortDirection::OUTPUT));

  size_t argNum = (direction == hw::PortDirection::INPUT)
                      ? (hwModuleInputArgNum++)
                      : (hwModuleResultArgNum++);

  portInfoList.push_back(
      {.name = name, .direction = direction, .type = type, .argNum = argNum});

  mapFuncInputToHWPortInfo.push_back(
      {.name = name, .direction = direction, .type = type, .argNum = argNum});
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

ArrayRef<hw::PortInfo> FuncToHWModulePortMap::getPortInfoList() {
  return portInfoList;
}

const hw::PortInfo
FuncToHWModulePortMap::getPortInfoForFuncInput(size_t inputArgNum) {
  auto modulePortInfo = mapFuncInputToHWPortInfo[inputArgNum];
  assert((modulePortInfo.direction == hw::PortDirection::INPUT) ||
         (modulePortInfo.direction == hw::PortDirection::OUTPUT));
  return modulePortInfo;
}

std::pair<SmallVector<Value>, SmallVector<Value>>
filterCallOpArgs(hir::FuncType funcTy, OperandRange args) {
  SmallVector<Value> inputs;
  SmallVector<Value> results;
  for (uint64_t i = 0; i < funcTy.getInputTypes().size(); i++) {
    auto ty = funcTy.getInputTypes()[i];
    if (helper::isBusLikeType(ty) && isSendBus(funcTy.getInputAttrs()[i])) {
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
    auto hwTy = *helper::convertToHWType(originalTy);
    assert(hwTy);
    auto attr = funcTy.getInputAttrs()[i];
    auto name = inputNames[i].dyn_cast<StringAttr>();
    if (helper::isBusLikeType(originalTy) && isSendBus(attr)) {
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

  // Add hir.func results.
  for (uint64_t i = 0; i < funcTy.getResultTypes().size(); i++) {
    auto hwTy = *helper::convertToHWType(funcTy.getResultTypes()[i]);
    auto name = resultNames[i].dyn_cast<StringAttr>();
    portMap.addFuncResult(name, hwTy);
  }

  return portMap;
}

Operation *getConstantX(OpBuilder *builder, Type originalTy) {
  auto hwTy = *helper::convertToHWType(originalTy);
  if (auto ty = hwTy.dyn_cast<hw::ArrayType>()) {
    auto *constX = getConstantX(builder, ty.getElementType());
    uint64_t size = ty.getSize();
    SmallVector<Value> elementCopies;
    for (uint64_t i = 0; i < size; i++) {
      elementCopies.push_back(constX->getResult(0));
    }
    return builder->create<hw::ArrayCreateOp>(builder->getUnknownLoc(),
                                              elementCopies);
  }
  assert(hwTy.isa<IntegerType>());
  return builder->create<sv::ConstantXOp>(builder->getUnknownLoc(), hwTy);
}

ArrayAttr getHWParams(Attribute paramsAttr, bool ignoreValues) {
  if (!paramsAttr)
    return ArrayAttr();

  auto params = paramsAttr.dyn_cast<DictionaryAttr>();
  assert(params);

  Builder builder(params.getContext());
  SmallVector<Attribute> hwParams;
  for (const NamedAttribute &param : params) {
    auto name = builder.getStringAttr(param.first.strref());
    auto type = TypeAttr::get(param.second.getType());
    auto value = ignoreValues ? Attribute() : param.second;
    auto hwParam =
        hw::ParamDeclAttr::get(builder.getContext(), name, type, value);
    hwParams.push_back(hwParam);
  }
  return ArrayAttr::get(builder.getContext(), hwParams);
}

Value getDelayedValue(OpBuilder *builder, Value input, int64_t delay,
                      Optional<StringRef> name, Location loc, Value clk) {
  assert(input.getType().isa<mlir::IntegerType>() ||
         input.getType().isa<hw::ArrayType>());

  auto nameAttr = name ? builder->getStringAttr(name.getValue()) : StringAttr();

  Type regTy;
  if (delay > 1)
    regTy = hw::ArrayType::get(input.getType(), delay);
  else
    regTy = input.getType();

  auto reg =
      builder->create<sv::RegOp>(builder->getUnknownLoc(), regTy, nameAttr);

  auto regOutput =
      builder->create<sv::ReadInOutOp>(builder->getUnknownLoc(), reg);
  Value regInput;
  if (delay > 1) {
    auto c1 =
        helper::materializeIntegerConstant(*builder, 1, helper::clog2(delay));
    auto sliceTy =
        hw::ArrayType::get(builder->getContext(), input.getType(), delay - 1);
    auto regSlice = builder->create<hw::ArraySliceOp>(builder->getUnknownLoc(),
                                                      sliceTy, regOutput, c1);
    regInput = builder->create<hw::ArrayConcatOp>(
        builder->getUnknownLoc(),
        ArrayRef<Value>({regSlice, builder->create<hw::ArrayCreateOp>(
                                       builder->getUnknownLoc(), input)}));
  } else {
    regInput = input;
  }
  builder->create<sv::AlwaysOp>(
      builder->getUnknownLoc(), sv::EventControl::AtPosEdge, clk,
      [builder, &reg, &regInput] {
        builder->create<sv::PAssignOp>(builder->getUnknownLoc(), reg.result(),
                                       regInput);
      });
  Value output;
  if (delay > 1) {
    auto c0 =
        helper::materializeIntegerConstant(*builder, 0, helper::clog2(delay));
    output = builder->create<hw::ArrayGetOp>(loc, regOutput, c0).getResult();
  } else {
    output = regOutput;
  }

  assert(input.getType() == output.getType());
  return output;
}
Value convertToNamedValue(OpBuilder &builder, StringRef name, Value val) {

  assert(val.getType().isa<mlir::IntegerType>() ||
         val.getType().isa<hw::ArrayType>());
  Value namedVal;
  auto wire = builder.create<sv::WireOp>(builder.getUnknownLoc(), val.getType(),
                                         builder.getStringAttr(name));
  builder.create<sv::AssignOp>(builder.getUnknownLoc(), wire, val);
  return builder.create<sv::ReadInOutOp>(builder.getUnknownLoc(), wire);
}

Value convertToOptionalNamedValue(OpBuilder &builder, Optional<StringRef> name,
                                  Value val) {

  assert(val.getType().isa<mlir::IntegerType>() ||
         val.getType().isa<hw::ArrayType>());
  if (name) {
    return convertToNamedValue(builder, name.getValue(), val);
  }
  return val;
}
