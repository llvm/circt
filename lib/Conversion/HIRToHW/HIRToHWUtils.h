#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HIR/IR/HIR.h"
#include "circt/Dialect/HIR/IR/HIRDialect.h"
#include "circt/Dialect/HIR/IR/helper.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"

using namespace circt;

class FuncToHWModulePortMap {
public:
  void addFuncInput(StringAttr name, hw::PortDirection direction, Type type);
  void addFuncResult(StringAttr name, Type type);
  void addClk(OpBuilder &);
  void addReset(OpBuilder &);
  ArrayRef<hw::PortInfo> getPortInfoList();
  const hw::PortInfo getPortInfoForFuncInput(size_t inputArgNum);

private:
  size_t hwModuleInputArgNum = 0;
  size_t hwModuleResultArgNum = 0;
  SmallVector<hw::PortInfo> portInfoList;
  SmallVector<hw::PortInfo> mapFuncInputToHWPortInfo;
};

bool isRecvBus(DictionaryAttr busAttr);

std::pair<SmallVector<Value>, SmallVector<Value>>
filterCallOpArgs(hir::FuncType funcTy, OperandRange args);

FuncToHWModulePortMap getHWModulePortMap(OpBuilder &builder,
                                         hir::FuncType funcTy,
                                         ArrayAttr inputNames,
                                         ArrayAttr resultNames);

Operation *getConstantX(OpBuilder *, Type);
Operation *
getConstantXArray(OpBuilder *builder, Type hwTy,
                  DenseMap<Value, SmallVector<Value>> &mapArrayToElements);

ArrayAttr getHWParams(Attribute, bool ignoreValues = false);

Value getDelayedValue(OpBuilder *builder, Value input, int64_t delay,
                      Optional<StringRef> name, Location loc, Value clk,
                      Value reset);

Value convertToNamedValue(OpBuilder &builder, StringRef name, Value val);
Value convertToOptionalNamedValue(OpBuilder &builder, Optional<StringRef> name,
                                  Value val);
SmallVector<Value> insertBusMapLogic(OpBuilder &builder, Block &bodyBlock,
                                     ArrayRef<Value> operands);

Value insertConstArrayGetLogic(OpBuilder &builder, Value arr, int idx);

Value getClkFromHWModule(hw::HWModuleOp op);
Value getResetFromHWModule(hw::HWModuleOp op);

class HIRToHWMapping {
private:
  DenseMap<Value, Value> mapHIRToHWValue;

public:
  Value lookup(Value hirValue) {
    assert(mapHIRToHWValue.find(hirValue) != mapHIRToHWValue.end());
    return mapHIRToHWValue.lookup(hirValue);
  }
  void map(Value hirValue, Value hwValue) {
    assert(hirValue.getParentRegion()->getParentOfType<hir::FuncOp>());
    assert(hwValue.getParentRegion()->getParentOfType<hw::HWModuleOp>());
    mapHIRToHWValue[hirValue] = hwValue;
  }

  BlockAndValueMapping getBlockAndValueMapping() {
    BlockAndValueMapping blockAndValueMap;
    for (auto keyValue : mapHIRToHWValue) {
      blockAndValueMap.map(keyValue.getFirst(), keyValue.getSecond());
    }
    return blockAndValueMap;
  }

  // This function ensures that all the 'hirValue -> from' get updated
  // to 'hirValue -> to'.
  // There can be multiple such values because of assign ops in hir dialect.
  void replaceAllHWUses(Value from, Value to) {
    assert(from.getParentRegion()->getParentOfType<hw::HWModuleOp>());
    assert(to.getParentRegion()->getParentOfType<hw::HWModuleOp>());
    from.replaceAllUsesWith(to);
    for (auto keyValue : mapHIRToHWValue) {
      if (keyValue.getSecond() == from)
        mapHIRToHWValue[keyValue.getFirst()] = to;
    }
  }
};
