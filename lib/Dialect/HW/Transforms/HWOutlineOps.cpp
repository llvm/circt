//===- HWOutlineOps.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass which moves certain ops out of hw.module bodies into separate modules.
// It also uniquifies them by types and attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <map>

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWOUTLINEOPS
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
/// Information about an operation which has been marked for outlining. Used for
/// both uniquing and creating the outlined module.
struct OpInfo {
  OpInfo() = default;
  OpInfo(const OpInfo &other) = default;

  StringAttr name;
  ArrayAttr operandTypes;
  ArrayAttr resultTypes;
  DictionaryAttr attributes;
};

/// Hash the OpInfo.
struct OpInfoHash {
  size_t operator()(const OpInfo &opInfo) const {
    return llvm::hash_combine(opInfo.name, opInfo.operandTypes,
                              opInfo.resultTypes, opInfo.attributes);
  }
};

/// Compare two OpInfos.
struct OpInfoEqual {
  size_t operator()(const OpInfo &a, const OpInfo &b) const {
    return a.name == b.name && a.operandTypes == b.operandTypes &&
           a.resultTypes == b.resultTypes && a.attributes == b.attributes;
  }
};
} // namespace

namespace {
/// Create a module containing only the given operation and ports corresponding
/// to its operands and results.
hw::HWModuleOp buildOutlinedModule(OpInfo opInfo, Operation *op,
                                   Operation *top) {
  MLIRContext *context = top->getContext();
  OpBuilder builder(context);
  SmallVector<PortInfo> ports;

  // Print the type identifier with a leading underscore and potentially
  // removing the leading '!'.
  auto emitTypeName = [](Type type, llvm::raw_ostream &os) {
    os << "_";
    std::string typeName;
    llvm::raw_string_ostream(typeName) << type;
    if (typeName[0] == '!')
      os << StringRef(typeName).drop_front(1);
    else
      os << typeName;
  };

  // Create the module name.
  std::string name;
  llvm::raw_string_ostream nameOS(name);
  nameOS << "outlined_" << opInfo.name.getValue();

  // Create the module input ports and append info to the module name.
  nameOS << "_opers";
  for (auto type : opInfo.operandTypes.getAsRange<TypeAttr>()) {
    emitTypeName(type.getValue(), nameOS);
    // Unfortunately, we don't have access to the operand names here, so we just
    // let downstream generate a port name.
    ports.push_back(PortInfo{{builder.getStringAttr(""), type.getValue(),
                              ModulePort::Direction::Input}});
  }

  // Create the module output ports and append info to the module name.
  DenseMap<Value, StringRef> resultNames;
  // Get the result names from the op if it implements the OpAsmOpInterface.
  if (auto asmNames = dyn_cast<mlir::OpAsmOpInterface>(op))
    asmNames.getAsmResultNames(
        [&](Value value, StringRef name) { resultNames[value] = name; });
  nameOS << "_res";
  llvm::SmallVector<Type, 16> resultTypes;
  for (auto [idx, type] :
       llvm::enumerate(opInfo.resultTypes.getAsRange<TypeAttr>())) {
    emitTypeName(type.getValue(), nameOS);
    resultTypes.push_back(type.getValue());
    auto name = resultNames.lookup(op->getResult(idx));
    if (name.empty())
      name = "";
    ports.push_back(PortInfo{{builder.getStringAttr(name), type.getValue(),
                              ModulePort::Direction::Output}});
  }

  // Append only the inherent attributes to the module name.
  nameOS << "_attrs";
  if (auto regOp = op->getRegisteredInfo()) {
    DenseSet<StringAttr> opAttrNames;
    for (auto attrName : regOp->getAttributeNames())
      opAttrNames.insert(attrName);

    for (auto attr : opInfo.attributes) {
      if (!opAttrNames.contains(attr.getName()))
        continue;
      nameOS << "_" << attr.getName().strref() << "_";

      // Print some attribute values specially.
      if (auto i = dyn_cast<IntegerAttr>(attr.getValue()))
        nameOS << i.getValue();
      else
        nameOS << attr.getValue();
    }
  }

  // Create a module for the outlined operation.
  builder.setInsertionPointToEnd(&top->getRegion(0).front());
  auto module = builder.create<hw::HWModuleOp>(
      top->getLoc(), builder.getStringAttr(name), ports);
  auto *body = module.getBodyBlock();

  // Create the outlined operation in the module.
  builder.setInsertionPointToStart(body);
  OperationState state(top->getLoc(), opInfo.name.getValue(),
                       body->getArguments(), resultTypes,
                       opInfo.attributes.getValue());
  Operation *outlinedOp = builder.create(state);

  // Set the hw.outputs to the results of the outlined operation.
  body->getTerminator()->setOperands(outlinedOp->getResults());

  return module;
}
} // namespace

namespace {
class HWOutlineOpsPass : public impl::HWOutlineOpsBase<HWOutlineOpsPass> {
public:
  using HWOutlineOpsBase::HWOutlineOpsBase;
  void runOnOperation() override;
};
} // namespace

void HWOutlineOpsPass::runOnOperation() {
  auto module = getOperation();
  MLIRContext *context = &getContext();

  StringSet<> opNamesToOutline;
  for (auto opName : opNames)
    opNamesToOutline.insert(opName);

  OpBuilder builder(context);
  std::unordered_map<OpInfo, hw::HWModuleOp, OpInfoHash, OpInfoEqual>
      outlinedModules;
  DenseSet<hw::HWModuleOp> outlinedModulesSet;

  module.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Skip modules which we have outlined.
    if (outlinedModulesSet.contains(dyn_cast<hw::HWModuleOp>(op)))
      return WalkResult::skip();
    // Skip ops which we don't want to outline.
    if (!opNamesToOutline.contains(op->getName().getStringRef()) ||
        op->getNumRegions() > 0)
      return WalkResult::advance();

    // Put together the OpInfo for this op.
    SmallVector<Attribute> operandTypes;
    for (auto operandType : op->getOperandTypes())
      operandTypes.push_back(TypeAttr::get(operandType));
    SmallVector<Attribute> resultTypes;
    for (auto resultType : op->getResultTypes())
      resultTypes.push_back(TypeAttr::get(resultType));
    OpInfo opInfo = {
        StringAttr::get(context, op->getName().getStringRef()),
        ArrayAttr::get(context, operandTypes),
        ArrayAttr::get(context, resultTypes),
        cast_or_null<DictionaryAttr>(op->getPropertiesAsAttribute()),
    };

    // Look up the module in the cache and build it if it doesn't exist.
    hw::HWModuleOp outlinedModule;
    auto moduleIter = outlinedModules.find(opInfo);
    if (moduleIter == outlinedModules.end()) {
      outlinedModule = buildOutlinedModule(opInfo, op, module);
      outlinedModules.emplace(opInfo, outlinedModule);
      outlinedModulesSet.insert(outlinedModule);
    } else {
      outlinedModule = moduleIter->second;
    }

    // Replace the op with an instance of the outlined module.
    builder.setInsertionPoint(op);
    auto inst = builder.create<hw::InstanceOp>(
        op->getLoc(), outlinedModule,
        builder.getStringAttr("outlined_" + op->getName().getStringRef()),
        SmallVector<Value>(op->getOperands()));
    inst->setDialectAttrs(op->getDialectAttrs());
    op->replaceAllUsesWith(inst.getResults());
    op->erase();

    // Since the op was erased, we need to skip the walk.
    return WalkResult::skip();
  });
}
