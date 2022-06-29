//===- HWOps.cpp - Implement the HW operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the HW ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/HWVisitors.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace circt;
using namespace hw;

/// Flip a port direction.
PortDirection hw::flip(PortDirection direction) {
  switch (direction) {
  case PortDirection::INPUT:
    return PortDirection::OUTPUT;
  case PortDirection::OUTPUT:
    return PortDirection::INPUT;
  case PortDirection::INOUT:
    return PortDirection::INOUT;
  }
  llvm_unreachable("unknown PortDirection");
}

/// Return true if the specified operation is a combinational logic op.
bool hw::isCombinational(Operation *op) {
  struct IsCombClassifier : public TypeOpVisitor<IsCombClassifier, bool> {
    bool visitInvalidTypeOp(Operation *op) { return false; }
    bool visitUnhandledTypeOp(Operation *op) { return true; }
  };

  return (op->getDialect() && op->getDialect()->getNamespace() == "comb") ||
         IsCombClassifier().dispatchTypeOpVisitor(op);
}

/// Get a special name to use when printing the entry block arguments of the
/// region contained by an operation in this dialect.
static void getAsmBlockArgumentNamesImpl(mlir::Region &region,
                                         OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  // Assign port names to the bbargs.
  auto *module = region.getParentOp();

  auto *block = &region.front();
  for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
    auto name = getModuleArgumentName(module, i);
    if (!name.empty())
      setNameFn(block->getArgument(i), name);
  }
}

enum class Delimiter {
  None,
  Paren,               // () enclosed list
  OptionalLessGreater, // <> enclosed list or absent
};

/// Check parameter specified by `value` to see if it is valid within the scope
/// of the specified module `module`.  If not, emit an error at the location of
/// `usingOp` and return failure, otherwise return success.  If `usingOp` is
/// null, then no diagnostic is generated.
///
/// If `disallowParamRefs` is true, then parameter references are not allowed.
LogicalResult hw::checkParameterInContext(Attribute value, Operation *module,
                                          Operation *usingOp,
                                          bool disallowParamRefs) {
  // Literals are always ok.  Their types are already known to match
  // expectations.
  if (value.isa<IntegerAttr>() || value.isa<FloatAttr>() ||
      value.isa<StringAttr>() || value.isa<ParamVerbatimAttr>())
    return success();

  // Check both subexpressions of an expression.
  if (auto expr = value.dyn_cast<ParamExprAttr>()) {
    for (auto op : expr.getOperands())
      if (failed(
              checkParameterInContext(op, module, usingOp, disallowParamRefs)))
        return failure();
    return success();
  }

  // Parameter references need more analysis to make sure they are valid within
  // this module.
  if (auto parameterRef = value.dyn_cast<ParamDeclRefAttr>()) {
    auto nameAttr = parameterRef.getName();

    // Don't allow references to parameters from the default values of a
    // parameter list.
    if (disallowParamRefs) {
      if (usingOp)
        usingOp->emitOpError("parameter ")
            << nameAttr << " cannot be used as a default value for a parameter";
      return failure();
    }

    // Find the corresponding attribute in the module.
    for (auto param : module->getAttrOfType<ArrayAttr>("parameters")) {
      auto paramAttr = param.cast<ParamDeclAttr>();
      if (paramAttr.getName() != nameAttr)
        continue;

      // If the types match then the reference is ok.
      if (paramAttr.getType().getValue() == parameterRef.getType())
        return success();

      if (usingOp) {
        auto diag = usingOp->emitOpError("parameter ")
                    << nameAttr << " used with type " << parameterRef.getType()
                    << "; should have type " << paramAttr.getType().getValue();
        diag.attachNote(module->getLoc()) << "module declared here";
      }
      return failure();
    }

    if (usingOp) {
      auto diag = usingOp->emitOpError("use of unknown parameter ") << nameAttr;
      diag.attachNote(module->getLoc()) << "module declared here";
    }
    return failure();
  }

  if (usingOp)
    usingOp->emitOpError("invalid parameter value ") << value;
  return failure();
}

/// Return true if the specified attribute tree is made up of nodes that are
/// valid in a parameter expression.
bool hw::isValidParameterExpression(Attribute attr, Operation *module) {
  return succeeded(checkParameterInContext(attr, module, nullptr, false));
}

/// Return the symbol (if any, else null) on the corresponding input port
/// argument.
StringAttr hw::getArgSym(Operation *op, unsigned i) {
  assert(isAnyModuleOrInstance(op) &&
         "Can only get module ports from an instance or module");
  StringAttr sym = {};
  auto argAttrs = op->getAttrOfType<ArrayAttr>(
      mlir::function_interface_impl::getArgDictAttrName());
  if (argAttrs && (i < argAttrs.size()))
    if (auto s = argAttrs[i].cast<DictionaryAttr>())
      if (auto symRef = s.get("hw.exportPort"))
        sym = symRef.cast<FlatSymbolRefAttr>().getAttr();
  return sym;
}

/// Return the symbol (if any, else null) on the corresponding output port
/// argument.
StringAttr hw::getResultSym(Operation *op, unsigned i) {
  assert(isAnyModuleOrInstance(op) &&
         "Can only get module ports from an instance or module");
  StringAttr sym = {};
  auto resAttrs = op->getAttrOfType<ArrayAttr>(
      mlir::function_interface_impl::getResultDictAttrName());
  if (resAttrs && (i < resAttrs.size()))
    if (auto s = resAttrs[i].cast<DictionaryAttr>())
      if (auto symRef = s.get("hw.exportPort"))
        sym = symRef.cast<FlatSymbolRefAttr>().getAttr();
  return sym;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttribute(valueAttr());
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  IntegerAttr valueAttr;

  if (parser.parseAttribute(valueAttr, "value", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(valueAttr.getType());
  return success();
}

LogicalResult ConstantOp::verify() {
  // If the result type has a bitwidth, then the attribute must match its width.
  if (value().getBitWidth() != getType().getWidth())
    return emitError(
        "hw.constant attribute bitwidth doesn't match return type");

  return success();
}

/// Build a ConstantOp from an APInt, infering the result type from the
/// width of the APInt.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       const APInt &value) {

  auto type = IntegerType::get(builder.getContext(), value.getBitWidth());
  auto attr = builder.getIntegerAttr(type, value);
  return build(builder, result, type, attr);
}

/// Build a ConstantOp from an APInt, infering the result type from the
/// width of the APInt.
void ConstantOp::build(OpBuilder &builder, OperationState &result,
                       IntegerAttr value) {
  return build(builder, result, value.getType(), value);
}

/// This builder allows construction of small signed integers like 0, 1, -1
/// matching a specified MLIR IntegerType.  This shouldn't be used for general
/// constant folding because it only works with values that can be expressed in
/// an int64_t.  Use APInt's instead.
void ConstantOp::build(OpBuilder &builder, OperationState &result, Type type,
                       int64_t value) {
  auto numBits = type.cast<IntegerType>().getWidth();
  build(builder, result, APInt(numBits, (uint64_t)value, /*isSigned=*/true));
}

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto intTy = getType();
  auto intCst = getValue();

  // Sugar i1 constants with 'true' and 'false'.
  if (intTy.getWidth() == 1)
    return setNameFn(getResult(), intCst.isZero() ? "false" : "true");

  // Otherwise, build a complex name with the value and type.
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << intCst << '_' << intTy;
  setNameFn(getResult(), specialName.str());
}

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "constant has no operands");
  return valueAttr();
}

//===----------------------------------------------------------------------===//
// ParamValueOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamValue(OpAsmParser &p, Attribute &value,
                                   Type &resultType) {
  if (p.parseType(resultType) || p.parseEqual() ||
      p.parseAttribute(value, resultType))
    return failure();
  return success();
}

static void printParamValue(OpAsmPrinter &p, Operation *, Attribute value,
                            Type resultType) {
  p << resultType << " = ";
  p.printAttributeWithoutType(value);
}

LogicalResult ParamValueOp::verify() {
  // Check that the attribute expression is valid in this module.
  return checkParameterInContext(
      value(), (*this)->getParentOfType<hw::HWModuleOp>(), *this);
}

OpFoldResult ParamValueOp::fold(ArrayRef<Attribute> constants) {
  assert(constants.empty() && "hw.param.value has no operands");
  return valueAttr();
}

//===----------------------------------------------------------------------===//
// HWModuleOp
//===----------------------------------------------------------------------===/

/// Return true if this is an hw.module, external module, generated module etc.
bool hw::isAnyModule(Operation *module) {
  return isa<HWModuleOp>(module) || isa<HWModuleExternOp>(module) ||
         isa<HWModuleGeneratedOp>(module);
}

/// Return true if isAnyModule or instance.
bool hw::isAnyModuleOrInstance(Operation *moduleOrInstance) {
  return isAnyModule(moduleOrInstance) || isa<InstanceOp>(moduleOrInstance);
}

/// Return the signature for a module as a function type from the module itself
/// or from an hw::InstanceOp.
FunctionType hw::getModuleType(Operation *moduleOrInstance) {
  if (auto instance = dyn_cast<InstanceOp>(moduleOrInstance)) {
    SmallVector<Type> inputs(instance->getOperandTypes());
    SmallVector<Type> results(instance->getResultTypes());
    return FunctionType::get(instance->getContext(), inputs, results);
  }

  assert(isAnyModule(moduleOrInstance) &&
         "must be called on instance or module");
  auto typeAttr =
      moduleOrInstance->getAttrOfType<TypeAttr>(HWModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
}

/// Return the name to use for the Verilog module that we're referencing
/// here.  This is typically the symbol, but can be overridden with the
/// verilogName attribute.
StringAttr hw::getVerilogModuleNameAttr(Operation *module) {
  auto nameAttr = module->getAttrOfType<StringAttr>("verilogName");
  if (nameAttr)
    return nameAttr;

  return module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
}

/// Return the port name for the specified argument or result.
StringAttr hw::getModuleArgumentNameAttr(Operation *module, size_t argNo) {
  auto argNames = module->getAttrOfType<ArrayAttr>("argNames");
  // Tolerate malformed IR here to enable debug printing etc.
  if (argNames && argNo < argNames.size())
    return argNames[argNo].cast<StringAttr>();
  return StringAttr();
}

StringAttr hw::getModuleResultNameAttr(Operation *module, size_t resultNo) {
  auto resultNames = module->getAttrOfType<ArrayAttr>("resultNames");
  // Tolerate malformed IR here to enable debug printing etc.
  if (resultNames && resultNo < resultNames.size())
    return resultNames[resultNo].cast<StringAttr>();
  return StringAttr();
}

void hw::setModuleArgumentNames(Operation *module, ArrayRef<Attribute> names) {
  assert(isAnyModule(module) && "Must be called on a module");
  assert(getModuleType(module).getNumInputs() == names.size() &&
         "incorrect number of arguments names specified");
  module->setAttr("argNames", ArrayAttr::get(module->getContext(), names));
}

void hw::setModuleResultNames(Operation *module, ArrayRef<Attribute> names) {
  assert(isAnyModule(module) && "Must be called on a module");
  assert(getModuleType(module).getNumResults() == names.size() &&
         "incorrect number of arguments names specified");
  module->setAttr("resultNames", ArrayAttr::get(module->getContext(), names));
}

// Flag for parsing different module types
enum ExternModKind { PlainMod, ExternMod, GenMod };

static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, const ModulePortInfo &ports,
                        ArrayAttr parameters,
                        ArrayRef<NamedAttribute> attributes,
                        StringAttr comment) {
  using namespace mlir::function_interface_impl;

  // Add an attribute for the name.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);

  SmallVector<Attribute> argNames, resultNames;
  SmallVector<Type, 4> argTypes, resultTypes;
  SmallVector<Attribute> argAttrs, resultAttrs;
  auto exportPortIdent = StringAttr::get(builder.getContext(), "hw.exportPort");

  for (auto elt : ports.inputs) {
    if (elt.direction == PortDirection::INOUT && !elt.type.isa<hw::InOutType>())
      elt.type = hw::InOutType::get(elt.type);
    argTypes.push_back(elt.type);
    argNames.push_back(elt.name);
    Attribute attr;
    if (elt.sym && !elt.sym.getValue().empty())
      attr = builder.getDictionaryAttr(
          {{exportPortIdent, FlatSymbolRefAttr::get(elt.sym)}});
    else
      attr = builder.getDictionaryAttr({});
    argAttrs.push_back(attr);
  }

  for (auto elt : ports.outputs) {
    resultTypes.push_back(elt.type);
    resultNames.push_back(elt.name);
    Attribute attr;
    if (elt.sym && !elt.sym.getValue().empty())
      attr = builder.getDictionaryAttr(
          {{exportPortIdent, FlatSymbolRefAttr::get(elt.sym)}});
    else
      attr = builder.getDictionaryAttr({});
    resultAttrs.push_back(attr);
  }

  // Allow clients to pass in null for the parameters list.
  if (!parameters)
    parameters = builder.getArrayAttr({});

  // Record the argument and result types as an attribute.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  result.addAttribute("argNames", builder.getArrayAttr(argNames));
  result.addAttribute("resultNames", builder.getArrayAttr(resultNames));
  result.addAttribute(mlir::function_interface_impl::getArgDictAttrName(),
                      builder.getArrayAttr(argAttrs));
  result.addAttribute(mlir::function_interface_impl::getResultDictAttrName(),
                      builder.getArrayAttr(resultAttrs));
  result.addAttribute("parameters", parameters);
  if (!comment)
    comment = builder.getStringAttr("");
  result.addAttribute("comment", comment);
  result.addAttributes(attributes);
  result.addRegion();
}

/// Internal implementation of argument/result insertion and removal on modules.
static void modifyModuleArgs(
    MLIRContext *context, ArrayRef<std::pair<unsigned, PortInfo>> insertArgs,
    ArrayRef<unsigned> removeArgs, ArrayRef<Attribute> oldArgNames,
    ArrayRef<Type> oldArgTypes, ArrayRef<Attribute> oldArgAttrs,
    SmallVector<Attribute> &newArgNames, SmallVector<Type> &newArgTypes,
    SmallVector<Attribute> &newArgAttrs) {

#ifndef NDEBUG
  // Check that the `insertArgs` and `removeArgs` indices are in ascending
  // order.
  assert(llvm::is_sorted(insertArgs,
                         [](auto &a, auto &b) { return a.first < b.first; }) &&
         "insertArgs must be in ascending order");
  assert(llvm::is_sorted(removeArgs, [](auto &a, auto &b) { return a < b; }) &&
         "removeArgs must be in ascending order");
#endif

  auto oldArgCount = oldArgTypes.size();
  auto newArgCount = oldArgCount + insertArgs.size() - removeArgs.size();
  assert((int)newArgCount >= 0);

  newArgNames.reserve(newArgCount);
  newArgTypes.reserve(newArgCount);
  newArgAttrs.reserve(newArgCount);

  auto exportPortAttrName = StringAttr::get(context, "hw.exportPort");
  auto emptyDictAttr = DictionaryAttr::get(context, {});

  for (unsigned argIdx = 0; argIdx <= oldArgCount; ++argIdx) {
    // Insert new ports at this position.
    while (!insertArgs.empty() && insertArgs[0].first == argIdx) {
      auto port = insertArgs[0].second;
      if (port.direction == PortDirection::INOUT && !port.type.isa<InOutType>())
        port.type = InOutType::get(port.type);
      Attribute attr =
          (port.sym && !port.sym.getValue().empty())
              ? DictionaryAttr::get(
                    context,
                    {{exportPortAttrName, FlatSymbolRefAttr::get(port.sym)}})
              : emptyDictAttr;
      newArgNames.push_back(port.name);
      newArgTypes.push_back(port.type);
      newArgAttrs.push_back(attr);
      insertArgs = insertArgs.drop_front();
    }
    if (argIdx == oldArgCount)
      break;

    // Migrate the old port at this position.
    bool removed = false;
    while (!removeArgs.empty() && removeArgs[0] == argIdx) {
      removeArgs = removeArgs.drop_front();
      removed = true;
    }
    if (!removed) {
      newArgNames.push_back(oldArgNames[argIdx]);
      newArgTypes.push_back(oldArgTypes[argIdx]);
      newArgAttrs.push_back(oldArgAttrs.empty() ? emptyDictAttr
                                                : oldArgAttrs[argIdx]);
    }
  }

  assert(newArgNames.size() == newArgCount);
  assert(newArgTypes.size() == newArgCount);
  assert(newArgAttrs.size() == newArgCount);
}

/// Insert and remove ports of a module. The insertion and removal indices must
/// be in ascending order. The indices refer to the port positions before any
/// insertion or removal occurs. Ports inserted at the same index will appear in
/// the module in the same order as they were listed in the `insert*` array.
///
/// The operation must be any of the module-like operations.
void hw::modifyModulePorts(
    Operation *op, ArrayRef<std::pair<unsigned, PortInfo>> insertInputs,
    ArrayRef<std::pair<unsigned, PortInfo>> insertOutputs,
    ArrayRef<unsigned> removeInputs, ArrayRef<unsigned> removeOutputs) {
  auto moduleOp = cast<mlir::FunctionOpInterface>(op);

  auto arrayOrEmpty = [](ArrayAttr attr) {
    return attr ? attr.getValue() : ArrayRef<Attribute>{};
  };

  // Dig up the old argument and result data.
  ArrayRef<Attribute> oldArgNames =
      moduleOp->getAttrOfType<ArrayAttr>("argNames").getValue();
  ArrayRef<Type> oldArgTypes = moduleOp.getArgumentTypes();
  ArrayRef<Attribute> oldArgAttrs =
      arrayOrEmpty(moduleOp->getAttrOfType<ArrayAttr>(
          mlir::function_interface_impl::getArgDictAttrName()));

  ArrayRef<Attribute> oldResultNames =
      moduleOp->getAttrOfType<ArrayAttr>("resultNames").getValue();
  ArrayRef<Type> oldResultTypes = moduleOp.getResultTypes();
  ArrayRef<Attribute> oldResultAttrs =
      arrayOrEmpty(moduleOp->getAttrOfType<ArrayAttr>(
          mlir::function_interface_impl::getResultDictAttrName()));

  // Modify the ports.
  SmallVector<Attribute> newArgNames, newResultNames;
  SmallVector<Type> newArgTypes, newResultTypes;
  SmallVector<Attribute> newArgAttrs, newResultAttrs;

  modifyModuleArgs(moduleOp.getContext(), insertInputs, removeInputs,
                   oldArgNames, oldArgTypes, oldArgAttrs, newArgNames,
                   newArgTypes, newArgAttrs);

  modifyModuleArgs(moduleOp.getContext(), insertOutputs, removeOutputs,
                   oldResultNames, oldResultTypes, oldResultAttrs,
                   newResultNames, newResultTypes, newResultAttrs);

  // Update the module operation types and attributes.
  moduleOp.setType(
      FunctionType::get(moduleOp.getContext(), newArgTypes, newResultTypes));
  moduleOp->setAttr("argNames",
                    ArrayAttr::get(moduleOp.getContext(), newArgNames));
  moduleOp->setAttr("resultNames",
                    ArrayAttr::get(moduleOp.getContext(), newResultNames));
  moduleOp->setAttr(mlir::function_interface_impl::getArgDictAttrName(),
                    ArrayAttr::get(moduleOp.getContext(), newArgAttrs));
  moduleOp->setAttr(mlir::function_interface_impl::getResultDictAttrName(),
                    ArrayAttr::get(moduleOp.getContext(), newResultAttrs));
}

void HWModuleOp::build(OpBuilder &builder, OperationState &result,
                       StringAttr name, const ModulePortInfo &ports,
                       ArrayAttr parameters,
                       ArrayRef<NamedAttribute> attributes,
                       StringAttr comment) {
  buildModule(builder, result, name, ports, parameters, attributes, comment);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports.inputs)
    body->addArgument(elt.type, builder.getUnknownLoc());

  HWModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void HWModuleOp::build(OpBuilder &builder, OperationState &result,
                       StringAttr name, ArrayRef<PortInfo> ports,
                       ArrayAttr parameters,
                       ArrayRef<NamedAttribute> attributes,
                       StringAttr comment) {
  build(builder, result, name, ModulePortInfo(ports), parameters, attributes,
        comment);
}

void HWModuleOp::modifyPorts(
    ArrayRef<std::pair<unsigned, PortInfo>> insertInputs,
    ArrayRef<std::pair<unsigned, PortInfo>> insertOutputs,
    ArrayRef<unsigned> eraseInputs, ArrayRef<unsigned> eraseOutputs) {
  hw::modifyModulePorts(*this, insertInputs, insertOutputs, eraseInputs,
                        eraseOutputs);
}

/// Return the name to use for the Verilog module that we're referencing
/// here.  This is typically the symbol, but can be overridden with the
/// verilogName attribute.
StringAttr HWModuleExternOp::getVerilogModuleNameAttr() {
  if (auto vName = verilogNameAttr())
    return vName;

  return (*this)->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
}

StringAttr HWModuleGeneratedOp::getVerilogModuleNameAttr() {
  if (auto vName = verilogNameAttr()) {
    return vName;
  }
  return (*this)->getAttrOfType<StringAttr>(
      ::mlir::SymbolTable::getSymbolAttrName());
}

void HWModuleExternOp::build(OpBuilder &builder, OperationState &result,
                             StringAttr name, const ModulePortInfo &ports,
                             StringRef verilogName, ArrayAttr parameters,
                             ArrayRef<NamedAttribute> attributes) {
  buildModule(builder, result, name, ports, parameters, attributes, {});

  if (!verilogName.empty())
    result.addAttribute("verilogName", builder.getStringAttr(verilogName));
}

void HWModuleExternOp::build(OpBuilder &builder, OperationState &result,
                             StringAttr name, ArrayRef<PortInfo> ports,
                             StringRef verilogName, ArrayAttr parameters,
                             ArrayRef<NamedAttribute> attributes) {
  build(builder, result, name, ModulePortInfo(ports), verilogName, parameters,
        attributes);
}

void HWModuleExternOp::modifyPorts(
    ArrayRef<std::pair<unsigned, PortInfo>> insertInputs,
    ArrayRef<std::pair<unsigned, PortInfo>> insertOutputs,
    ArrayRef<unsigned> eraseInputs, ArrayRef<unsigned> eraseOutputs) {
  hw::modifyModulePorts(*this, insertInputs, insertOutputs, eraseInputs,
                        eraseOutputs);
}

void HWModuleGeneratedOp::build(OpBuilder &builder, OperationState &result,
                                FlatSymbolRefAttr genKind, StringAttr name,
                                const ModulePortInfo &ports,
                                StringRef verilogName, ArrayAttr parameters,
                                ArrayRef<NamedAttribute> attributes) {
  buildModule(builder, result, name, ports, parameters, attributes, {});
  result.addAttribute("generatorKind", genKind);
  if (!verilogName.empty())
    result.addAttribute("verilogName", builder.getStringAttr(verilogName));
}

void HWModuleGeneratedOp::build(OpBuilder &builder, OperationState &result,
                                FlatSymbolRefAttr genKind, StringAttr name,
                                ArrayRef<PortInfo> ports, StringRef verilogName,
                                ArrayAttr parameters,
                                ArrayRef<NamedAttribute> attributes) {
  build(builder, result, genKind, name, ModulePortInfo(ports), verilogName,
        parameters, attributes);
}

void HWModuleGeneratedOp::modifyPorts(
    ArrayRef<std::pair<unsigned, PortInfo>> insertInputs,
    ArrayRef<std::pair<unsigned, PortInfo>> insertOutputs,
    ArrayRef<unsigned> eraseInputs, ArrayRef<unsigned> eraseOutputs) {
  hw::modifyModulePorts(*this, insertInputs, insertOutputs, eraseInputs,
                        eraseOutputs);
}

/// Return an encapsulated set of information about input and output ports of
/// the specified module or instance.  The input ports always come before the
/// output ports in the list.
ModulePortInfo hw::getModulePortInfo(Operation *op) {
  assert(isAnyModuleOrInstance(op) &&
         "Can only get module ports from an instance or module");

  SmallVector<PortInfo> inputs, outputs;
  auto argTypes = getModuleType(op).getInputs();

  auto argNames = op->getAttrOfType<ArrayAttr>("argNames");
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    auto direction = isInOut ? PortDirection::INOUT : PortDirection::INPUT;
    inputs.push_back(
        {argNames[i].cast<StringAttr>(), direction, type, i, getArgSym(op, i)});
  }

  auto resultNames = op->getAttrOfType<ArrayAttr>("resultNames");
  auto resultTypes = getModuleType(op).getResults();
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    outputs.push_back({resultNames[i].cast<StringAttr>(), PortDirection::OUTPUT,
                       resultTypes[i], i, getResultSym(op, i)});
  }
  return ModulePortInfo(inputs, outputs);
}

/// Return an encapsulated set of information about input and output ports of
/// the specified module or instance.  The input ports always come before the
/// output ports in the list.
SmallVector<PortInfo> hw::getAllModulePortInfos(Operation *op) {
  assert(isAnyModuleOrInstance(op) &&
         "Can only get module ports from an instance or module");

  SmallVector<PortInfo> results;
  auto argTypes = getModuleType(op).getInputs();
  auto argNames = op->getAttrOfType<ArrayAttr>("argNames");
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    auto direction = isInOut ? PortDirection::INOUT : PortDirection::INPUT;
    results.push_back(
        {argNames[i].cast<StringAttr>(), direction, type, i, getArgSym(op, i)});
  }

  auto resultNames = op->getAttrOfType<ArrayAttr>("resultNames");
  auto resultTypes = getModuleType(op).getResults();
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    results.push_back({resultNames[i].cast<StringAttr>(), PortDirection::OUTPUT,
                       resultTypes[i], i, getResultSym(op, i)});
  }
  return results;
}

/// Return the PortInfo for the specified input or inout port.
PortInfo hw::getModuleInOrInoutPort(Operation *op, size_t idx) {
  auto argTypes = getModuleType(op).getInputs();
  auto argNames = op->getAttrOfType<ArrayAttr>("argNames");
  bool isInOut = false;
  auto type = argTypes[idx];

  if (auto inout = type.dyn_cast<InOutType>()) {
    isInOut = true;
    type = inout.getElementType();
  }

  auto direction = isInOut ? PortDirection::INOUT : PortDirection::INPUT;
  return {argNames[idx].cast<StringAttr>(), direction, type, idx,
          getArgSym(op, idx)};
}

/// Return the PortInfo for the specified output port.
PortInfo hw::getModuleOutputPort(Operation *op, size_t idx) {
  auto resultNames = op->getAttrOfType<ArrayAttr>("resultNames");
  auto resultTypes = getModuleType(op).getResults();
  assert(idx < resultNames.size() && "invalid result number");
  return {resultNames[idx].cast<StringAttr>(), PortDirection::OUTPUT,
          resultTypes[idx], idx, getResultSym(op, idx)};
}

static bool hasAttribute(StringRef name, ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs)
    if (argAttr.getName() == name)
      return true;
  return false;
}

/// Parse an parameter list if present.
/// module-parameter-list ::= `<` parameter-decl (`,` parameter-decl)* `>`
/// parameter-decl ::= identifier `:` type
/// parameter-decl ::= identifier `:` type `=` attribute
///
static ParseResult parseOptionalParameters(OpAsmParser &parser,
                                           SmallVector<Attribute> &parameters) {

  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::OptionalLessGreater, [&]() {
        std::string name;
        Type type;
        Attribute value;

        if (parser.parseKeywordOrString(&name) || parser.parseColonType(type))
          return failure();

        // Parse the default value if present.
        if (succeeded(parser.parseOptionalEqual())) {
          if (parser.parseAttribute(value, type))
            return failure();
        }

        auto &builder = parser.getBuilder();
        parameters.push_back(ParamDeclAttr::get(builder.getContext(),
                                                builder.getStringAttr(name),
                                                TypeAttr::get(type), value));
        return success();
      });
}

static ParseResult parseHWModuleOp(OpAsmParser &parser, OperationState &result,
                                   ExternModKind modKind = PlainMod) {

  using namespace mlir::function_interface_impl;

  auto loc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute> parameters;
  auto &builder = parser.getBuilder();

  // Parse the visibility attribute.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  FlatSymbolRefAttr kindAttr;
  if (modKind == GenMod) {
    if (parser.parseComma() ||
        parser.parseAttribute(kindAttr, "generatorKind", result.attributes)) {
      return failure();
    }
  }

  // Parse the function signature.
  bool isVariadic = false;
  SmallVector<Attribute> resultNames;
  if (parseOptionalParameters(parser, parameters) ||
      module_like_impl::parseModuleFunctionSignature(
          parser, entryArgs, isVariadic, resultTypes, resultAttrs,
          resultNames) ||
      // If function attributes are present, parse them.
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.
  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);

  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  auto *context = result.getContext();

  if (hasAttribute("resultNames", result.attributes) ||
      hasAttribute("parameters", result.attributes)) {
    parser.emitError(
        loc, "explicit `resultNames` / `parameters` attributes not allowed");
    return failure();
  }

  // Use the argument and result names if not already specified.
  SmallVector<Attribute> argNames;
  if (!entryArgs.empty()) {
    for (auto &arg : entryArgs)
      argNames.push_back(
          module_like_impl::getPortNameAttr(context, arg.ssaName.name));
  } else if (!argTypes.empty()) {
    // The parser returns empty names in a special way.
    argNames.assign(argTypes.size(), StringAttr::get(context, ""));
  }

  // An explicit `argNames` attribute overrides the MLIR names.  This is how
  // we represent port names that aren't valid MLIR identifiers.  Result and
  // parameter names are printed quoted when they aren't valid identifiers, so
  // they don't need this affordance.
  if (!hasAttribute("argNames", result.attributes))
    result.addAttribute("argNames", ArrayAttr::get(context, argNames));
  result.addAttribute("resultNames", ArrayAttr::get(context, resultNames));
  result.addAttribute("parameters", ArrayAttr::get(context, parameters));
  if (!hasAttribute("comment", result.attributes))
    result.addAttribute("comment", StringAttr::get(context, ""));

  assert(resultAttrs.size() == resultTypes.size());

  // Add the attributes to the function arguments.
  addArgAndResultAttrs(builder, result, entryArgs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (modKind == PlainMod) {
    if (parser.parseRegion(*body, entryArgs))
      return failure();

    HWModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  }
  return success();
}

ParseResult HWModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseHWModuleOp(parser, result);
}

ParseResult HWModuleExternOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  return parseHWModuleOp(parser, result, ExternMod);
}

ParseResult HWModuleGeneratedOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  return parseHWModuleOp(parser, result, GenMod);
}

FunctionType getHWModuleOpType(Operation *op) {
  auto typeAttr = op->getAttrOfType<TypeAttr>(HWModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
}

/// Print a parameter list for a module or instance.
static void printParameterList(ArrayAttr parameters, OpAsmPrinter &p) {
  if (parameters.empty())
    return;

  p << '<';
  llvm::interleaveComma(parameters, p, [&](Attribute param) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    p << paramAttr.getName().getValue() << ": " << paramAttr.getType();
    if (auto value = paramAttr.getValue()) {
      p << " = ";
      p.printAttributeWithoutType(value);
    }
  });
  p << '>';
}

static void printModuleOp(OpAsmPrinter &p, Operation *op,
                          ExternModKind modKind) {
  using namespace mlir::function_interface_impl;

  FunctionType fnType = getHWModuleOpType(op);
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  p << ' ';

  // Print the visibility of the module.
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility = op->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';

  // Print the operation and the function name.
  p.printSymbolName(SymbolTable::getSymbolName(op).getValue());
  if (modKind == GenMod) {
    p << ", ";
    p.printSymbolName(cast<HWModuleGeneratedOp>(op).generatorKind());
  }

  // Print the parameter list if present.
  printParameterList(op->getAttrOfType<ArrayAttr>("parameters"), p);

  bool needArgNamesAttr = false;
  module_like_impl::printModuleSignature(p, op, argTypes, /*isVariadic=*/false,
                                         resultTypes, needArgNamesAttr);

  SmallVector<StringRef, 3> omittedAttrs;
  if (modKind == GenMod)
    omittedAttrs.push_back("generatorKind");
  if (!needArgNamesAttr)
    omittedAttrs.push_back("argNames");
  omittedAttrs.push_back("resultNames");
  omittedAttrs.push_back("parameters");
  omittedAttrs.push_back(visibilityAttrName);
  if (op->getAttrOfType<StringAttr>("comment").getValue().empty())
    omittedAttrs.push_back("comment");

  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          omittedAttrs);
}

void HWModuleExternOp::print(OpAsmPrinter &p) {
  printModuleOp(p, *this, ExternMod);
}
void HWModuleGeneratedOp::print(OpAsmPrinter &p) {
  printModuleOp(p, *this, GenMod);
}

void HWModuleOp::print(OpAsmPrinter &p) {
  printModuleOp(p, *this, PlainMod);

  // Print the body if this is not an external function.
  Region &body = getBody();
  if (!body.empty()) {
    p << " ";
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

static LogicalResult verifyModuleCommon(Operation *module) {
  assert(isAnyModule(module) &&
         "verifier hook should only be called on modules");

  auto moduleType = getModuleType(module);
  auto argNames = module->getAttrOfType<ArrayAttr>("argNames");
  auto resultNames = module->getAttrOfType<ArrayAttr>("resultNames");
  if (argNames.size() != moduleType.getNumInputs())
    return module->emitOpError("incorrect number of argument names");
  if (resultNames.size() != moduleType.getNumResults())
    return module->emitOpError("incorrect number of result names");

  SmallPtrSet<Attribute, 4> paramNames;

  // Check parameter default values are sensible.
  for (auto param : module->getAttrOfType<ArrayAttr>("parameters")) {
    auto paramAttr = param.cast<ParamDeclAttr>();

    // Check that we don't have any redundant parameter names.  These are
    // resolved by string name: reuse of the same name would cause ambiguities.
    if (!paramNames.insert(paramAttr.getName()).second)
      return module->emitOpError("parameter ")
             << paramAttr << " has the same name as a previous parameter";

    // Default values are allowed to be missing, check them if present.
    auto value = paramAttr.getValue();
    if (!value)
      continue;

    if (value.getType() != paramAttr.getType().getValue())
      return module->emitOpError("parameter ")
             << paramAttr << " should have type "
             << paramAttr.getType().getValue() << "; has type "
             << value.getType();

    // Verify that this is a valid parameter value, disallowing parameter
    // references.  We could allow parameters to refer to each other in the
    // future with lexical ordering if there is a need.
    if (failed(checkParameterInContext(value, module, module,
                                       /*disallowParamRefs=*/true)))
      return failure();
  }
  return success();
}

LogicalResult HWModuleOp::verify() { return verifyModuleCommon(*this); }

LogicalResult HWModuleExternOp::verify() { return verifyModuleCommon(*this); }

void HWModuleOp::insertOutputs(unsigned index,
                               ArrayRef<std::pair<StringAttr, Value>> outputs) {

  auto output = cast<OutputOp>(getBodyBlock()->getTerminator());
  assert(index <= output->getNumOperands() && "invalid output index");

  // Rewrite the port list of the module.
  SmallVector<std::pair<unsigned, PortInfo>> indexedNewPorts;
  for (auto &[name, value] : outputs) {
    PortInfo port;
    port.name = name;
    port.direction = PortDirection::OUTPUT;
    port.type = value.getType();
    indexedNewPorts.emplace_back(index, port);
  }
  insertPorts({}, indexedNewPorts);

  // Rewrite the output op.
  for (auto &[name, value] : outputs)
    output->insertOperands(index++, value);
}

void HWModuleOp::appendOutputs(ArrayRef<std::pair<StringAttr, Value>> outputs) {
  return insertOutputs(getResultTypes().size(), outputs);
}

void HWModuleOp::getAsmBlockArgumentNames(mlir::Region &region,
                                          mlir::OpAsmSetValueNameFn setNameFn) {
  getAsmBlockArgumentNamesImpl(region, setNameFn);
}

void HWModuleExternOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  getAsmBlockArgumentNamesImpl(region, setNameFn);
}

/// Lookup the generator for the symbol.  This returns null on
/// invalid IR.
Operation *HWModuleGeneratedOp::getGeneratorKindOp() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol(generatorKind());
}

LogicalResult
HWModuleGeneratedOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *referencedKind =
      symbolTable.lookupNearestSymbolFrom(*this, generatorKindAttr());

  if (referencedKind == nullptr)
    return emitError("Cannot find generator definition '")
           << generatorKind() << "'";

  if (!isa<HWGeneratorSchemaOp>(referencedKind))
    return emitError("Symbol resolved to '")
           << referencedKind->getName()
           << "' which is not a HWGeneratorSchemaOp";

  auto referencedKindOp = dyn_cast<HWGeneratorSchemaOp>(referencedKind);
  auto paramRef = referencedKindOp.requiredAttrs();
  auto dict = (*this)->getAttrDictionary();
  for (auto str : paramRef) {
    auto strAttr = str.dyn_cast<StringAttr>();
    if (!strAttr)
      return emitError("Unknown attribute type, expected a string");
    if (!dict.get(strAttr.getValue()))
      return emitError("Missing attribute '") << strAttr.getValue() << "'";
  }

  return success();
}

LogicalResult HWModuleGeneratedOp::verify() {
  return verifyModuleCommon(*this);
}

void HWModuleGeneratedOp::getAsmBlockArgumentNames(
    mlir::Region &region, mlir::OpAsmSetValueNameFn setNameFn) {
  getAsmBlockArgumentNamesImpl(region, setNameFn);
}

LogicalResult HWModuleOp::verifyBody() { return success(); }

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

/// Create a instance that refers to a known module.
void InstanceOp::build(OpBuilder &builder, OperationState &result,
                       Operation *module, StringAttr name,
                       ArrayRef<Value> inputs, ArrayAttr parameters,
                       StringAttr sym_name) {
  assert(isAnyModule(module) && "Can only reference a module");

  if (!parameters)
    parameters = builder.getArrayAttr({});

  FunctionType modType = getModuleType(module);
  build(builder, result, modType.getResults(), name,
        FlatSymbolRefAttr::get(SymbolTable::getSymbolName(module)), inputs,
        module->getAttrOfType<ArrayAttr>("argNames"),
        module->getAttrOfType<ArrayAttr>("resultNames"), parameters, sym_name);
}

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *InstanceOp::getReferencedModule(const HWSymbolCache *cache) {
  if (cache)
    if (auto *result = cache->getDefinition(moduleNameAttr()))
      return result;

  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol(moduleName());
}

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module = symbolTable.lookupNearestSymbolFrom(*this, moduleNameAttr());
  if (module == nullptr)
    return emitError("Cannot find module definition '") << moduleName() << "'";

  // It must be some sort of module.
  if (!isAnyModule(module))
    return emitError("symbol reference '")
           << moduleName() << "' isn't a module";

  // Check that input and result types are consistent with the referenced
  // module.
  // Emit an error message on the instance, with a note indicating which module
  // is being referenced.
  auto emitError =
      [&](std::function<void(InFlightDiagnostic & diag)> fn) -> LogicalResult {
    auto diag = emitOpError();
    fn(diag);
    diag.attachNote(module->getLoc()) << "module declared here";
    return failure();
  };

  // Make sure our port and result names match.
  ArrayAttr argNames = argNamesAttr();
  ArrayAttr modArgNames = module->getAttrOfType<ArrayAttr>("argNames");

  // Check operand types first.
  auto numOperands = getOperation()->getNumOperands();
  auto expectedOperandTypes = getModuleType(module).getInputs();

  if (expectedOperandTypes.size() != numOperands)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of operands; expected "
           << expectedOperandTypes.size() << " but got " << numOperands;
    });

  if (argNames.size() != numOperands)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of input port names; expected " << numOperands
           << " but got " << argNames.size();
    });

  for (size_t i = 0; i != numOperands; ++i) {
    auto expectedType =
        evaluateParametricType(getLoc(), parameters(), expectedOperandTypes[i]);
    if (failed(expectedType))
      return emitError([&](auto &diag) {
        diag << "failed to resolve parametric input of instantiated module";
      });
    auto operandType = getOperand(i).getType();
    if (operandType != expectedType.getValue()) {
      return emitError([&](auto &diag) {
        diag << "operand type #" << i << " must be " << expectedType.getValue()
             << ", but got " << operandType;
      });
    }

    if (argNames[i] != modArgNames[i])
      return emitError([&](auto &diag) {
        diag << "input label #" << i << " must be " << modArgNames[i]
             << ", but got " << argNames[i];
      });
  }

  // Check result types and labels.
  auto numResults = getOperation()->getNumResults();
  auto expectedResultTypes = getModuleType(module).getResults();
  ArrayAttr resultNames = resultNamesAttr();
  ArrayAttr modResultNames = module->getAttrOfType<ArrayAttr>("resultNames");

  if (expectedResultTypes.size() != numResults)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of results; expected "
           << expectedResultTypes.size() << " but got " << numResults;
    });
  if (resultNames.size() != numResults)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of results port labels; expected "
           << numResults << " but got " << resultNames.size();
    });

  for (size_t i = 0; i != numResults; ++i) {
    auto expectedType =
        evaluateParametricType(getLoc(), parameters(), expectedResultTypes[i]);
    if (failed(expectedType))
      return emitError([&](auto &diag) {
        diag << "failed to resolve parametric input of instantiated module";
      });
    auto resultType = getResult(i).getType();
    if (resultType != expectedType.getValue())
      return emitError([&](auto &diag) {
        diag << "result type #" << i << " must be " << expectedType.getValue()
             << ", but got " << resultType;
      });

    if (resultNames[i] != modResultNames[i])
      return emitError([&](auto &diag) {
        diag << "input label #" << i << " must be " << modResultNames[i]
             << ", but got " << resultNames[i];
      });
  }

  // Check parameters match up.
  ArrayAttr parameters = this->parameters();
  ArrayAttr modParameters = module->getAttrOfType<ArrayAttr>("parameters");
  auto numParameters = parameters.size();
  if (numParameters != modParameters.size())
    return emitError([&](auto &diag) {
      diag << "expected " << modParameters.size() << " parameters but had "
           << numParameters;
    });

  for (size_t i = 0; i != numParameters; ++i) {
    auto param = parameters[i].cast<ParamDeclAttr>();
    auto modParam = modParameters[i].cast<ParamDeclAttr>();

    auto paramName = param.getName();
    if (paramName != modParam.getName())
      return emitError([&](auto &diag) {
        diag << "parameter #" << i << " should have name " << modParam.getName()
             << " but has name " << paramName;
      });

    if (param.getType() != modParam.getType())
      return emitError([&](auto &diag) {
        diag << "parameter " << paramName << " should have type "
             << modParam.getType() << " but has type " << param.getType();
      });

    // All instance parameters must have a value.  Specify the same value as
    // a module's default value if you want the default.
    if (!param.getValue())
      return emitOpError("parameter ") << paramName << " must have a value";
  }

  return success();
}

LogicalResult InstanceOp::verify() {
  // Check that all the parameter values specified to the instance are
  // structurally valid.
  for (auto param : parameters()) {
    auto paramAttr = param.cast<ParamDeclAttr>();
    auto value = paramAttr.getValue();
    // The SymbolUses verifier which checks that this exists may not have been
    // run yet. Let it issue the error.
    if (!value)
      continue;

    if (value.getType() != paramAttr.getType().getValue())
      return emitOpError("parameter ") << paramAttr << " should have type "
                                       << paramAttr.getType().getValue()
                                       << "; has type " << value.getType();

    if (failed(checkParameterInContext(
            value, (*this)->getParentOfType<HWModuleOp>(), *this)))
      return failure();
  }
  return success();
}

ParseResult InstanceOp::parse(OpAsmParser &parser, OperationState &result) {
  auto *context = result.getContext();
  StringAttr instanceNameAttr;
  StringAttr sym_nameAttr;
  FlatSymbolRefAttr moduleNameAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  SmallVector<Type> inputsTypes;
  SmallVector<Type> allResultTypes;
  SmallVector<Attribute> argNames, resultNames, parameters;
  auto noneType = parser.getBuilder().getType<NoneType>();

  if (parser.parseAttribute(instanceNameAttr, noneType, "instanceName",
                            result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("sym"))) {
    // Parsing an optional symbol name doesn't fail, so no need to check the
    // result.
    (void)parser.parseOptionalSymbolName(
        sym_nameAttr, InnerName::getInnerNameAttrName(), result.attributes);
  }

  auto parseInputPort = [&]() -> ParseResult {
    std::string portName;
    if (parser.parseKeywordOrString(&portName))
      return failure();
    argNames.push_back(StringAttr::get(context, portName));
    inputsOperands.push_back({});
    inputsTypes.push_back({});
    return failure(parser.parseColon() ||
                   parser.parseOperand(inputsOperands.back()) ||
                   parser.parseColon() || parser.parseType(inputsTypes.back()));
  };

  auto parseResultPort = [&]() -> ParseResult {
    std::string portName;
    if (parser.parseKeywordOrString(&portName))
      return failure();
    resultNames.push_back(StringAttr::get(parser.getContext(), portName));
    allResultTypes.push_back({});
    return parser.parseColonType(allResultTypes.back());
  };

  llvm::SMLoc parametersLoc, inputsOperandsLoc;
  if (parser.parseAttribute(moduleNameAttr, noneType, "moduleName",
                            result.attributes) ||
      parser.getCurrentLocation(&parametersLoc) ||
      parseOptionalParameters(parser, parameters) ||
      parser.getCurrentLocation(&inputsOperandsLoc) ||
      parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseInputPort) ||
      parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.parseArrow() ||
      parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseResultPort) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  result.addAttribute("argNames", parser.getBuilder().getArrayAttr(argNames));
  result.addAttribute("resultNames",
                      parser.getBuilder().getArrayAttr(resultNames));
  result.addAttribute("parameters",
                      parser.getBuilder().getArrayAttr(parameters));
  result.addTypes(allResultTypes);
  return success();
}

void InstanceOp::print(OpAsmPrinter &p) {
  ModulePortInfo portInfo = getModulePortInfo(*this);
  size_t nextInputPort = 0, nextOutputPort = 0;

  auto printPortName = [&](size_t &nextPort, SmallVector<PortInfo> &portList) {
    // Allow printing mangled instances.
    if (nextPort >= portList.size()) {
      p << "<corrupt port>: ";
      return;
    }

    p.printKeywordOrString(portList[nextPort++].name.getValue());
    p << ": ";
  };

  p << ' ';
  p.printAttributeWithoutType(instanceNameAttr());
  if (auto attr = inner_symAttr()) {
    p << " sym ";
    p.printSymbolName(attr.getValue());
  }
  p << ' ';
  p.printAttributeWithoutType(moduleNameAttr());
  printParameterList(parameters(), p);
  p << '(';
  llvm::interleaveComma(inputs(), p, [&](Value op) {
    printPortName(nextInputPort, portInfo.inputs);
    p << op << ": " << op.getType();
  });
  p << ") -> (";
  llvm::interleaveComma(getResults(), p, [&](Value res) {
    printPortName(nextOutputPort, portInfo.outputs);
    p << res.getType();
  });
  p << ')';
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{"instanceName", InnerName::getInnerNameAttrName(),
                       "moduleName", "argNames", "resultNames", "parameters"});
}

/// Return the name of the specified input port or null if it cannot be
/// determined.
StringAttr InstanceOp::getArgumentName(size_t idx) {
  auto names = argNames();
  // Tolerate malformed IR here to enable debug printing etc.
  if (names && idx < names.size())
    return names[idx].cast<StringAttr>();
  return StringAttr();
}

/// Return the name of the specified result or null if it cannot be
/// determined.
StringAttr InstanceOp::getResultName(size_t idx) {
  auto names = resultNames();
  // Tolerate malformed IR here to enable debug printing etc.
  if (names && idx < names.size())
    return names[idx].cast<StringAttr>();
  return StringAttr();
}

/// Change the name of the specified input port.
void InstanceOp::setArgumentName(size_t i, StringAttr name) {
  auto names = argNames();
  SmallVector<Attribute> newNames(names.begin(), names.end());
  if (newNames[i] == name)
    return;
  newNames[i] = name;
  setArgumentNames(ArrayAttr::get(getContext(), names));
}

/// Change the name of the specified output port.
void InstanceOp::setResultName(size_t i, StringAttr name) {
  auto names = resultNames();
  SmallVector<Attribute> newNames(names.begin(), names.end());
  if (newNames[i] == name)
    return;
  newNames[i] = name;
  setResultNames(ArrayAttr::get(getContext(), names));
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Provide default names for instance results.
  std::string name = instanceName().str() + ".";
  size_t baseNameLen = name.size();

  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    auto resName = getResultName(i);
    name.resize(baseNameLen);
    if (resName && !resName.getValue().empty())
      name += resName.getValue().str();
    else
      name += std::to_string(i);
    setNameFn(getResult(i), name);
  }
}

//===----------------------------------------------------------------------===//
// HWOutputOp
//===----------------------------------------------------------------------===//

/// Verify that the num of operands and types fit the declared results.
LogicalResult OutputOp::verify() {
  // Check that the we (hw.output) have the same number of operands as our
  // region has results.
  auto *opParent = (*this)->getParentOp();
  FunctionType modType = getModuleType(opParent);
  ArrayRef<Type> modResults = modType.getResults();
  OperandRange outputValues = getOperands();
  if (modResults.size() != outputValues.size()) {
    emitOpError("must have same number of operands as region results.");
    return failure();
  }

  // Check that the types of our operands and the region's results match.
  for (size_t i = 0, e = modResults.size(); i < e; ++i) {
    if (modResults[i] != outputValues[i].getType()) {
      emitOpError("output types must match module. In "
                  "operand ")
          << i << ", expected " << modResults[i] << ", but got "
          << outputValues[i].getType() << ".";
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

LogicalResult
GlobalRefOp::verifySymbolUses(mlir::SymbolTableCollection &symTables) {
  Operation *parent = (*this)->getParentOp();
  SymbolTable &symTable = symTables.getSymbolTable(parent);
  StringAttr symNameAttr = (*this).sym_nameAttr();
  auto hasGlobalRef = [&](Attribute attr) -> bool {
    if (!attr)
      return false;
    for (auto ref : attr.cast<ArrayAttr>().getAsRange<GlobalRefAttr>())
      if (ref.getGlblSym().getAttr() == symNameAttr)
        return true;
    return false;
  };
  // For all inner refs in the namepath, ensure they have a corresponding
  // GlobalRefAttr to this GlobalRefOp.
  for (auto innerRef : namepath().getAsRange<hw::InnerRefAttr>()) {
    StringAttr modName = innerRef.getModule();
    StringAttr innerSym = innerRef.getName();
    Operation *mod = symTable.lookup(modName);
    if (!mod) {
      (*this)->emitOpError("module:'" + modName.str() + "' not found");
      return failure();
    }
    bool glblSymNotFound = true;
    bool innerSymOpNotFound = true;
    mod->walk([&](Operation *op) -> WalkResult {
      StringAttr attr = op->getAttrOfType<StringAttr>("inner_sym");
      // If this is one of the ops in the instance path for the GlobalRefOp.
      if (attr && attr == innerSym) {
        innerSymOpNotFound = false;
        // Each op can have an array of GlobalRefAttr, check if this op is one
        // of them.
        if (hasGlobalRef(op->getAttr(GlobalRefAttr::DialectAttrName))) {
          glblSymNotFound = false;
          return WalkResult::interrupt();
        }
        // If cannot find the ref, then its an error.
        return failure();
      }
      return WalkResult::advance();
    });
    if (glblSymNotFound) {
      // TODO: Doesn't yet work for symbls on FIRRTL module ports. Need to
      // implement an interface.
      if (isa<HWModuleOp, HWModuleExternOp>(mod)) {
        if (auto argAttrs = mod->getAttr(
                mlir::function_interface_impl::getArgDictAttrName()))
          for (auto attr :
               argAttrs.cast<ArrayAttr>().getAsRange<DictionaryAttr>())
            if (auto symRef = attr.get("hw.exportPort"))
              if (symRef.cast<FlatSymbolRefAttr>().getValue() == innerSym)
                if (hasGlobalRef(attr.get(GlobalRefAttr::DialectAttrName)))
                  return success();

        if (auto resAttrs = mod->getAttr(
                mlir::function_interface_impl::getResultDictAttrName()))
          for (auto attr :
               resAttrs.cast<ArrayAttr>().getAsRange<DictionaryAttr>())
            if (auto symRef = attr.get("hw.exportPort"))
              if (symRef.cast<FlatSymbolRefAttr>().getValue() == innerSym)
                if (hasGlobalRef(attr.get(GlobalRefAttr::DialectAttrName)))
                  return success();
      }
    }
    if (innerSymOpNotFound)
      return (*this)->emitOpError("operation:'" + innerSym.str() +
                                  "' in module:'" + modName.str() +
                                  "' could not be found");
    if (glblSymNotFound)
      return (*this)->emitOpError(
          "operation:'" + innerSym.str() + "' in module:'" + modName.str() +
          "' does not contain a reference to '" + symNameAttr.str() + "'");
  }
  return success();
}

static ParseResult parseSliceTypes(OpAsmParser &p, Type &srcType,
                                   Type &idxType) {
  Type type;
  if (p.parseType(type))
    return p.emitError(p.getCurrentLocation(), "Expected type");
  auto arrType = type_dyn_cast<ArrayType>(type);
  if (!arrType)
    return p.emitError(p.getCurrentLocation(), "Expected !hw.array type");
  srcType = type;
  unsigned idxWidth = llvm::Log2_64_Ceil(arrType.getSize());
  idxType = IntegerType::get(p.getBuilder().getContext(), idxWidth);
  return success();
}

static void printSliceTypes(OpAsmPrinter &p, Operation *, Type srcType,
                            Type idxType) {
  p.printType(srcType);
}

ParseResult ArrayCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 16> operands;
  Type elemType;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(elemType))
    return failure();

  if (operands.size() == 0)
    return parser.emitError(inputOperandsLoc,
                            "Cannot construct an array of length 0");
  result.addTypes({ArrayType::get(elemType, operands.size())});

  for (auto operand : operands)
    if (parser.resolveOperand(operand, elemType, result.operands))
      return failure();
  return success();
}

void ArrayCreateOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperands(inputs());
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : " << inputs()[0].getType();
}

void ArrayCreateOp::build(OpBuilder &b, OperationState &state,
                          ValueRange values) {
  assert(values.size() > 0 && "Cannot build array of zero elements");
  Type elemType = values[0].getType();
  assert(llvm::all_of(
             values,
             [elemType](Value v) -> bool { return v.getType() == elemType; }) &&
         "All values must have same type.");
  build(b, state, ArrayType::get(elemType, values.size()), values);
}

LogicalResult ArrayCreateOp::verify() {
  unsigned returnSize = getType().cast<ArrayType>().getSize();
  if (inputs().size() != returnSize)
    return failure();
  return success();
}

LogicalResult ArraySliceOp::verify() {
  unsigned inputSize = type_cast<ArrayType>(input().getType()).getSize();
  if (llvm::Log2_64_Ceil(inputSize) !=
      lowIndex().getType().getIntOrFloatBitWidth())
    return emitOpError(
        "ArraySlice: index width must match clog2 of array size");
  return success();
}

static ParseResult parseArrayConcatTypes(OpAsmParser &p,
                                         SmallVectorImpl<Type> &inputTypes,
                                         Type &resultType) {
  Type elemType;
  uint64_t resultSize = 0;

  auto parseElement = [&]() -> ParseResult {
    Type ty;
    if (p.parseType(ty))
      return failure();
    auto arrTy = type_dyn_cast<ArrayType>(ty);
    if (!arrTy)
      return p.emitError(p.getCurrentLocation(), "Expected !hw.array type");
    if (elemType && elemType != arrTy.getElementType())
      return p.emitError(p.getCurrentLocation(), "Expected array element type ")
             << elemType;

    elemType = arrTy.getElementType();
    inputTypes.push_back(ty);
    resultSize += arrTy.getSize();
    return success();
  };

  if (p.parseCommaSeparatedList(parseElement))
    return failure();

  resultType = ArrayType::get(elemType, resultSize);
  return success();
}

static void printArrayConcatTypes(OpAsmPrinter &p, Operation *,
                                  TypeRange inputTypes, Type resultType) {
  llvm::interleaveComma(inputTypes, p, [&p](Type t) { p << t; });
}

void ArrayConcatOp::build(OpBuilder &b, OperationState &state,
                          ValueRange values) {
  assert(!values.empty() && "Cannot build array of zero elements");
  ArrayType arrayTy = values[0].getType().cast<ArrayType>();
  Type elemTy = arrayTy.getElementType();
  assert(llvm::all_of(values,
                      [elemTy](Value v) -> bool {
                        return v.getType().isa<ArrayType>() &&
                               v.getType().cast<ArrayType>().getElementType() ==
                                   elemTy;
                      }) &&
         "All values must be of ArrayType with the same element type.");

  uint64_t resultSize = 0;
  for (Value val : values)
    resultSize += val.getType().cast<ArrayType>().getSize();
  build(b, state, ArrayType::get(elemTy, resultSize), values);
}

//===----------------------------------------------------------------------===//
// EnumConstantOp
//===----------------------------------------------------------------------===//

ParseResult EnumConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  EnumValueAttr value;
  if (parser.parseAttribute(value))
    return failure();

  result.addAttribute("enumerator", value);
  result.addTypes(value.getType().getValue());

  return success();
}

void EnumConstantOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printAttribute(enumerator());
}

void EnumConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), enumerator().getValue().str());
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

ParseResult StructCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  Type declOrAliasType;

  if (parser.parseLParen() || parser.parseOperandList(operands) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declOrAliasType))
    return failure();

  auto declType = type_dyn_cast<StructType>(declOrAliasType);
  if (!declType)
    return parser.emitError(parser.getNameLoc(),
                            "expected !hw.struct type or alias");

  llvm::SmallVector<Type, 4> structInnerTypes;
  declType.getInnerTypes(structInnerTypes);
  result.addTypes(declOrAliasType);

  if (parser.resolveOperands(operands, structInnerTypes, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void StructCreateOp::print(OpAsmPrinter &printer) {
  printer << " (";
  printer.printOperands(input());
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getType();
}

//===----------------------------------------------------------------------===//
// StructExplodeOp
//===----------------------------------------------------------------------===//

ParseResult StructExplodeOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  Type declType;

  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();
  auto structType = type_dyn_cast<StructType>(declType);
  if (!structType)
    return parser.emitError(parser.getNameLoc(),
                            "invalid kind of type specified");

  llvm::SmallVector<Type, 4> structInnerTypes;
  structType.getInnerTypes(structInnerTypes);
  result.addTypes(structInnerTypes);

  if (parser.resolveOperand(operand, declType, result.operands))
    return failure();
  return success();
}

void StructExplodeOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOperand(input());
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << input().getType();
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

/// Use the same parser for both struct_extract and union_extract since the
/// syntax is identical.
template <typename AggregateType>
static ParseResult parseExtractOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand operand;
  StringAttr fieldName;
  Type declType;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(fieldName, "field", result.attributes) ||
      parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();
  auto aggType = type_dyn_cast<AggregateType>(declType);
  if (!aggType)
    return parser.emitError(parser.getNameLoc(),
                            "invalid kind of type specified");

  Type resultType = aggType.getFieldType(fieldName.getValue());
  if (!resultType) {
    parser.emitError(parser.getNameLoc(), "invalid field name specified");
    return failure();
  }
  result.addTypes(resultType);

  if (parser.resolveOperand(operand, declType, result.operands))
    return failure();
  return success();
}

/// Use the same printer for both struct_extract and union_extract since the
/// syntax is identical.
template <typename AggType>
static void printExtractOp(OpAsmPrinter &printer, AggType op) {
  printer << " ";
  printer.printOperand(op.input());
  printer << "[\"" << op.field() << "\"]";
  printer.printOptionalAttrDict(op->getAttrs(), {"field"});
  printer << " : " << op.input().getType();
}

ParseResult StructExtractOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  return parseExtractOp<StructType>(parser, result);
}

void StructExtractOp::print(OpAsmPrinter &printer) {
  printExtractOp(printer, *this);
}

void StructExtractOp::build(OpBuilder &builder, OperationState &odsState,
                            Value input, StructType::FieldInfo field) {
  build(builder, odsState, field.type, input, field.name);
}

void StructExtractOp::build(OpBuilder &builder, OperationState &odsState,
                            Value input, StringAttr fieldAttr) {
  auto structType = type_cast<StructType>(input.getType());
  auto resultType = structType.getFieldType(fieldAttr);
  build(builder, odsState, resultType, input, fieldAttr);
}

// A struct extract of a struct create -> corresponding struct create operand.
OpFoldResult StructExtractOp::fold(ArrayRef<Attribute> operands) {
  auto structCreate = dyn_cast_or_null<StructCreateOp>(input().getDefiningOp());
  if (!structCreate)
    return {};
  auto ty = type_cast<StructType>(input().getType());
  if (!ty)
    return {};
  if (auto idx = ty.getFieldIndex(field()))
    return structCreate.getOperand(*idx);
  return {};
}

//===----------------------------------------------------------------------===//
// StructInjectOp
//===----------------------------------------------------------------------===//

ParseResult StructInjectOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  OpAsmParser::UnresolvedOperand operand, val;
  StringAttr fieldName;
  Type declType;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(fieldName, "field", result.attributes) ||
      parser.parseRSquare() || parser.parseComma() ||
      parser.parseOperand(val) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();
  auto structType = type_dyn_cast<StructType>(declType);
  if (!structType)
    return parser.emitError(inputOperandsLoc, "invalid kind of type specified");

  Type resultType = structType.getFieldType(fieldName.getValue());
  if (!resultType) {
    parser.emitError(inputOperandsLoc, "invalid field name specified");
    return failure();
  }
  result.addTypes(declType);

  if (parser.resolveOperands({operand, val}, {declType, resultType},
                             inputOperandsLoc, result.operands))
    return failure();
  return success();
}

void StructInjectOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOperand(input());
  printer << "[\"" << field() << "\"], ";
  printer.printOperand(newValue());
  printer.printOptionalAttrDict((*this)->getAttrs(), {"field"});
  printer << " : " << input().getType();
}

//===----------------------------------------------------------------------===//
// UnionCreateOp
//===----------------------------------------------------------------------===//

ParseResult UnionCreateOp::parse(OpAsmParser &parser, OperationState &result) {
  Type declOrAliasType;
  StringAttr field;
  OpAsmParser::UnresolvedOperand input;
  llvm::SMLoc fieldLoc = parser.getCurrentLocation();

  if (parser.parseAttribute(field, "field", result.attributes) ||
      parser.parseComma() || parser.parseOperand(input) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declOrAliasType))
    return failure();

  auto declType = type_dyn_cast<UnionType>(declOrAliasType);
  if (!declType)
    return parser.emitError(parser.getNameLoc(),
                            "expected !hw.union type or alias");

  Type inputType = declType.getFieldType(field.getValue());
  if (!inputType) {
    parser.emitError(fieldLoc, "cannot find union field '")
        << field.getValue() << '\'';
    return failure();
  }

  if (parser.resolveOperand(input, inputType, result.operands))
    return failure();
  result.addTypes({declOrAliasType});
  return success();
}

void UnionCreateOp::print(OpAsmPrinter &printer) {
  printer << " \"" << field() << "\", ";
  printer.printOperand(input());
  printer.printOptionalAttrDict((*this)->getAttrs(), {"field"});
  printer << " : " << getType();
}

//===----------------------------------------------------------------------===//
// UnionExtractOp
//===----------------------------------------------------------------------===//

ParseResult UnionExtractOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseExtractOp<UnionType>(parser, result);
}

void UnionExtractOp::print(OpAsmPrinter &printer) {
  printExtractOp(printer, *this);
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

void ArrayGetOp::build(OpBuilder &builder, OperationState &result, Value input,
                       Value index) {
  auto resultType = type_cast<ArrayType>(input.getType()).getElementType();
  build(builder, result, resultType, input, index);
}

// An array_get of an array_create with a constant index can just be the
// array_create operand at the constant index.
OpFoldResult ArrayGetOp::fold(ArrayRef<Attribute> operands) {
  auto inputCreate = dyn_cast_or_null<ArrayCreateOp>(input().getDefiningOp());
  if (!inputCreate)
    return {};

  IntegerAttr constIdx = operands[1].dyn_cast_or_null<IntegerAttr>();
  if (!constIdx || constIdx.getValue().getBitWidth() > 64)
    return {};

  uint64_t idx = constIdx.getValue().getLimitedValue();
  auto createInputs = inputCreate.inputs();
  if (idx >= createInputs.size())
    return {};
  return createInputs[createInputs.size() - idx - 1];
}

//===----------------------------------------------------------------------===//
// TypedeclOp
//===----------------------------------------------------------------------===//

StringRef TypedeclOp::getPreferredName() {
  return verilogName().getValueOr(getName());
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BitcastOp::fold(ArrayRef<Attribute> operands) {
  // Identity.
  // bitcast(%a) : A -> A ==> %a
  if (getOperand().getType() == getType())
    return getOperand();

  return {};
}

LogicalResult BitcastOp::canonicalize(BitcastOp op, PatternRewriter &rewriter) {
  // Composition.
  // %b = bitcast(%a) : A -> B
  //      bitcast(%b) : B -> C
  // ===> bitcast(%a) : A -> C
  auto inputBitcast = dyn_cast_or_null<BitcastOp>(op.input().getDefiningOp());
  if (!inputBitcast)
    return failure();
  auto bitcast = rewriter.createOrFold<BitcastOp>(op.getLoc(), op.getType(),
                                                  inputBitcast.input());
  rewriter.replaceOp(op, bitcast);
  return success();
}

LogicalResult BitcastOp::verify() {
  if (getBitWidth(input().getType()) != getBitWidth(result().getType()))
    return this->emitOpError("Bitwidth of input must match result");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/HW/HW.cpp.inc"
