//===- SystemCOps.cpp - Implement the SystemC operations ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemC ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SystemC/SystemCOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::systemc;

//===----------------------------------------------------------------------===//
// ImplicitSSAName Custom Directive
//===----------------------------------------------------------------------===//

static ParseResult parseImplicitSSAName(OpAsmParser &parser,
                                        StringAttr &nameAttr) {
  nameAttr = parser.getBuilder().getStringAttr(parser.getResultName(0).first);
  return success();
}

static void printImplicitSSAName(OpAsmPrinter &p, Operation *op,
                                 StringAttr nameAttr) {}

//===----------------------------------------------------------------------===//
// SCModuleOp
//===----------------------------------------------------------------------===//

static hw::PortDirection getDirection(Type type) {
  return TypeSwitch<Type, hw::PortDirection>(type)
      .Case<InOutType>([](auto ty) { return hw::PortDirection::INOUT; })
      .Case<InputType>([](auto ty) { return hw::PortDirection::INPUT; })
      .Case<OutputType>([](auto ty) { return hw::PortDirection::OUTPUT; });
}

SCModuleOp::PortDirectionRange
SCModuleOp::getPortsOfDirection(hw::PortDirection direction) {
  std::function<bool(const BlockArgument &)> predicateFn =
      [&](const BlockArgument &arg) -> bool {
    return getDirection(arg.getType()) == direction;
  };
  return llvm::make_filter_range(getArguments(), predicateFn);
}

void SCModuleOp::getPortInfoList(SmallVectorImpl<hw::PortInfo> &portInfoList) {
  for (int i = 0, e = getNumArguments(); i < e; ++i) {
    hw::PortInfo info;
    info.name = getPortNames()[i].cast<StringAttr>();
    info.type = getSignalBaseType(getArgument(i).getType());
    info.direction = getDirection(info.type);
    portInfoList.push_back(info);
  }
}

mlir::Region *SCModuleOp::getCallableRegion() { return &getBody(); }

ArrayRef<mlir::Type> SCModuleOp::getCallableResults() {
  return getResultTypes();
}

StringRef SCModuleOp::getModuleName() {
  return (*this)
      ->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
      .getValue();
}

ParseResult SCModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr moduleName;
  SmallVector<OpAsmParser::Argument, 4> args;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute> argNames;
  SmallVector<DictionaryAttr> resultAttrs;

  // Parse the visibility attribute.
  (void)mlir::impl::parseOptionalVisibilityKeyword(parser, result.attributes);

  if (parser.parseSymbolName(moduleName, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  bool isVariadic = false;
  if (hw::module_like_impl::parseModuleFunctionSignature(
          parser, args, isVariadic, resultTypes, resultAttrs, argNames))
    return failure();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  for (auto &arg : args) {
    argNames.push_back(
        StringAttr::get(parser.getContext(), arg.ssaName.name.drop_front()));
    argTypes.push_back(arg.type);
  }

  result.addAttribute("portNames",
                      ArrayAttr::get(parser.getContext(), argNames));

  auto type = parser.getBuilder().getFunctionType(argTypes, resultTypes);
  result.addAttribute(SCModuleOp::getTypeAttrName(), TypeAttr::get(type));

  mlir::function_interface_impl::addArgAndResultAttrs(
      parser.getBuilder(), result, args, resultAttrs);

  auto &body = *result.addRegion();
  if (parser.parseRegion(body, args))
    return failure();
  if (body.empty())
    body.push_back(std::make_unique<Block>().release());

  return success();
}

void SCModuleOp::print(OpAsmPrinter &p) {
  p << ' ';

  // Print the visibility of the module.
  StringRef visibilityAttrName = SymbolTable::getVisibilityAttrName();
  if (auto visibility =
          getOperation()->getAttrOfType<StringAttr>(visibilityAttrName))
    p << visibility.getValue() << ' ';

  p.printSymbolName(SymbolTable::getSymbolName(*this).getValue());
  p << ' ';

  bool needArgNamesAttr = false;
  hw::module_like_impl::printModuleSignature(
      p, *this, getFunctionType().getInputs(), false, {}, needArgNamesAttr);
  mlir::function_interface_impl::printFunctionAttributes(
      p, *this, getFunctionType().getInputs().size(), 0, {"portNames"});

  p << ' ';
  p.printRegion(getBody(), false, false);
}

static Type wrapPortType(Type type, hw::PortDirection direction) {
  if (auto inoutTy = type.dyn_cast<hw::InOutType>())
    type = inoutTy.getElementType();

  switch (direction) {
  case hw::PortDirection::INOUT:
    return InOutType::get(type);
  case hw::PortDirection::INPUT:
    return InputType::get(type);
  case hw::PortDirection::OUTPUT:
    return OutputType::get(type);
  }
}

void SCModuleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       StringAttr name, ArrayAttr portNames,
                       ArrayRef<Type> portTypes,
                       ArrayRef<NamedAttribute> attributes) {
  odsState.addAttribute(getPortNamesAttrName(odsState.name), portNames);
  Region *region = odsState.addRegion();

  auto moduleType = odsBuilder.getFunctionType(portTypes, {});
  odsState.addAttribute(getTypeAttrName(), TypeAttr::get(moduleType));

  odsState.addAttribute(SymbolTable::getSymbolAttrName(), name);
  region->push_back(new Block);
  region->addArguments(
      portTypes,
      SmallVector<Location>(portTypes.size(), odsBuilder.getUnknownLoc()));
  odsState.addAttributes(attributes);
}

void SCModuleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       StringAttr name, ArrayRef<hw::PortInfo> ports,
                       ArrayRef<NamedAttribute> attributes) {
  MLIRContext *ctxt = odsBuilder.getContext();
  SmallVector<Attribute> portNames;
  SmallVector<Type> portTypes;
  for (auto port : ports) {
    portNames.push_back(StringAttr::get(ctxt, port.getName()));
    portTypes.push_back(wrapPortType(port.type, port.direction));
  }
  build(odsBuilder, odsState, name, ArrayAttr::get(ctxt, portNames), portTypes);
}

void SCModuleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       StringAttr name, const hw::ModulePortInfo &ports,
                       ArrayRef<NamedAttribute> attributes) {
  SmallVector<hw::PortInfo> portInfos(ports.inputs);
  portInfos.append(ports.outputs);
  build(odsBuilder, odsState, name, portInfos, attributes);
}

void SCModuleOp::getAsmBlockArgumentNames(mlir::Region &region,
                                          mlir::OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;

  ArrayAttr portNames = getPortNames();
  for (size_t i = 0, e = getNumArguments(); i != e; ++i) {
    auto str = portNames[i].cast<StringAttr>().getValue();
    setNameFn(getArgument(i), str);
  }
}

LogicalResult SCModuleOp::verify() {
  if (getFunctionType().getNumResults() != 0)
    return emitOpError(
        "incorrect number of function results (always has to be 0)");
  if (getPortNames().size() != getFunctionType().getNumInputs())
    return emitOpError("incorrect number of port names");

  for (auto arg : getArguments()) {
    if (!hw::type_isa<InputType, OutputType, InOutType>(arg.getType()))
      return mlir::emitError(
          arg.getLoc(),
          "module port must be of type 'sc_in', 'sc_out', or 'sc_inout'");
  }

  for (auto portName : getPortNames()) {
    if (portName.cast<StringAttr>().getValue().empty())
      return emitOpError("port name must not be empty");
  }

  return success();
}

LogicalResult SCModuleOp::verifyRegions() {
  DenseMap<StringRef, BlockArgument> portNames;
  DenseMap<StringRef, Operation *> memberNames;
  DenseMap<StringRef, Operation *> localNames;

  bool portsVerified = true;

  for (auto arg : llvm::zip(getPortNames(), getArguments())) {
    StringRef argName = std::get<0>(arg).cast<StringAttr>().getValue();
    BlockArgument argValue = std::get<1>(arg);

    if (portNames.count(argName)) {
      auto diag = mlir::emitError(argValue.getLoc(), "redefines port name '")
                  << argName << "'";
      diag.attachNote(portNames[argName].getLoc())
          << "'" << argName << "' first defined here";
      diag.attachNote(getLoc()) << "in module '@" << getModuleName() << "'";
      portsVerified = false;
      continue;
    }

    portNames.insert({argName, argValue});
  }

  WalkResult result = walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<SCModuleOp>(op->getParentOp()))
      localNames.clear();

    if (auto nameDeclOp = dyn_cast<SystemCNameDeclOpInterface>(op)) {
      StringRef name = nameDeclOp.getName();

      auto reportNameRedefinition = [&](Location firstLoc) -> WalkResult {
        auto diag = mlir::emitError(op->getLoc(), "redefines name '")
                    << name << "'";
        diag.attachNote(firstLoc) << "'" << name << "' first defined here";
        diag.attachNote(getLoc()) << "in module '@" << getModuleName() << "'";
        return WalkResult::interrupt();
      };

      if (portNames.count(name))
        return reportNameRedefinition(portNames[name].getLoc());
      if (memberNames.count(name))
        return reportNameRedefinition(memberNames[name]->getLoc());
      if (localNames.count(name))
        return reportNameRedefinition(localNames[name]->getLoc());

      if (isa<SCModuleOp>(op->getParentOp()))
        memberNames.insert({name, op});
      else
        localNames.insert({name, op});
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted() || !portsVerified)
    return failure();

  return success();
}

CtorOp SCModuleOp::getOrCreateCtor() {
  CtorOp ctor;
  getBody().walk([&](Operation *op) {
    if ((ctor = dyn_cast<CtorOp>(op)))
      return WalkResult::interrupt();

    return WalkResult::skip();
  });

  if (ctor)
    return ctor;

  return OpBuilder(getBody()).create<CtorOp>(getLoc());
}

DestructorOp SCModuleOp::getOrCreateDestructor() {
  DestructorOp destructor;
  getBody().walk([&](Operation *op) {
    if ((destructor = dyn_cast<DestructorOp>(op)))
      return WalkResult::interrupt();

    return WalkResult::skip();
  });

  if (destructor)
    return destructor;

  return OpBuilder::atBlockEnd(getBodyBlock()).create<DestructorOp>(getLoc());
}

//===----------------------------------------------------------------------===//
// SignalOp
//===----------------------------------------------------------------------===//

void SignalOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getSignal(), getName());
}

//===----------------------------------------------------------------------===//
// CtorOp
//===----------------------------------------------------------------------===//

LogicalResult CtorOp::verify() {
  if (getBody().getNumArguments() != 0)
    return emitOpError("must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// SCFuncOp
//===----------------------------------------------------------------------===//

void SCFuncOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getHandle(), getName());
}

LogicalResult SCFuncOp::verify() {
  if (getBody().getNumArguments() != 0)
    return emitOpError("must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceDeclOp
//===----------------------------------------------------------------------===//

void InstanceDeclOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getInstanceHandle(), getName());
}

Operation *InstanceDeclOp::getReferencedModule(const hw::HWSymbolCache *cache) {
  if (cache)
    if (auto *result = cache->getDefinition(getModuleNameAttr()))
      return result;

  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  return topLevelModuleOp.lookupSymbol(getModuleName());
}

Operation *InstanceDeclOp::getReferencedModule() {
  return getReferencedModule(/*cache=*/nullptr);
}

LogicalResult
InstanceDeclOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module =
      symbolTable.lookupNearestSymbolFrom(*this, getModuleNameAttr());
  if (module == nullptr)
    return emitError("cannot find module definition '")
           << getModuleName() << "'";

  auto emitError = [&](const std::function<void(InFlightDiagnostic & diag)> &fn)
      -> LogicalResult {
    auto diag = emitOpError();
    fn(diag);
    diag.attachNote(module->getLoc()) << "module declared here";
    return failure();
  };

  // It must be a systemc module.
  if (!isa<SCModuleOp>(module))
    return emitError([&](auto &diag) {
      diag << "symbol reference '" << getModuleName()
           << "' isn't a systemc module";
    });

  auto scModule = cast<SCModuleOp>(module);

  // Check that the module name of the symbol and instance type match.
  if (scModule.getModuleName() != getInstanceType().getModuleName())
    return emitError([&](auto &diag) {
      diag << "module names must match; expected '" << scModule.getModuleName()
           << "' but got '" << getInstanceType().getModuleName().getValue()
           << "'";
    });

  // Check that port types and names are consistent with the referenced module.
  ArrayRef<ModuleType::PortInfo> ports = getInstanceType().getPorts();
  ArrayAttr modArgNames = scModule.getPortNames();
  auto numPorts = ports.size();
  auto expectedPortTypes = scModule.getArgumentTypes();

  if (expectedPortTypes.size() != numPorts)
    return emitError([&](auto &diag) {
      diag << "has a wrong number of ports; expected "
           << expectedPortTypes.size() << " but got " << numPorts;
    });

  for (size_t i = 0; i != numPorts; ++i) {
    if (ports[i].type != expectedPortTypes[i]) {
      return emitError([&](auto &diag) {
        diag << "port type #" << i << " must be " << expectedPortTypes[i]
             << ", but got " << ports[i].type;
      });
    }

    if (ports[i].name != modArgNames[i])
      return emitError([&](auto &diag) {
        diag << "port name #" << i << " must be " << modArgNames[i]
             << ", but got " << ports[i].name;
      });
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DestructorOp
//===----------------------------------------------------------------------===//

LogicalResult DestructorOp::verify() {
  if (getBody().getNumArguments() != 0)
    return emitOpError("must not have any arguments");

  return success();
}

//===----------------------------------------------------------------------===//
// Arithmetic operations
//===----------------------------------------------------------------------===//

LogicalResult
AddOp::inferReturnTypes(MLIRContext *context, Optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        mlir::RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  Type lhsTy = operands[0].getType();
  Type rhsTy = operands[1].getType();

  if (lhsTy.isa<SignedType>() || rhsTy.isa<SignedType>()) {
    inferredReturnTypes.push_back(SignedType::get(context));
    return success();
  }

  inferredReturnTypes.push_back(UnsignedType::get(context));

  return success();
}

LogicalResult
SubOp::inferReturnTypes(MLIRContext *context, Optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        mlir::RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(SignedType::get(context));
  return success();
}

//===----------------------------------------------------------------------===//
// Bitwise operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/SystemC/SystemC.cpp.inc"
