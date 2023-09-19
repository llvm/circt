//===- MSFTOps.cpp - Implement MSFT dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MSFT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Misc helper functions
//===----------------------------------------------------------------------===//

static bool hasAttribute(StringRef name, ArrayRef<NamedAttribute> attrs) {
  for (const auto &argAttr : attrs)
    if (argAttr.getName() == name)
      return true;
  return false;
}

// Copied nearly exactly from hwops.cpp.
// TODO: Unify code once a `ModuleLike` op interface exists.
static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, const hw::ModulePortInfo &ports) {
  using namespace mlir::function_interface_impl;
  LocationAttr unknownLoc = builder.getUnknownLoc();

  // Add an attribute for the name.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);

  SmallVector<Attribute> argNames, resultNames;
  SmallVector<Type, 4> argTypes, resultTypes;
  SmallVector<Attribute> argAttrs, resultAttrs;
  SmallVector<Attribute> argLocs, resultLocs;
  auto exportPortIdent = StringAttr::get(builder.getContext(), "hw.exportPort");

  for (auto elt : ports.getInputs()) {
    if (elt.dir == hw::ModulePort::Direction::InOut &&
        !elt.type.isa<hw::InOutType>())
      elt.type = hw::InOutType::get(elt.type);
    argTypes.push_back(elt.type);
    argNames.push_back(elt.name);
    argLocs.push_back(elt.loc ? elt.loc : unknownLoc);
    Attribute attr;
    if (elt.sym && !elt.sym.empty())
      attr = builder.getDictionaryAttr({{exportPortIdent, elt.sym}});
    else
      attr = builder.getDictionaryAttr({});
    argAttrs.push_back(attr);
  }

  for (auto elt : ports.getOutputs()) {
    resultTypes.push_back(elt.type);
    resultNames.push_back(elt.name);
    resultLocs.push_back(elt.loc ? elt.loc : unknownLoc);
    Attribute attr;
    if (elt.sym && !elt.sym.empty())
      attr = builder.getDictionaryAttr({{exportPortIdent, elt.sym}});
    else
      attr = builder.getDictionaryAttr({});
    resultAttrs.push_back(attr);
  }

  // Record the argument and result types as an attribute.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(MSFTModuleOp::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute("argNames", builder.getArrayAttr(argNames));
  result.addAttribute("resultNames", builder.getArrayAttr(resultNames));
  result.addAttribute("argLocs", builder.getArrayAttr(argLocs));
  result.addAttribute("resultLocs", builder.getArrayAttr(resultLocs));
  result.addAttribute("parameters", builder.getDictionaryAttr({}));
  result.addAttribute(MSFTModuleOp::getArgAttrsAttrName(result.name),
                      builder.getArrayAttr(argAttrs));
  result.addAttribute(MSFTModuleOp::getResAttrsAttrName(result.name),
                      builder.getArrayAttr(resultAttrs));
  result.addRegion();
}

//===----------------------------------------------------------------------===//
// Custom directive parsers/printers
//===----------------------------------------------------------------------===//

static ParseResult parsePhysLoc(OpAsmParser &p, PhysLocationAttr &attr) {
  llvm::SMLoc loc = p.getCurrentLocation();
  StringRef devTypeStr;
  uint64_t x, y, num;

  if (p.parseKeyword(&devTypeStr) || p.parseKeyword("x") || p.parseColon() ||
      p.parseInteger(x) || p.parseKeyword("y") || p.parseColon() ||
      p.parseInteger(y) || p.parseKeyword("n") || p.parseColon() ||
      p.parseInteger(num))
    return failure();

  std::optional<PrimitiveType> devType = symbolizePrimitiveType(devTypeStr);
  if (!devType) {
    p.emitError(loc, "Unknown device type '" + devTypeStr + "'");
    return failure();
  }
  PrimitiveTypeAttr devTypeAttr =
      PrimitiveTypeAttr::get(p.getContext(), *devType);
  attr = PhysLocationAttr::get(p.getContext(), devTypeAttr, x, y, num);
  return success();
}

static void printPhysLoc(OpAsmPrinter &p, Operation *, PhysLocationAttr loc) {
  p << stringifyPrimitiveType(loc.getPrimitiveType().getValue())
    << " x: " << loc.getX() << " y: " << loc.getY() << " n: " << loc.getNum();
}

static ParseResult parseListOptionalRegLocList(OpAsmParser &p,
                                               LocationVectorAttr &locs) {
  SmallVector<PhysLocationAttr, 32> locArr;
  TypeAttr type;
  if (p.parseAttribute(type) || p.parseLSquare() ||
      p.parseCommaSeparatedList(
          [&]() { return parseOptionalRegLoc(locArr, p); }) ||
      p.parseRSquare())
    return failure();

  if (failed(LocationVectorAttr::verify(
          [&p]() { return p.emitError(p.getNameLoc()); }, type, locArr)))
    return failure();
  locs = LocationVectorAttr::get(p.getContext(), type, locArr);
  return success();
}

static void printListOptionalRegLocList(OpAsmPrinter &p, Operation *,
                                        LocationVectorAttr locs) {
  p << locs.getType() << " [";
  llvm::interleaveComma(locs.getLocs(), p, [&p](PhysLocationAttr loc) {
    printOptionalRegLoc(loc, p);
  });
  p << "]";
}

static ParseResult parseImplicitInnerRef(OpAsmParser &p,
                                         hw::InnerRefAttr &innerRef) {
  SymbolRefAttr sym;
  if (p.parseAttribute(sym))
    return failure();
  auto loc = p.getCurrentLocation();
  if (sym.getNestedReferences().size() != 1)
    return p.emitError(loc, "expected <module sym>::<inner name>");
  innerRef = hw::InnerRefAttr::get(
      sym.getRootReference(),
      sym.getNestedReferences().front().getRootReference());
  return success();
}
void printImplicitInnerRef(OpAsmPrinter &p, Operation *,
                           hw::InnerRefAttr innerRef) {
  MLIRContext *ctxt = innerRef.getContext();
  StringRef innerRefNameStr, moduleStr;
  if (innerRef.getName())
    innerRefNameStr = innerRef.getName().getValue();
  if (innerRef.getModule())
    moduleStr = innerRef.getModule().getValue();
  p << SymbolRefAttr::get(ctxt, moduleStr,
                          {FlatSymbolRefAttr::get(ctxt, innerRefNameStr)});
}

/// Parse an parameter list if present. Same format as HW dialect.
/// module-parameter-list ::= `<` parameter-decl (`,` parameter-decl)* `>`
/// parameter-decl ::= identifier `:` type
/// parameter-decl ::= identifier `:` type `=` attribute
///
static ParseResult parseParameterList(OpAsmParser &parser,
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
        parameters.push_back(hw::ParamDeclAttr::get(
            builder.getContext(), builder.getStringAttr(name), type, value));
        return success();
      });
}

/// Shim to also use this for the InstanceOp custom parser.
static ParseResult parseParameterList(OpAsmParser &parser,
                                      ArrayAttr &parameters) {
  SmallVector<Attribute> parseParameters;
  if (failed(parseParameterList(parser, parseParameters)))
    return failure();

  parameters = ArrayAttr::get(parser.getContext(), parseParameters);

  return success();
}

/// Print a parameter list for a module or instance. Same format as HW dialect.
static void printParameterList(OpAsmPrinter &p, Operation *op,
                               ArrayAttr parameters) {
  if (!parameters || parameters.empty())
    return;

  p << '<';
  llvm::interleaveComma(parameters, p, [&](Attribute param) {
    auto paramAttr = param.cast<hw::ParamDeclAttr>();
    p << paramAttr.getName().getValue() << ": " << paramAttr.getType();
    if (auto value = paramAttr.getValue()) {
      p << " = ";
      p.printAttributeWithoutType(value);
    }
  });
  p << '>';
}

static ParseResult parseModuleLikeOp(OpAsmParser &parser,
                                     OperationState &result,
                                     bool withParameters = false) {
  using namespace mlir::function_interface_impl;
  auto loc = parser.getCurrentLocation();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  if (withParameters) {
    // Parse the parameters
    DictionaryAttr paramsAttr;
    if (parser.parseAttribute(paramsAttr))
      return failure();
    result.addAttribute("parameters", paramsAttr);
  }

  // Parse the function signature.
  bool isVariadic = false;
  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  SmallVector<Attribute> argNames;
  SmallVector<Attribute> argLocs;
  SmallVector<Attribute> resultNames;
  SmallVector<DictionaryAttr, 4> resultAttrs;
  SmallVector<Attribute> resultLocs;
  TypeAttr functionType;
  if (hw::module_like_impl::parseModuleFunctionSignature(
          parser, isVariadic, entryArgs, argNames, argLocs, resultNames,
          resultAttrs, resultLocs, functionType))
    return failure();

  // If function attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  if (hasAttribute("argNames", result.attributes) ||
      hasAttribute("resultNames", result.attributes)) {
    parser.emitError(
        loc, "explicit argNames and resultNames attributes not allowed");
    return failure();
  }

  auto *context = result.getContext();
  result.addAttribute("argNames", ArrayAttr::get(context, argNames));
  result.addAttribute("argLocs", ArrayAttr::get(context, argLocs));
  result.addAttribute("resultNames", ArrayAttr::get(context, resultNames));
  result.addAttribute("resultLocs", ArrayAttr::get(context, resultLocs));
  result.addAttribute(MSFTModuleOp::getFunctionTypeAttrName(result.name),
                      functionType);

  // Add the attributes to the module arguments.
  addArgAndResultAttrs(parser.getBuilder(), result, entryArgs, resultAttrs,
                       MSFTModuleOp::getArgAttrsAttrName(result.name),
                       MSFTModuleOp::getResAttrsAttrName(result.name));

  // Parse the optional module body.
  auto regionSuccess =
      parser.parseOptionalRegion(*result.addRegion(), entryArgs);
  if (regionSuccess.has_value() && failed(*regionSuccess))
    return failure();

  return success();
}

template <typename ModuleTy>
static void printModuleLikeOp(hw::HWModuleLike moduleLike, OpAsmPrinter &p,
                              Attribute parameters = nullptr) {
  using namespace mlir::function_interface_impl;

  auto argTypes = moduleLike.getInputTypes();
  auto resultTypes = moduleLike.getOutputTypes();

  // Print the operation and the function name.
  p << ' ';
  p.printSymbolName(SymbolTable::getSymbolName(moduleLike).getValue());

  if (parameters) {
    // Print the parameterization.
    p << ' ';
    p.printAttribute(parameters);
  }

  p << ' ';
  bool needArgNamesAttr = false;
  hw::module_like_impl::printModuleSignature(p, moduleLike, argTypes,
                                             /*isVariadic=*/false, resultTypes,
                                             needArgNamesAttr);

  SmallVector<StringRef, 3> omittedAttrs;
  if (!needArgNamesAttr)
    omittedAttrs.push_back("argNames");
  omittedAttrs.push_back("argLocs");
  omittedAttrs.push_back("resultNames");
  omittedAttrs.push_back("resultLocs");
  omittedAttrs.push_back("parameters");
  omittedAttrs.push_back(
      ModuleTy::getFunctionTypeAttrName(moduleLike->getName()));
  omittedAttrs.push_back(ModuleTy::getArgAttrsAttrName(moduleLike->getName()));
  omittedAttrs.push_back(ModuleTy::getResAttrsAttrName(moduleLike->getName()));

  printFunctionAttributes(p, moduleLike, omittedAttrs);

  // Print the body if this is not an external function.
  Region &mbody = moduleLike.getModuleBody();
  if (!mbody.empty()) {
    p << ' ';
    p.printRegion(mbody, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

//===----------------------------------------------------------------------===//
// DynamicInstanceOp
//===----------------------------------------------------------------------===//

ArrayAttr DynamicInstanceOp::globalRefPath() {
  SmallVector<Attribute, 16> path;
  DynamicInstanceOp next = *this;
  do {
    path.push_back(next.getInstanceRefAttr());
    next = next->getParentOfType<DynamicInstanceOp>();
  } while (next);
  std::reverse(path.begin(), path.end());
  return ArrayAttr::get(getContext(), path);
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

size_t InstanceOp::getNumPorts() { return getNumOperands() + getNumResults(); }
size_t InstanceOp::getNumInputPorts() { return getNumOperands(); }
size_t InstanceOp::getNumOutputPorts() { return getNumResults(); }

size_t InstanceOp::getPortIdForInputId(size_t idx) { return idx; }

size_t InstanceOp::getPortIdForOutputId(size_t idx) {
  return idx + getNumInputPorts();
}

std::optional<size_t> InstanceOp::getTargetResultIndex() {
  // Inner symbols on instance operations target the op not any result.
  return std::nullopt;
}

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module =
      symbolTable.lookupNearestSymbolFrom(*this, getModuleNameAttr());
  if (module == nullptr)
    return emitError("Cannot find module definition '")
           << getModuleName() << "'";

  // It must be some sort of module.
  if (!isa<hw::HWModuleLike>(module))
    return emitError("symbol reference '")
           << getModuleName() << "' isn't a module";
  return success();
}

Operation *InstanceOp::getReferencedModuleSlow() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;
  return topLevelModuleOp.lookupSymbol(getModuleName());
}

hw::ModulePortInfo InstanceOp::getPortList() {
  return cast<hw::PortList>(getReferencedModuleSlow()).getPortList();
}

StringAttr InstanceOp::getResultName(size_t idx) {
  if (auto *refMod = getReferencedModuleSlow()) {
    if (auto mod = dyn_cast<hw::HWModuleLike>(refMod)) {
      return mod.getOutputNameAttr(idx);
    }
    llvm::report_fatal_error("Instance not of Module");
  }
  return StringAttr();
}

/// Instance name is the same as the symbol name. This may change in the
/// future.
mlir::StringAttr InstanceOp::getInstanceNameAttr() {
  return getInnerNameAttr();
}

llvm::StringRef InstanceOp::getInstanceName() { return *getInnerName(); }

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Provide default names for instance results.
  std::string name = getInstanceName().str() + ".";
  size_t baseNameLen = name.size();

  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    name.resize(baseNameLen);
    StringAttr resNameAttr = getResultName(i);
    if (resNameAttr)
      name += resNameAttr.getValue().str();
    else
      name += std::to_string(i);
    setNameFn(getResult(i), name);
  }
}

LogicalResult
InstanceOp::verifySignatureMatch(const hw::ModulePortInfo &ports) {
  if (ports.sizeInputs() != getNumOperands())
    return emitOpError("wrong number of inputs (expected ")
           << ports.sizeInputs() << ")";
  if (ports.sizeOutputs() != getNumResults())
    return emitOpError("wrong number of outputs (expected ")
           << ports.sizeOutputs() << ")";
  for (auto port : ports.getInputs())
    if (getOperand(port.argNum).getType() != port.type)
      return emitOpError("in input port ")
             << port.name << ", expected type " << port.type << " got "
             << getOperand(port.argNum).getType();
  for (auto port : ports.getOutputs())
    if (getResult(port.argNum).getType() != port.type)
      return emitOpError("in output port ")
             << port.name << ", expected type " << port.type << " got "
             << getResult(port.argNum).getType();

  return success();
}

void InstanceOp::build(OpBuilder &builder, OperationState &state,
                       ArrayRef<Type> resultTypes, StringAttr symName,
                       FlatSymbolRefAttr moduleName, ArrayRef<Value> inputs) {
  build(builder, state, resultTypes, hw::InnerSymAttr::get(symName), moduleName,
        inputs, ArrayAttr(), SymbolRefAttr());
}

//===----------------------------------------------------------------------===//
// MSFTModuleOp
//===----------------------------------------------------------------------===//

/// Return an encapsulated set of information about input and output ports of
/// the specified module or instance.  The input ports always come before the
/// output ports in the list.
/// TODO: This should really be shared with the HW dialect instead of cloned.
/// Consider adding a `HasModulePorts` op interface to facilitate.
hw::ModulePortInfo MSFTModuleOp::getPortList() {
  SmallVector<hw::PortInfo> inputs, outputs;
  auto argNames = this->getArgNames();
  auto argTypes = getArgumentTypes();
  auto argLocs = getArgLocs();

  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto argName = argNames[i].cast<StringAttr>();
    auto direction = isInOut ? hw::ModulePort::Direction::InOut
                             : hw::ModulePort::Direction::Input;
    auto type = argTypes[i];
    if (auto inout = type.dyn_cast<hw::InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }
    auto argLoc = argLocs[i].cast<LocationAttr>();
    inputs.push_back({{argName, type, direction}, i, {}, {}, argLoc});
  }

  auto resultNames = this->getResultNames();
  auto resultTypes = getResultTypes();
  auto resultLocs = getResultLocs();
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    outputs.push_back({{resultNames[i].cast<StringAttr>(), resultTypes[i],
                        hw::ModulePort::Direction::Output},
                       i,
                       {},
                       {},
                       resultLocs[i].cast<LocationAttr>()});
  }
  return hw::ModulePortInfo(inputs, outputs);
}

SmallVector<BlockArgument>
MSFTModuleOp::addPorts(ArrayRef<std::pair<StringAttr, Type>> inputs,
                       ArrayRef<std::pair<StringAttr, Value>> outputs) {
  auto *ctxt = getContext();
  Block *body = getBodyBlock();

  // Append new inputs.
  SmallVector<Type, 32> modifiedArgs(getArgumentTypes().begin(),
                                     getArgumentTypes().end());
  SmallVector<Attribute> modifiedArgNames(
      getArgNames().getAsRange<Attribute>());
  SmallVector<Attribute> modifiedArgLocs(getArgLocs().getAsRange<Attribute>());
  SmallVector<BlockArgument> newBlockArgs;
  Location unknownLoc = UnknownLoc::get(ctxt);
  for (auto ttPair : inputs) {
    modifiedArgNames.push_back(ttPair.first);
    modifiedArgs.push_back(ttPair.second);
    modifiedArgLocs.push_back(unknownLoc);
    newBlockArgs.push_back(
        body->addArgument(ttPair.second, Builder(ctxt).getUnknownLoc()));
  }
  setArgNamesAttr(ArrayAttr::get(ctxt, modifiedArgNames));
  setArgLocsAttr(ArrayAttr::get(ctxt, modifiedArgLocs));

  // Append new outputs.
  SmallVector<Type, 32> modifiedResults(getResultTypes().begin(),
                                        getResultTypes().end());
  SmallVector<Attribute> modifiedResultNames(
      getResultNames().getAsRange<Attribute>());
  SmallVector<Attribute> modifiedResultLocs(
      getResultLocs().getAsRange<Attribute>());
  Operation *terminator = body->getTerminator();
  SmallVector<Value, 32> modifiedOutputs(terminator->getOperands());
  for (auto tvPair : outputs) {
    modifiedResultNames.push_back(tvPair.first);
    modifiedResults.push_back(tvPair.second.getType());
    modifiedResultLocs.push_back(unknownLoc);
    modifiedOutputs.push_back(tvPair.second);
  }
  setResultNamesAttr(ArrayAttr::get(ctxt, modifiedResultNames));
  setResultLocsAttr(ArrayAttr::get(ctxt, modifiedResultLocs));
  terminator->setOperands(modifiedOutputs);

  // Finalize and return.
  setFunctionType(FunctionType::get(ctxt, modifiedArgs, modifiedResults));
  return newBlockArgs;
}

// Remove the ports at the specified indexes.
SmallVector<unsigned> MSFTModuleOp::removePorts(llvm::BitVector inputs,
                                                llvm::BitVector outputs) {
  Block *body = getBodyBlock();
  Operation *terminator = body->getTerminator();

  SmallVector<unsigned> removeInputs;
  unsigned originalNumArgs = getNumInputPorts();
  for (unsigned i = 0; i < originalNumArgs; ++i) {
    if (inputs.test(i))
      removeInputs.push_back(i);
  }

  SmallVector<unsigned> removeOutputs;
  unsigned originalNumOutputs = getNumOutputPorts();
  for (size_t i = 0; i < originalNumOutputs; ++i) {
    if (outputs.test(i))
      removeOutputs.push_back(i);
  }
  modifyPorts({}, {}, removeInputs, removeOutputs);

  // Build new operand list for output op. Construct an output mapping to
  // return as a side-effect.
  SmallVector<Value> newOutputValues;
  SmallVector<unsigned> newToOldResultMap;

  for (unsigned i = 0; i < originalNumOutputs; ++i) {
    if (!outputs.test(i)) {
      newOutputValues.push_back(terminator->getOperand(i));
      newToOldResultMap.push_back(i);
    }
  }
  terminator->setOperands(newOutputValues);

  // Erase the arguments after setting the new output op operands since the
  // arguments might be used by output op.
  for (unsigned argNum = 0, e = body->getArguments().size(); argNum < e;
       ++argNum)
    if (inputs.test(argNum))
      body->getArgument(argNum).dropAllUses();
  body->eraseArguments(inputs);

  return newToOldResultMap;
}

void MSFTModuleOp::modifyPorts(
    llvm::ArrayRef<std::pair<unsigned int, circt::hw::PortInfo>> insertInputs,
    llvm::ArrayRef<std::pair<unsigned int, circt::hw::PortInfo>> insertOutputs,
    llvm::ArrayRef<unsigned int> eraseInputs,
    llvm::ArrayRef<unsigned int> eraseOutputs) {
  hw::modifyModulePorts(*this, insertInputs, insertOutputs, eraseInputs,
                        eraseOutputs);
}

void MSFTModuleOp::appendOutputs(
    ArrayRef<std::pair<StringAttr, Value>> outputs) {
  addPorts({}, outputs);
}

void MSFTModuleOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name, hw::ModulePortInfo ports,
                         ArrayRef<NamedAttribute> params) {
  buildModule(builder, result, name, ports);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);
  auto unknownLoc = builder.getUnknownLoc();

  // Add arguments to the body block.
  for (auto port : ports.getInputs()) {
    auto type = port.type;
    if (port.isInOut() && !type.isa<hw::InOutType>())
      type = hw::InOutType::get(type);
    auto loc = port.loc ? Location(port.loc) : unknownLoc;
    body->addArgument(type, loc);
  }

  MSFTModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

ParseResult MSFTModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseModuleLikeOp(parser, result, /*withParameters=*/true);
}

void MSFTModuleOp::print(OpAsmPrinter &p) {
  printModuleLikeOp<MSFTModuleOp>(*this, p, getParametersAttr());
}

LogicalResult MSFTModuleOp::verify() {
  auto &body = getBody();
  if (body.empty())
    return success();

  // Verify the number of block arguments.
  auto type = getFunctionType();
  auto numInputs = type.getNumInputs();
  auto *bodyBlock = &body.front();
  if (bodyBlock->getNumArguments() != numInputs)
    return emitOpError("entry block must have")
           << numInputs << " arguments to match module signature";

  // Verify that the block arguments match the op's attributes.
  for (auto [arg, type, loc] :
       llvm::zip(body.getArguments(), type.getInputs(), getArgLocs())) {
    if (arg.getType() != type)
      return emitOpError("block argument types should match signature types");
    if (arg.getLoc() != loc.cast<LocationAttr>())
      return emitOpError(
          "block argument locations should match signature locations");
  }

  return success();
}

SmallVector<Location> MSFTModuleOp::getAllPortLocs() {
  auto empty = UnknownLoc::get(getContext());
  SmallVector<Location> retval;
  if (auto locs = getArgLocs()) {
    for (auto l : locs)
      retval.push_back(cast<Location>(l));
  }
  retval.resize(getNumInputPorts(), empty);

  if (auto locs = getResultLocs()) {
    for (auto l : locs)
      retval.push_back(cast<Location>(l));
  }
  retval.resize(getNumInputPorts() + getNumOutputPorts(), empty);
  return retval;
}

SmallVector<Attribute> MSFTModuleOp::getAllPortAttrs() {
  SmallVector<Attribute> retval;
  auto empty = DictionaryAttr::get(getContext());
  auto attrs = getArgAttrs();
  if (attrs)
    for (auto a : *attrs)
      retval.push_back(a);
  retval.resize(getNumInputPorts(), empty);
  attrs = getResAttrs();
  if (attrs)
    for (auto a : *attrs)
      retval.push_back(a);
  retval.resize(getNumInputPorts() + getNumOutputPorts(), empty);
  return retval;
}

void MSFTModuleOp::setAllPortLocs(ArrayRef<Location> locs) {
  auto numInputs = getNumInputPorts();
  SmallVector<Attribute> argLocs(locs.begin(), locs.begin() + numInputs);
  SmallVector<Attribute> resLocs(locs.begin() + numInputs, locs.end());
  setArgLocsAttr(ArrayAttr::get(getContext(), argLocs));
  setResultLocsAttr(ArrayAttr::get(getContext(), resLocs));
}

void MSFTModuleOp::setAllPortAttrs(ArrayRef<Attribute> attrs) {
  auto numInputs = getNumInputPorts();
  SmallVector<Attribute> argAttrs(attrs.begin(), attrs.begin() + numInputs);
  SmallVector<Attribute> resAttrs(attrs.begin() + numInputs, attrs.end());
  setArgAttrsAttr(ArrayAttr::get(getContext(), argAttrs));
  setResAttrsAttr(ArrayAttr::get(getContext(), resAttrs));
}

void MSFTModuleOp::removeAllPortAttrs() {
  removeArgAttrsAttr();
  removeResAttrsAttr();
}

void MSFTModuleOp::setAllPortNames(ArrayRef<Attribute> names) {
  auto numInputs = getNumInputPorts();
  SmallVector<Attribute> argNames(names.begin(), names.begin() + numInputs);
  SmallVector<Attribute> resNames(names.begin() + numInputs, names.end());
  setArgNamesAttr(ArrayAttr::get(getContext(), argNames));
  setResultNamesAttr(ArrayAttr::get(getContext(), resNames));
}

void MSFTModuleOp::setHWModuleType(hw::ModuleType type) {
  auto argAttrs = getAllInputAttrs();
  auto resAttrs = getAllOutputAttrs();
  setFunctionTypeAttr(TypeAttr::get(type.getFuncType()));

  unsigned newNumArgs = type.getNumInputs();
  unsigned newNumResults = type.getNumOutputs();

  setArgNamesAttr(ArrayAttr::get(getContext(), type.getInputNames()));
  setResultNamesAttr(ArrayAttr::get(getContext(), type.getOutputNames()));

  auto emptyDict = DictionaryAttr::get(getContext());
  argAttrs.resize(newNumArgs, emptyDict);
  resAttrs.resize(newNumResults, emptyDict);

  SmallVector<Attribute> attrs;
  attrs.append(argAttrs.begin(), argAttrs.end());
  attrs.append(resAttrs.begin(), resAttrs.end());

  if (attrs.empty())
    return removeAllPortAttrs();
  setAllPortAttrs(attrs);
}

//===----------------------------------------------------------------------===//
// MSFTModuleExternOp
//===----------------------------------------------------------------------===//

/// Check parameter specified by `value` to see if it is valid within the
/// scope of the specified module `module`.  If not, emit an error at the
/// location of `usingOp` and return failure, otherwise return success.  If
/// `usingOp` is null, then no diagnostic is generated. Same format as HW
/// dialect.
///
/// If `disallowParamRefs` is true, then parameter references are not allowed.
static LogicalResult checkParameterInContextMSFT(Attribute value,
                                                 Operation *module,
                                                 Operation *usingOp,
                                                 bool disallowParamRefs) {
  // Literals are always ok.  Their types are already known to match
  // expectations.
  if (value.isa<IntegerAttr>() || value.isa<FloatAttr>() ||
      value.isa<StringAttr>() || value.isa<hw::ParamVerbatimAttr>())
    return success();

  // Check both subexpressions of an expression.
  if (auto expr = value.dyn_cast<hw::ParamExprAttr>()) {
    for (auto op : expr.getOperands())
      if (failed(checkParameterInContextMSFT(op, module, usingOp,
                                             disallowParamRefs)))
        return failure();
    return success();
  }

  // Parameter references need more analysis to make sure they are valid
  // within this module.
  if (auto parameterRef = value.dyn_cast<hw::ParamDeclRefAttr>()) {
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
      auto paramAttr = param.cast<hw::ParamDeclAttr>();
      if (paramAttr.getName() != nameAttr)
        continue;

      // If the types match then the reference is ok.
      if (paramAttr.getType() == parameterRef.getType())
        return success();

      if (usingOp) {
        auto diag = usingOp->emitOpError("parameter ")
                    << nameAttr << " used with type " << parameterRef.getType()
                    << "; should have type " << paramAttr.getType();
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

ParseResult MSFTModuleExternOp::parse(OpAsmParser &parser,
                                      OperationState &result) {
  using namespace mlir::function_interface_impl;
  auto loc = parser.getCurrentLocation();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the parameters.
  SmallVector<Attribute> parameters;
  if (parseParameterList(parser, parameters))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  SmallVector<OpAsmParser::Argument, 4> entryArgs;
  SmallVector<Attribute> argNames;
  SmallVector<Attribute> argLocs;
  SmallVector<Attribute> resultNames;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Attribute> resultLocs;
  TypeAttr functionType;
  if (failed(hw::module_like_impl::parseModuleFunctionSignature(
          parser, isVariadic, entryArgs, argNames, argLocs, resultNames,
          resultAttrs, resultLocs, functionType)))
    return failure();

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  if (hasAttribute("resultNames", result.attributes) ||
      hasAttribute("parameters", result.attributes)) {
    parser.emitError(
        loc, "explicit `resultNames` / `parameters` attributes not allowed");
    return failure();
  }

  auto *context = result.getContext();

  // An explicit `argNames` attribute overrides the MLIR names.  This is how
  // we represent port names that aren't valid MLIR identifiers.  Result and
  // parameter names are printed quoted when they aren't valid identifiers, so
  // they don't need this affordance.
  if (!hasAttribute("argNames", result.attributes))
    result.addAttribute("argNames", ArrayAttr::get(context, argNames));
  result.addAttribute("argLocs", ArrayAttr::get(context, argLocs));
  result.addAttribute("resultNames", ArrayAttr::get(context, resultNames));
  result.addAttribute("resultLocs", ArrayAttr::get(context, resultLocs));
  result.addAttribute("parameters", ArrayAttr::get(context, parameters));
  result.addAttribute(MSFTModuleExternOp::getFunctionTypeAttrName(result.name),
                      functionType);

  // Add the attributes to the function arguments.
  addArgAndResultAttrs(parser.getBuilder(), result, entryArgs, resultAttrs,
                       MSFTModuleExternOp::getArgAttrsAttrName(result.name),
                       MSFTModuleExternOp::getResAttrsAttrName(result.name));

  // Extern modules carry an empty region to work with
  // HWModuleImplementation.h.
  result.addRegion();

  return success();
}

void MSFTModuleExternOp::print(OpAsmPrinter &p) {
  using namespace mlir::function_interface_impl;

  auto typeAttr = (*this)->getAttrOfType<TypeAttr>(getFunctionTypeAttrName());
  FunctionType fnType = typeAttr.getValue().cast<FunctionType>();
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // Print the operation and the function name.
  p << ' ';
  p.printSymbolName(SymbolTable::getSymbolName(*this).getValue());

  // Print the parameter list if present.
  printParameterList(p, *this, (*this)->getAttrOfType<ArrayAttr>("parameters"));

  bool needArgNamesAttr = false;
  hw::module_like_impl::printModuleSignature(p, *this, argTypes,
                                             /*isVariadic=*/false, resultTypes,
                                             needArgNamesAttr);

  SmallVector<StringRef, 3> omittedAttrs;
  if (!needArgNamesAttr)
    omittedAttrs.push_back("argNames");
  omittedAttrs.push_back("argLocs");
  omittedAttrs.push_back("resultNames");
  omittedAttrs.push_back("resultLocs");
  omittedAttrs.push_back("parameters");
  omittedAttrs.push_back(getFunctionTypeAttrName());
  omittedAttrs.push_back(getArgAttrsAttrName());
  omittedAttrs.push_back(getResAttrsAttrName());

  printFunctionAttributes(p, *this, omittedAttrs);
}

LogicalResult MSFTModuleExternOp::verify() {
  using namespace mlir::function_interface_impl;
  auto typeAttr = (*this)->getAttrOfType<TypeAttr>(getFunctionTypeAttrName());
  auto moduleType = typeAttr.getValue().cast<FunctionType>();
  auto argNames = (*this)->getAttrOfType<ArrayAttr>("argNames");
  auto resultNames = (*this)->getAttrOfType<ArrayAttr>("resultNames");
  if (argNames.size() != moduleType.getNumInputs())
    return emitOpError("incorrect number of argument names");
  if (resultNames.size() != moduleType.getNumResults())
    return emitOpError("incorrect number of result names");

  SmallPtrSet<Attribute, 4> paramNames;

  // Check parameter default values are sensible.
  for (auto param : (*this)->getAttrOfType<ArrayAttr>("parameters")) {
    auto paramAttr = param.cast<hw::ParamDeclAttr>();

    // Check that we don't have any redundant parameter names.  These are
    // resolved by string name: reuse of the same name would cause
    // ambiguities.
    if (!paramNames.insert(paramAttr.getName()).second)
      return emitOpError("parameter ")
             << paramAttr << " has the same name as a previous parameter";

    // Default values are allowed to be missing, check them if present.
    auto value = paramAttr.getValue();
    if (!value)
      continue;

    auto typedValue = value.dyn_cast<mlir::TypedAttr>();
    if (!typedValue)
      return emitOpError("parameter ")
             << paramAttr << " should have a typed value; has value " << value;

    if (typedValue.getType() != paramAttr.getType())
      return emitOpError("parameter ")
             << paramAttr << " should have type " << paramAttr.getType()
             << "; has type " << typedValue.getType();

    // Verify that this is a valid parameter value, disallowing parameter
    // references.  We could allow parameters to refer to each other in the
    // future with lexical ordering if there is a need.
    if (failed(checkParameterInContextMSFT(value, *this, *this,
                                           /*disallowParamRefs=*/true)))
      return failure();
  }
  return success();
}

hw::ModulePortInfo MSFTModuleExternOp::getPortList() {
  using namespace mlir::function_interface_impl;

  SmallVector<hw::PortInfo> inputs, outputs;

  auto typeAttr =
      getOperation()->getAttrOfType<TypeAttr>(getFunctionTypeAttrName());
  auto moduleType = typeAttr.getValue().cast<FunctionType>();
  auto argTypes = moduleType.getInputs();
  auto resultTypes = moduleType.getResults();

  auto argNames = getOperation()->getAttrOfType<ArrayAttr>("argNames");
  auto argLocs = getOperation()->getAttrOfType<ArrayAttr>("argLocs");
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto type = argTypes[i];
    auto name = argNames[i].cast<StringAttr>();
    auto loc = argLocs[i].cast<LocationAttr>();

    if (auto inout = type.dyn_cast<hw::InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    auto direction = isInOut ? hw::ModulePort::Direction::InOut
                             : hw::ModulePort::Direction::Input;

    inputs.push_back({{name, type, direction}, i, {}, {}, loc});
  }

  auto resultNames = getOperation()->getAttrOfType<ArrayAttr>("resultNames");
  auto resultLocs = getOperation()->getAttrOfType<ArrayAttr>("resultLocs");
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    auto name = resultNames[i].cast<StringAttr>();
    auto loc = resultLocs[i].cast<LocationAttr>();
    outputs.push_back(
        {{name, resultTypes[i], hw::ModulePort::Direction::Output},
         i,
         {},
         {},
         loc});
  }

  return hw::ModulePortInfo(inputs, outputs);
}

size_t MSFTModuleExternOp::getNumPorts() {
  auto typeAttr =
      getOperation()->getAttrOfType<TypeAttr>(getFunctionTypeAttrName());
  auto moduleType = typeAttr.getValue().cast<FunctionType>();
  return moduleType.getNumInputs() + moduleType.getNumResults();
}

size_t MSFTModuleExternOp::getNumInputPorts() {
  auto typeAttr =
      getOperation()->getAttrOfType<TypeAttr>(getFunctionTypeAttrName());
  auto moduleType = typeAttr.getValue().cast<FunctionType>();
  return moduleType.getNumInputs();
}

size_t MSFTModuleExternOp::getNumOutputPorts() {
  auto typeAttr =
      getOperation()->getAttrOfType<TypeAttr>(getFunctionTypeAttrName());
  auto moduleType = typeAttr.getValue().cast<FunctionType>();
  return moduleType.getNumResults();
}

size_t MSFTModuleExternOp::getPortIdForInputId(size_t idx) { return idx; }

size_t MSFTModuleExternOp::getPortIdForOutputId(size_t idx) {
  return idx + getNumInputPorts();
}

hw::InnerSymAttr MSFTModuleExternOp::getPortSymbolAttr(size_t) { return {}; }

SmallVector<Location> MSFTModuleExternOp::getAllPortLocs() {
  SmallVector<Location> retval;
  auto empty = UnknownLoc::get(getContext());
  for (unsigned i = 0, e = getNumPorts(); i < e; ++i)
    retval.push_back(empty);
  return retval;
}

SmallVector<Attribute> MSFTModuleExternOp::getAllPortAttrs() {
  SmallVector<Attribute> retval(getNumPorts());
  if (auto attrs = getArgAttrs()) {
    for (auto a : *attrs)
      retval.push_back(a);
  } else {
    for (unsigned i = 0, e = getNumInputPorts(); i < e; ++i)
      retval.push_back({});
  }
  if (auto attrs = getResAttrs()) {
    for (auto a : *attrs)
      retval.push_back(a);
  } else {
    for (unsigned i = 0, e = getNumOutputPorts(); i < e; ++i)
      retval.push_back({});
  }
  return retval;
}

void MSFTModuleExternOp::setAllPortLocs(ArrayRef<Location> locs) {
  emitError("port locations not supported");
}

void MSFTModuleExternOp::setAllPortAttrs(ArrayRef<Attribute> attrs) {
  auto numInputs = getNumInputPorts();
  SmallVector<Attribute> argAttrs(attrs.begin(), attrs.begin() + numInputs);
  SmallVector<Attribute> resAttrs(attrs.begin() + numInputs, attrs.end());
  setArgAttrsAttr(ArrayAttr::get(getContext(), argAttrs));
  setResAttrsAttr(ArrayAttr::get(getContext(), resAttrs));
}

void MSFTModuleExternOp::removeAllPortAttrs() {
  removeArgAttrsAttr();
  removeResAttrsAttr();
}

void MSFTModuleExternOp::setAllPortNames(ArrayRef<Attribute> names) {
  auto numInputs = getNumInputPorts();
  SmallVector<Attribute> argNames(names.begin(), names.begin() + numInputs);
  SmallVector<Attribute> resNames(names.begin() + numInputs, names.end());
  setArgNamesAttr(ArrayAttr::get(getContext(), argNames));
  setResultNamesAttr(ArrayAttr::get(getContext(), resNames));
}

void MSFTModuleExternOp::setHWModuleType(hw::ModuleType type) {
  auto argAttrs = getAllInputAttrs();
  auto resAttrs = getAllOutputAttrs();
  setFunctionTypeAttr(TypeAttr::get(type.getFuncType()));
  unsigned newNumArgs = getNumInputPorts();
  unsigned newNumResults = getNumOutputPorts();

  auto emptyDict = DictionaryAttr::get(getContext());
  argAttrs.resize(newNumArgs, emptyDict);
  resAttrs.resize(newNumResults, emptyDict);

  SmallVector<Attribute> attrs;
  attrs.append(argAttrs.begin(), argAttrs.end());
  attrs.append(resAttrs.begin(), resAttrs.end());

  if (attrs.empty())
    return removeAllPortAttrs();
  setAllPortAttrs(attrs);
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

void OutputOp::build(OpBuilder &odsBuilder, OperationState &odsState) {}

//===----------------------------------------------------------------------===//
// MSFT high level design constructs
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SystolicArrayOp
//===----------------------------------------------------------------------===//

ParseResult SystolicArrayOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  uint64_t numRows, numColumns;
  Type rowType, columnType;
  OpAsmParser::UnresolvedOperand rowInputs, columnInputs;
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseLSquare() || parser.parseOperand(rowInputs) ||
      parser.parseColon() || parser.parseInteger(numRows) ||
      parser.parseKeyword("x") || parser.parseType(rowType) ||
      parser.parseRSquare() || parser.parseLSquare() ||
      parser.parseOperand(columnInputs) || parser.parseColon() ||
      parser.parseInteger(numColumns) || parser.parseKeyword("x") ||
      parser.parseType(columnType) || parser.parseRSquare())
    return failure();

  hw::ArrayType rowInputType = hw::ArrayType::get(rowType, numRows);
  hw::ArrayType columnInputType = hw::ArrayType::get(columnType, numColumns);
  SmallVector<Value> operands;
  if (parser.resolveOperands({rowInputs, columnInputs},
                             {rowInputType, columnInputType}, loc, operands))
    return failure();
  result.addOperands(operands);

  Type peOutputType;
  SmallVector<OpAsmParser::Argument> peArgs;
  if (parser.parseKeyword("pe")) {
    return failure();
  }
  llvm::SMLoc peLoc = parser.getCurrentLocation();
  if (parser.parseArgumentList(peArgs, AsmParser::Delimiter::Paren)) {
    return failure();
  }
  if (peArgs.size() != 2) {
    return parser.emitError(peLoc, "expected two operands");
  }

  peArgs[0].type = rowType;
  peArgs[1].type = columnType;

  if (parser.parseArrow() || parser.parseLParen() ||
      parser.parseType(peOutputType) || parser.parseRParen())
    return failure();

  result.addTypes({hw::ArrayType::get(
      hw::ArrayType::get(peOutputType, numColumns), numRows)});

  Region *pe = result.addRegion();

  peLoc = parser.getCurrentLocation();

  if (parser.parseRegion(*pe, peArgs))
    return failure();

  if (pe->getBlocks().size() != 1)
    return parser.emitError(peLoc, "expected one block for the PE");
  Operation *peTerm = pe->getBlocks().front().getTerminator();
  if (peTerm->getOperands().size() != 1)
    return peTerm->emitOpError("expected one return value");
  if (peTerm->getOperand(0).getType() != peOutputType)
    return peTerm->emitOpError("expected return type as given in parent: ")
           << peOutputType;

  return success();
}

void SystolicArrayOp::print(OpAsmPrinter &p) {
  hw::ArrayType rowInputType = getRowInputs().getType().cast<hw::ArrayType>();
  hw::ArrayType columnInputType =
      getColInputs().getType().cast<hw::ArrayType>();
  p << " [";
  p.printOperand(getRowInputs());
  p << " : " << rowInputType.getNumElements() << " x ";
  p.printType(rowInputType.getElementType());
  p << "] [";
  p.printOperand(getColInputs());
  p << " : " << columnInputType.getNumElements() << " x ";
  p.printType(columnInputType.getElementType());

  p << "] pe (";
  p.printOperand(getPe().getArgument(0));
  p << ", ";
  p.printOperand(getPe().getArgument(1));
  p << ") -> (";
  p.printType(getPeOutputs()
                  .getType()
                  .cast<hw::ArrayType>()
                  .getElementType()
                  .cast<hw::ArrayType>()
                  .getElementType());
  p << ") ";
  p.printRegion(getPe(), false);
}

//===----------------------------------------------------------------------===//
// LinearOp
//===----------------------------------------------------------------------===//

LogicalResult LinearOp::verify() {

  for (auto &op : *getBodyBlock()) {
    if (!isa<hw::HWDialect, comb::CombDialect, msft::MSFTDialect>(
            op.getDialect()))
      return emitOpError() << "expected only hw, comb, and msft dialect ops "
                              "inside the datapath.";
  }

  return success();
}

#define GET_OP_CLASSES
#include "circt/Dialect/MSFT/MSFT.cpp.inc"
