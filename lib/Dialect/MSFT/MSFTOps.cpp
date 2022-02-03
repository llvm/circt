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
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"
#include "circt/Dialect/MSFT/MSFTAttributes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/FunctionSupport.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *InstanceOp::getReferencedModule() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;
  return topLevelModuleOp.lookupSymbol(moduleName());
}

StringAttr InstanceOp::getResultName(size_t idx) {
  if (auto *refMod = getReferencedModule())
    return hw::getModuleResultNameAttr(refMod, idx);
  return StringAttr();
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Provide default names for instance results.
  std::string name = instanceName().str() + ".";
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
  if (ports.inputs.size() != getNumOperands())
    return emitOpError("wrong number of inputs (expected ")
           << ports.inputs.size() << ")";
  if (ports.outputs.size() != getNumResults())
    return emitOpError("wrong number of outputs (expected ")
           << ports.outputs.size() << ")";
  for (auto port : ports.inputs)
    if (getOperand(port.argNum).getType() != port.type)
      return emitOpError("in input port ")
             << port.name << ", expected type " << port.type << " got "
             << getOperand(port.argNum).getType();
  for (auto port : ports.outputs)
    if (getResult(port.argNum).getType() != port.type)
      return emitOpError("in output port ")
             << port.name << ", expected type " << port.type << " got "
             << getResult(port.argNum).getType();

  return success();
}

/// Return an encapsulated set of information about input and output ports of
/// the specified module or instance.  The input ports always come before the
/// output ports in the list.
/// TODO: This should really be shared with the HW dialect instead of cloned.
/// Consider adding a `HasModulePorts` op interface to facilitate.
hw::ModulePortInfo MSFTModuleOp::getPorts() {
  SmallVector<hw::PortInfo> inputs, outputs;
  auto argTypes = getType().getInputs();

  auto argNames = this->argNames();
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<hw::InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    auto direction =
        isInOut ? hw::PortDirection::INOUT : hw::PortDirection::INPUT;
    inputs.push_back({argNames[i].cast<StringAttr>(), direction, type, i});
  }

  auto resultNames = this->resultNames();
  auto resultTypes = getType().getResults();
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    outputs.push_back({resultNames[i].cast<StringAttr>(),
                       hw::PortDirection::OUTPUT, resultTypes[i], i});
  }
  return hw::ModulePortInfo(inputs, outputs);
}

SmallVector<BlockArgument>
MSFTModuleOp::addPorts(ArrayRef<std::pair<StringAttr, Type>> inputs,
                       ArrayRef<std::pair<StringAttr, Value>> outputs) {
  auto *ctxt = getContext();
  FunctionType ftype = getType();
  Block *body = getBodyBlock();

  // Append new inputs.
  SmallVector<Type, 32> modifiedArgs(ftype.getInputs().begin(),
                                     ftype.getInputs().end());
  SmallVector<Attribute> modifiedArgNames(argNames().getAsRange<Attribute>());
  SmallVector<BlockArgument> newBlockArgs;
  for (auto ttPair : inputs) {
    modifiedArgNames.push_back(ttPair.first);
    modifiedArgs.push_back(ttPair.second);
    newBlockArgs.push_back(body->addArgument(ttPair.second));
  }
  argNamesAttr(ArrayAttr::get(ctxt, modifiedArgNames));

  // Append new outputs.
  SmallVector<Type, 32> modifiedResults(ftype.getResults().begin(),
                                        ftype.getResults().end());
  SmallVector<Attribute> modifiedResultNames(
      resultNames().getAsRange<Attribute>());
  Operation *terminator = body->getTerminator();
  SmallVector<Value, 32> modifiedOutputs(terminator->getOperands());
  for (auto tvPair : outputs) {
    modifiedResultNames.push_back(tvPair.first);
    modifiedResults.push_back(tvPair.second.getType());
    modifiedOutputs.push_back(tvPair.second);
  }
  resultNamesAttr(ArrayAttr::get(ctxt, modifiedResultNames));
  terminator->setOperands(modifiedOutputs);

  // Finalize and return.
  setType(FunctionType::get(ctxt, modifiedArgs, modifiedResults));
  return newBlockArgs;
}

// Remove the ports at the specified indexes.
SmallVector<unsigned> MSFTModuleOp::removePorts(llvm::BitVector inputs,
                                                llvm::BitVector outputs) {
  MLIRContext *ctxt = getContext();
  FunctionType ftype = getType();
  Block *body = getBodyBlock();
  Operation *terminator = body->getTerminator();

  SmallVector<Type, 4> newInputTypes;
  SmallVector<Attribute, 4> newArgNames;
  unsigned originalNumArgs = ftype.getNumInputs();
  ArrayRef<Attribute> origArgNames = argNamesAttr().getValue();
  assert(origArgNames.size() == originalNumArgs);
  for (size_t i = 0; i < originalNumArgs; ++i) {
    if (!inputs.test(i)) {
      newInputTypes.emplace_back(ftype.getInput(i));
      newArgNames.emplace_back(origArgNames[i]);
    }
  }

  SmallVector<Type, 4> newResultTypes;
  SmallVector<Attribute, 4> newResultNames;
  unsigned originalNumResults = getNumResults();
  ArrayRef<Attribute> origResNames = resultNamesAttr().getValue();
  assert(origResNames.size() == originalNumResults);
  for (size_t i = 0; i < originalNumResults; ++i) {
    if (!outputs.test(i)) {
      newResultTypes.emplace_back(ftype.getResult(i));
      newResultNames.emplace_back(origResNames[i]);
    }
  }

  setType(FunctionType::get(ctxt, newInputTypes, newResultTypes));
  resultNamesAttr(ArrayAttr::get(ctxt, newResultNames));
  argNamesAttr(ArrayAttr::get(ctxt, newArgNames));

  // Build new operand list for output op. Construct an output mapping to
  // return as a side-effect.
  unsigned numResults = ftype.getNumResults();
  SmallVector<Value> newOutputValues;
  SmallVector<unsigned> newToOldResultMap;

  for (unsigned i = 0; i < numResults; ++i) {
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

// Copied nearly exactly from hwops.cpp.
// TODO: Unify code once a `ModuleLike` op interface exists.
static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, const hw::ModulePortInfo &ports) {
  using namespace mlir::function_like_impl;

  // Add an attribute for the name.
  result.addAttribute(SymbolTable::getSymbolAttrName(), name);

  SmallVector<Attribute> argNames, resultNames;
  SmallVector<Type, 4> argTypes, resultTypes;
  SmallVector<Attribute> argAttrs, resultAttrs;
  auto exportPortIdent = StringAttr::get(builder.getContext(), "hw.exportPort");

  for (auto elt : ports.inputs) {
    if (elt.direction == hw::PortDirection::INOUT &&
        !elt.type.isa<hw::InOutType>())
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

  // Record the argument and result types as an attribute.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  result.addAttribute("argNames", builder.getArrayAttr(argNames));
  result.addAttribute("resultNames", builder.getArrayAttr(resultNames));
  result.addAttribute("parameters", builder.getDictionaryAttr({}));
  result.addAttribute(mlir::function_like_impl::getArgDictAttrName(),
                      builder.getArrayAttr(argAttrs));
  result.addAttribute(mlir::function_like_impl::getResultDictAttrName(),
                      builder.getArrayAttr(resultAttrs));
  result.addRegion();
}

void MSFTModuleOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name, hw::ModulePortInfo ports,
                         ArrayRef<NamedAttribute> params) {
  buildModule(builder, result, name, ports);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports.inputs)
    body->addArgument(elt.type);

  MSFTModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

void MSFTModuleOp::getAsmBlockArgumentNames(Region &region,
                                            OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;
  Block *block = getBodyBlock();
  ArrayAttr argNames = argNamesAttr();
  for (size_t i = 0, e = block->getNumArguments(); i != e; ++i) {
    auto name = argNames[i].cast<StringAttr>().getValue();
    if (!name.empty())
      setNameFn(block->getArgument(i), name);
  }
}

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module = symbolTable.lookupNearestSymbolFrom(*this, moduleNameAttr());
  if (module == nullptr)
    return emitError("Cannot find module definition '") << moduleName() << "'";

  // It must be some sort of module.
  if (!hw::isAnyModule(module) &&
      !isa<MSFTModuleOp, MSFTModuleExternOp>(module))
    return emitError("symbol reference '")
           << moduleName() << "' isn't a module";
  return success();
}

static bool hasAttribute(StringRef name, ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs)
    if (argAttr.getName() == name)
      return true;
  return false;
}

static ParseResult parseMSFTModuleOp(OpAsmParser &parser,
                                     OperationState &result) {
  using namespace mlir::function_like_impl;

  auto loc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the parameters
  DictionaryAttr paramsAttr;
  if (parser.parseAttribute(paramsAttr))
    return failure();
  result.addAttribute("parameters", paramsAttr);

  // Parse the function signature.
  bool isVariadic = false;
  SmallVector<Attribute> resultNames;
  if (hw::module_like_impl::parseModuleFunctionSignature(
          parser, entryArgs, argTypes, argAttrs, isVariadic, resultTypes,
          resultAttrs, resultNames))
    return failure();

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // If function attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  auto *context = result.getContext();

  if (hasAttribute("argNames", result.attributes) ||
      hasAttribute("resultNames", result.attributes)) {
    parser.emitError(
        loc, "explicit argNames and resultNames attributes not allowed");
    return failure();
  }

  // Use the argument and result names if not already specified.
  SmallVector<Attribute> argNames;
  if (!entryArgs.empty()) {
    for (auto &arg : entryArgs)
      argNames.push_back(
          hw::module_like_impl::getPortNameAttr(context, arg.name));
  } else if (!argTypes.empty()) {
    // The parser returns empty names in a special way.
    argNames.assign(argTypes.size(), StringAttr::get(context, ""));
  }

  result.addAttribute("argNames", ArrayAttr::get(context, argNames));
  result.addAttribute("resultNames", ArrayAttr::get(context, resultNames));

  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());

  // Add the attributes to the module arguments.
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional module body.
  auto regionSuccess = parser.parseOptionalRegion(
      *result.addRegion(), entryArgs,
      entryArgs.empty() ? ArrayRef<Type>() : argTypes);
  if (regionSuccess.hasValue() && failed(*regionSuccess))
    return failure();

  return success();
}

static void printMSFTModuleOp(OpAsmPrinter &p, MSFTModuleOp mod) {
  using namespace mlir::function_like_impl;

  FunctionType fnType = mod.getType();
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // Print the operation and the function name.
  p << ' ';
  p.printSymbolName(SymbolTable::getSymbolName(mod).getValue());

  // Print the parameterization.
  p << ' ';
  p.printAttribute(mod.parametersAttr());

  p << ' ';
  bool needArgNamesAttr = false;
  hw::module_like_impl::printModuleSignature(
      p, mod, argTypes, /*isVariadic=*/false, resultTypes, needArgNamesAttr);

  SmallVector<StringRef, 3> omittedAttrs;
  if (!needArgNamesAttr)
    omittedAttrs.push_back("argNames");
  omittedAttrs.push_back("resultNames");
  omittedAttrs.push_back("parameters");

  printFunctionAttributes(p, mod, argTypes.size(), resultTypes.size(),
                          omittedAttrs);

  // Print the body if this is not an external function.
  Region &body = mod.getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
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
            builder.getContext(), builder.getStringAttr(name),
            TypeAttr::get(type), value));
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

/// Check parameter specified by `value` to see if it is valid within the scope
/// of the specified module `module`.  If not, emit an error at the location of
/// `usingOp` and return failure, otherwise return success.  If `usingOp` is
/// null, then no diagnostic is generated. Same format as HW dialect.
///
/// If `disallowParamRefs` is true, then parameter references are not allowed.
static LogicalResult checkParameterInContext(Attribute value, Operation *module,
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
      if (failed(
              checkParameterInContext(op, module, usingOp, disallowParamRefs)))
        return failure();
    return success();
  }

  // Parameter references need more analysis to make sure they are valid within
  // this module.
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

static ParseResult parseMSFTModuleExternOp(OpAsmParser &parser,
                                           OperationState &result) {
  using namespace mlir::function_like_impl;

  auto loc = parser.getCurrentLocation();

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute> parameters;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;
  SmallVector<Attribute> resultNames;
  if (parseParameterList(parser, parameters) ||
      hw::module_like_impl::parseModuleFunctionSignature(
          parser, entryArgs, argTypes, argAttrs, isVariadic, resultTypes,
          resultAttrs, resultNames) ||
      // If function attributes are present, parse them.
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.
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
          hw::module_like_impl::getPortNameAttr(context, arg.name));
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

  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());

  // Add the attributes to the function arguments.
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Extern modules carry an empty region to work with HWModuleImplementation.h.
  result.addRegion();

  return success();
}

static void printMSFTModuleExternOp(OpAsmPrinter &p, MSFTModuleExternOp op) {
  using namespace mlir::function_like_impl;

  auto typeAttr = op->getAttrOfType<TypeAttr>(getTypeAttrName());
  FunctionType fnType = typeAttr.getValue().cast<FunctionType>();
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // Print the operation and the function name.
  p << ' ';
  p.printSymbolName(SymbolTable::getSymbolName(op).getValue());

  // Print the parameter list if present.
  printParameterList(p, op, op->getAttrOfType<ArrayAttr>("parameters"));

  bool needArgNamesAttr = false;
  hw::module_like_impl::printModuleSignature(
      p, op, argTypes, /*isVariadic=*/false, resultTypes, needArgNamesAttr);

  SmallVector<StringRef, 3> omittedAttrs;
  if (!needArgNamesAttr)
    omittedAttrs.push_back("argNames");
  omittedAttrs.push_back("resultNames");
  omittedAttrs.push_back("parameters");

  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size(),
                          omittedAttrs);
}

static LogicalResult verifyMSFTModuleExternOp(MSFTModuleExternOp module) {
  using namespace mlir::function_like_impl;
  auto typeAttr = module->getAttrOfType<TypeAttr>(getTypeAttrName());
  auto moduleType = typeAttr.getValue().cast<FunctionType>();
  auto argNames = module->getAttrOfType<ArrayAttr>("argNames");
  auto resultNames = module->getAttrOfType<ArrayAttr>("resultNames");
  if (argNames.size() != moduleType.getNumInputs())
    return module->emitOpError("incorrect number of argument names");
  if (resultNames.size() != moduleType.getNumResults())
    return module->emitOpError("incorrect number of result names");

  SmallPtrSet<Attribute, 4> paramNames;

  // Check parameter default values are sensible.
  for (auto param : module->getAttrOfType<ArrayAttr>("parameters")) {
    auto paramAttr = param.cast<hw::ParamDeclAttr>();

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

hw::ModulePortInfo MSFTModuleExternOp::getPorts() {
  using namespace mlir::function_like_impl;

  SmallVector<hw::PortInfo> inputs, outputs;

  auto typeAttr = getOperation()->getAttrOfType<TypeAttr>(getTypeAttrName());
  auto moduleType = typeAttr.getValue().cast<FunctionType>();
  auto argTypes = moduleType.getInputs();
  auto resultTypes = moduleType.getResults();

  auto argNames = getOperation()->getAttrOfType<ArrayAttr>("argNames");
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<hw::InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    auto direction =
        isInOut ? hw::PortDirection::INOUT : hw::PortDirection::INPUT;

    inputs.push_back({argNames[i].cast<StringAttr>(), direction, type, i});
  }

  auto resultNames = getOperation()->getAttrOfType<ArrayAttr>("resultNames");
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i)
    outputs.push_back({resultNames[i].cast<StringAttr>(),
                       hw::PortDirection::OUTPUT, resultTypes[i], i});

  return hw::ModulePortInfo(inputs, outputs);
}

void OutputOp::build(OpBuilder &builder, OperationState &result) {}

#define GET_OP_CLASSES
#include "circt/Dialect/MSFT/MSFT.cpp.inc"
