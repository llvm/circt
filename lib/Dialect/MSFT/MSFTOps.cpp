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
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/ModuleImplementation.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace msft;

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *InstanceOp::getReferencedModule() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  assert(topLevelModuleOp && "Required to have a ModuleOp parent.");
  return topLevelModuleOp.lookupSymbol(moduleName());
}

StringAttr InstanceOp::getResultName(size_t idx) {
  return hw::getModuleResultNameAttr(getReferencedModule(), idx);
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

void MSFTModuleOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                         StringAttr name, ArrayRef<hw::PortInfo> ports,
                         ArrayRef<NamedAttribute> params) {
  assert(false && "Unimplemented");
}

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto *module = symbolTable.lookupNearestSymbolFrom(*this, moduleNameAttr());
  if (module == nullptr)
    return emitError("Cannot find module definition '") << moduleName() << "'";

  // It must be some sort of module.
  if (!hw::isAnyModule(module) && !isa<MSFTModuleOp>(module))
    return emitError("symbol reference '")
           << moduleName() << "' isn't a module";
  return success();
}

static bool hasAttribute(StringRef name, ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs)
    if (argAttr.first == name)
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
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

#define GET_OP_CLASSES
#include "circt/Dialect/MSFT/MSFT.cpp.inc"
