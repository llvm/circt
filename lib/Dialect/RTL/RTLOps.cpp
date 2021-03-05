//===- RTLOps.cpp - Implement the RTL operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the RTL ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/RTL/RTLOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/RTL/RTLVisitors.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace circt;
using namespace rtl;

/// Return true if the specified operation is a combinatorial logic op.
bool rtl::isCombinatorial(Operation *op) {
  struct IsCombClassifier : public TypeOpVisitor<IsCombClassifier, bool> {
    bool visitInvalidTypeOp(Operation *op) { return false; }
    bool visitUnhandledTypeOp(Operation *op) { return true; }
  };

  return op->getDialect()->getNamespace() == "comb" ||
         IsCombClassifier().dispatchTypeOpVisitor(op);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static void printConstantOp(OpAsmPrinter &p, ConstantOp &op) {
  p << "rtl.constant ";
  p.printAttribute(op.valueAttr());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{"value"});
}

static ParseResult parseConstantOp(OpAsmParser &parser,
                                   OperationState &result) {
  IntegerAttr valueAttr;

  if (parser.parseAttribute(valueAttr, "value", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  result.addTypes(valueAttr.getType());
  return success();
}

static LogicalResult verifyConstantOp(ConstantOp constant) {
  // If the result type has a bitwidth, then the attribute must match its width.
  if (constant.value().getBitWidth() != constant.getType().getWidth())
    return constant.emitError(
        "rtl.constant attribute bitwidth doesn't match return type");

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
    return setNameFn(getResult(), intCst.isNullValue() ? "false" : "true");

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
// RTLModuleOp
//===----------------------------------------------------------------------===/

static void buildModule(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<ModulePortInfo> ports) {
  using namespace mlir::impl;

  // Add an attribute for the name.
  result.addAttribute(::mlir::SymbolTable::getSymbolAttrName(), name);

  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  for (auto elt : ports) {
    if (elt.isOutput())
      resultTypes.push_back(elt.type);
    else
      argTypes.push_back(elt.type);
  }

  // Record the argument and result types as an attribute.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // Record the names of the arguments if present.
  SmallString<8> attrNameBuf;
  SmallString<8> attrDirBuf;
  for (const ModulePortInfo &port : ports) {
    SmallVector<NamedAttribute, 2> argAttrs;
    if (!port.name.getValue().empty())
      argAttrs.push_back(
          NamedAttribute(builder.getIdentifier("rtl.name"), port.name));

    StringRef attrName = port.isOutput()
                             ? getResultAttrName(port.argNum, attrNameBuf)
                             : getArgAttrName(port.argNum, attrNameBuf);
    result.addAttribute(attrName, builder.getDictionaryAttr(argAttrs));
  }
  result.addRegion();
}

void RTLModuleOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name, ArrayRef<ModulePortInfo> ports) {
  buildModule(builder, result, name, ports);

  // Create a region and a block for the body.
  auto *bodyRegion = result.regions[0].get();
  Block *body = new Block();
  bodyRegion->push_back(body);

  // Add arguments to the body block.
  for (auto elt : ports)
    if (!elt.isOutput())
      body->addArgument(elt.type);

  RTLModuleOp::ensureTerminator(*bodyRegion, builder, result.location);
}

/// Return the name to use for the Verilog module that we're referencing
/// here.  This is typically the symbol, but can be overridden with the
/// verilogName attribute.
StringRef RTLModuleExternOp::getVerilogModuleName() {
  if (auto vname = verilogName())
    return vname.getValue();
  return getName();
}

void RTLModuleExternOp::build(OpBuilder &builder, OperationState &result,
                              StringAttr name, ArrayRef<ModulePortInfo> ports,
                              StringRef verilogName) {
  buildModule(builder, result, name, ports);

  if (!verilogName.empty())
    result.addAttribute("verilogName", builder.getStringAttr(verilogName));
}

FunctionType rtl::getModuleType(Operation *op) {
  auto typeAttr = op->getAttrOfType<TypeAttr>(RTLModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
}

StringAttr rtl::getRTLNameAttr(ArrayRef<NamedAttribute> attrs) {
  for (auto &argAttr : attrs) {
    if (argAttr.first != "rtl.name")
      continue;
    return argAttr.second.dyn_cast<StringAttr>();
  }
  return StringAttr();
}

void rtl::getModulePortInfo(Operation *op,
                            SmallVectorImpl<ModulePortInfo> &results) {
  auto argTypes = getModuleType(op).getInputs();

  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    auto argAttrs = ::mlir::impl::getArgAttrs(op, i);
    bool isInOut = false;
    auto type = argTypes[i];

    if (auto inout = type.dyn_cast<InOutType>()) {
      isInOut = true;
      type = inout.getElementType();
    }

    results.push_back({getRTLNameAttr(argAttrs),
                       isInOut ? PortDirection::INOUT : PortDirection::INPUT,
                       type, i});
  }

  auto resultTypes = getModuleType(op).getResults();
  for (unsigned i = 0, e = resultTypes.size(); i < e; ++i) {
    auto argAttrs = ::mlir::impl::getResultAttrs(op, i);
    results.push_back(
        {getRTLNameAttr(argAttrs), PortDirection::OUTPUT, resultTypes[i], i});
  }
}

/// Parse a function result list.
///
///   function-result-list ::= function-result-list-parens
///   function-result-list-parens ::= `(` `)`
///                                 | `(` function-result-list-no-parens `)`
///   function-result-list-no-parens ::= function-result (`,` function-result)*
///   function-result ::= (percent-identifier `:`) type attribute-dict?
///
static ParseResult
parseFunctionResultList(OpAsmParser &parser, SmallVectorImpl<Type> &resultTypes,
                        SmallVectorImpl<NamedAttrList> &resultAttrs) {
  if (parser.parseLParen())
    return failure();

  // Special case for an empty set of parens.
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  // Parse individual function results.
  do {
    resultTypes.emplace_back();
    resultAttrs.emplace_back();

    OpAsmParser::OperandType operandName;
    auto namePresent = parser.parseOptionalOperand(operandName);
    StringRef implicitName;
    if (namePresent.hasValue()) {
      if (namePresent.getValue() || parser.parseColon())
        return failure();

      // If the name was specified, then we will use it.
      implicitName = operandName.name.drop_front();
    }

    if (parser.parseType(resultTypes.back()) ||
        parser.parseOptionalAttrDict(resultAttrs.back()))
      return failure();

    // If we have an implicit name and no explicit rtl.name attribute, then use
    // the implicit name as the rtl.name attribute.
    if (!implicitName.empty() && !getRTLNameAttr(resultAttrs.back())) {
      auto nameAttr = parser.getBuilder().getStringAttr(implicitName);
      resultAttrs.back().append("rtl.name", nameAttr);
    }
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRParen();
}

/// This is a variant of mlor::parseFunctionSignature that allows names on
/// result arguments.
static ParseResult parseModuleFunctionSignature(
    OpAsmParser &parser, bool allowVariadic,
    SmallVectorImpl<OpAsmParser::OperandType> &argNames,
    SmallVectorImpl<Type> &argTypes, SmallVectorImpl<NamedAttrList> &argAttrs,
    bool &isVariadic, SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<NamedAttrList> &resultAttrs) {
  using namespace mlir::impl;
  bool allowArgAttrs = true;
  if (parseFunctionArgumentList(parser, allowArgAttrs, allowVariadic, argNames,
                                argTypes, argAttrs, isVariadic))
    return failure();
  if (succeeded(parser.parseOptionalArrow()))
    return parseFunctionResultList(parser, resultTypes, resultAttrs);
  return success();
}

static ParseResult parseRTLModuleOp(OpAsmParser &parser, OperationState &result,
                                    bool isExtModule = false) {
  using namespace mlir::impl;

  SmallVector<OpAsmParser::OperandType, 4> entryArgs;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  bool isVariadic = false;

  if (parseModuleFunctionSignature(parser, /*allowVariadic=*/false, entryArgs,
                                   argTypes, argAttrs, isVariadic, resultTypes,
                                   resultAttrs))
    return failure();

  // Record the argument and result types as an attribute.  This is necessary
  // for external modules.
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));

  // If function attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  assert(argAttrs.size() == argTypes.size());
  assert(resultAttrs.size() == resultTypes.size());

  auto *context = result.getContext();

  // Postprocess each of the arguments.  If there was no 'rtl.name'
  // attribute, and if the argument name was non-numeric, then add the
  // rtl.name attribute with the textual name from the IR.  The name in the
  // text file is a load-bearing part of the IR, but we don't want the
  // verbosity in dumps of including it explicitly in the attribute
  // dictionary.
  for (size_t i = 0, e = argAttrs.size(); i != e; ++i) {
    auto &attrs = argAttrs[i];

    // If an explicit name attribute was present, don't add the implicit one.
    bool hasNameAttr = false;
    for (auto &elt : attrs)
      if (elt.first.str() == "rtl.name")
        hasNameAttr = true;
    if (hasNameAttr || entryArgs.empty())
      continue;

    auto &arg = entryArgs[i];

    // The name of an argument is of the form "%42" or "%id", and since
    // parsing succeeded, we know it always has one character.
    assert(arg.name.size() > 1 && arg.name[0] == '%' && "Unknown MLIR name");
    if (isdigit(arg.name[1]))
      continue;

    auto nameAttr = StringAttr::get(context, arg.name.drop_front());
    attrs.push_back({Identifier::get("rtl.name", context), nameAttr});
  }

  // Add the attributes to the function arguments.
  addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the optional function body.
  auto *body = result.addRegion();
  if (!isExtModule) {
    if (parser.parseRegion(*body, entryArgs,
                           entryArgs.empty() ? ArrayRef<Type>() : argTypes))
      return failure();

    RTLModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  }
  return success();
}

static ParseResult parseRTLModuleExternOp(OpAsmParser &parser,
                                          OperationState &result) {
  return parseRTLModuleOp(parser, result, /*isExtModule:*/ true);
}

FunctionType getRTLModuleOpType(Operation *op) {
  auto typeAttr = op->getAttrOfType<TypeAttr>(RTLModuleOp::getTypeAttrName());
  return typeAttr.getValue().cast<FunctionType>();
}

static void printModuleSignature(OpAsmPrinter &p, Operation *op,
                                 ArrayRef<Type> argTypes, bool isVariadic,
                                 ArrayRef<Type> resultTypes) {
  Region &body = op->getRegion(0);
  bool isExternal = body.empty();

  p << '(';
  for (unsigned i = 0, e = argTypes.size(); i < e; ++i) {
    if (i > 0)
      p << ", ";

    Value argumentValue;
    if (!isExternal) {
      argumentValue = body.front().getArgument(i);
      p.printOperand(argumentValue);
      p << ": ";
    }

    p.printType(argTypes[i]);

    auto argAttrs = ::mlir::impl::getArgAttrs(op, i);

    // If the argument has the rtl.name attribute, and if it was used by
    // the printer exactly (not name mangled with a suffix etc) then we can
    // omit the rtl.name attribute from the argument attribute dictionary.
    ArrayRef<StringRef> elidedAttrs;
    StringRef tmp;
    if (argumentValue) {
      if (auto nameAttr = getRTLNameAttr(argAttrs)) {

        // Check to make sure the asmprinter is printing it correctly.
        SmallString<32> resultNameStr;
        llvm::raw_svector_ostream tmpStream(resultNameStr);
        p.printOperand(argumentValue, tmpStream);

        // If the name is the same as we would otherwise use, then we're good!
        if (tmpStream.str().drop_front() == nameAttr.getValue()) {
          tmp = "rtl.name";
          elidedAttrs = tmp;
        }
      }
    }
    p.printOptionalAttrDict(argAttrs, elidedAttrs);
  }

  if (isVariadic) {
    if (!argTypes.empty())
      p << ", ";
    p << "...";
  }

  p << ')';

  // We print result types specially since we support named arguments.
  if (!resultTypes.empty()) {
    auto &os = p.getStream();
    os << " -> (";
    for (size_t i = 0, e = resultTypes.size(); i < e; ++i) {
      if (i != 0)
        os << ", ";
      auto resultAttrs = ::mlir::impl::getResultAttrs(op, i);
      StringAttr name = getRTLNameAttr(resultAttrs);
      if (name)
        os << '%' << name.getValue() << ": ";

      p.printType(resultTypes[i]);
      p.printOptionalAttrDict(resultAttrs, {"rtl.name"});
    }
    os << ')';
  }
}

static void printRTLModuleOp(OpAsmPrinter &p, Operation *op) {
  using namespace mlir::impl;

  FunctionType fnType = getRTLModuleOpType(op);
  auto argTypes = fnType.getInputs();
  auto resultTypes = fnType.getResults();

  // Print the operation and the function name.
  auto funcName =
      op->getAttrOfType<StringAttr>(::mlir::SymbolTable::getSymbolAttrName())
          .getValue();
  p << op->getName() << ' ';
  p.printSymbolName(funcName);

  printModuleSignature(p, op, argTypes, /*isVariadic=*/false, resultTypes);
  printFunctionAttributes(p, op, argTypes.size(), resultTypes.size());
}

static void print(OpAsmPrinter &p, RTLModuleExternOp op) {
  printRTLModuleOp(p, op);
}

static void print(OpAsmPrinter &p, RTLModuleOp op) {
  printRTLModuleOp(p, op);

  // Print the body if this is not an external function.
  Region &body = op.getBody();
  if (!body.empty())
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

/// Lookup the module or extmodule for the symbol.  This returns null on
/// invalid IR.
Operation *InstanceOp::getReferencedModule() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;

  return topLevelModuleOp.lookupSymbol(moduleName());
}

// Helper function to verify instance op types
static LogicalResult verifyInstanceOpTypes(InstanceOp op) {
  auto referencedModule = op.getReferencedModule();
  assert(referencedModule && "referenced module must not be null");

  // Check operand types first.
  auto numOperands = op->getNumOperands();
  auto expectedOperandTypes = getModuleType(referencedModule).getInputs();

  if (expectedOperandTypes.size() != numOperands) {
    auto diag = op.emitOpError()
                << "has a wrong number of operands; expected "
                << expectedOperandTypes.size() << " but got " << numOperands;
    diag.attachNote(referencedModule->getLoc())
        << "original module declared here";

    return failure();
  }

  for (size_t i = 0; i != numOperands; ++i) {
    auto expectedType = expectedOperandTypes[i];
    auto operandType = op.getOperand(i).getType();
    if (operandType != expectedType) {
      auto diag = op.emitOpError()
                  << "#" << i << " operand type must be " << expectedType
                  << ", but got " << operandType;

      diag.attachNote(referencedModule->getLoc())
          << "original module declared here";
      return failure();
    }
  }

  // Checke result types.
  auto numResults = op->getNumResults();
  auto expectedResultTypes = getModuleType(referencedModule).getResults();

  if (expectedResultTypes.size() != numResults) {
    auto diag = op.emitOpError()
                << "has a wrong number of results; expected "
                << expectedResultTypes.size() << " but got " << numResults;
    diag.attachNote(referencedModule->getLoc())
        << "original module declared here";

    return failure();
  }

  for (size_t i = 0; i != numResults; ++i) {
    auto expectedType = expectedResultTypes[i];
    auto resultType = op.getResult(i).getType();
    if (resultType != expectedType) {
      auto diag = op.emitOpError()
                  << "#" << i << " result type must be " << expectedType
                  << ", but got " << resultType;

      diag.attachNote(referencedModule->getLoc())
          << "original module declared here";
      return failure();
    }
  }

  return success();
}

static LogicalResult verifyInstanceOp(InstanceOp op) {
  // Check that this instance is inside a module.
  auto module = dyn_cast<RTLModuleOp>(op->getParentOp());
  if (!module) {
    op.emitOpError("should be embedded in an 'rtl.module'");
    return failure();
  }

  auto referencedModule = op.getReferencedModule();
  if (referencedModule == nullptr)
    return op.emitError("Cannot find module definition '")
           << op.moduleName() << "'";

  if (!isa<RTLModuleOp>(referencedModule) &&
      !isa<RTLModuleExternOp>(referencedModule))
    return op.emitError("Symbol resolved to '")
           << referencedModule->getName()
           << "' which is not a RTL[Ext]ModuleOp";

  if (auto paramDictOpt = op.parameters()) {
    DictionaryAttr paramDict = paramDictOpt.getValue();
    auto checkParmValue = [&](NamedAttribute elt) -> bool {
      auto value = elt.second;
      if (value.isa<IntegerAttr>() || value.isa<StringAttr>() ||
          value.isa<FloatAttr>())
        return true;
      op.emitError() << "has unknown extmodule parameter value '" << elt.first
                     << "' = " << value;
      return false;
    };

    if (!llvm::all_of(paramDict, checkParmValue))
      return failure();
  }

  // If the referenced moudle is internal, check that input and result types are
  // consistent with the referenced module.
  if (!isa<RTLModuleOp>(referencedModule))
    return success();

  return verifyInstanceOpTypes(op);
}

StringAttr InstanceOp::getResultName(size_t idx) {
  auto *module = getReferencedModule();
  if (!module)
    return {};

  SmallVector<ModulePortInfo, 4> results;
  getModulePortInfo(module, results);

  for (auto &port : results) {
    if (!port.isOutput())
      continue;
    if (idx == 0)
      return port.name;
    --idx;
  }

  return StringAttr();
}

/// Suggest a name for each result value based on the saved result names
/// attribute.
void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  // Provide default names for instance results.
  std::string name;
  for (size_t i = 0, e = getNumResults(); i != e; ++i) {
    auto resultName = getResultName(i);
    name = instanceName().str() + ".";
    if (resultName)
      name += resultName.getValue().str();
    else
      name += std::to_string(i);
    setNameFn(getResult(i), name);
  }
}

//===----------------------------------------------------------------------===//
// RTLOutputOp
//===----------------------------------------------------------------------===//

/// Verify that the num of operands and types fit the declared results.
static LogicalResult verifyOutputOp(OutputOp *op) {
  OperandRange outputValues = op->getOperands();
  auto opParent = (*op)->getParentOp();

  // Check that we are in the correct region. OutputOp should be directly
  // contained by an RTLModuleOp region. We'll loosen this restriction if
  // there's a compelling use case.
  if (!isa<RTLModuleOp>(opParent)) {
    op->emitOpError("operation expected to be in a RTLModuleOp.");
    return failure();
  }

  // Check that the we (rtl.output) have the same number of operands as our
  // region has results.
  FunctionType modType = getModuleType(opParent);
  ArrayRef<Type> modResults = modType.getResults();
  if (modResults.size() != outputValues.size()) {
    op->emitOpError("must have same number of operands as region results.");
    return failure();
  }

  // Check that the types of our operands and the region's results match.
  for (size_t i = 0, e = modResults.size(); i < e; ++i) {
    if (modResults[i] != outputValues[i].getType()) {
      op->emitOpError("output types must match module. In "
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

static ParseResult parseSliceTypes(OpAsmParser &p, Type &srcType,
                                   Type &idxType) {
  ArrayType arrType;
  if (p.parseType(arrType))
    return failure();
  srcType = arrType;
  unsigned idxWidth = llvm::Log2_64_Ceil(arrType.getSize());
  idxType = IntegerType::get(p.getBuilder().getContext(), idxWidth);
  return success();
}

static void printSliceTypes(OpAsmPrinter &p, Operation *, Type srcType,
                            Type idxType) {
  p.printType(srcType);
}

static ParseResult parseArrayCreateOp(OpAsmParser &parser,
                                      OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::OperandType, 16> operands;
  Type elemType;

  if (parser.parseOperandList(operands) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseLParen() || parser.parseType(elemType) ||
      parser.parseRParen())
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

static void print(OpAsmPrinter &p, ArrayCreateOp op) {
  p << "rtl.array_create ";
  p.printOperands(op.inputs());
  p << " : (" << op.inputs()[0].getType() << ")";
}

void ArrayCreateOp::build(OpBuilder &b, OperationState &state,
                          ArrayRef<Value> values) {
  assert(values.size() > 0 && "Cannot build array of zero elements");
  Type elemType = values[0].getType();
  assert(llvm::all_of(
             values,
             [elemType](Value v) -> bool { return v.getType() == elemType; }) &&
         "All values must have same type.");
  build(b, state, ArrayType::get(elemType, values.size()), values);
}

static ParseResult parseArrayConcatTypes(OpAsmParser &p,
                                         SmallVectorImpl<Type> &inputTypes,
                                         Type &resultType) {
  Type elemType;
  uint64_t resultSize = 0;
  do {
    ArrayType ty;
    if (p.parseType(ty))
      return p.emitError(p.getCurrentLocation(), "Expected !rtl.array type");
    if (elemType && elemType != ty.getElementType())
      return p.emitError(p.getCurrentLocation(), "Expected array element type ")
             << elemType;

    elemType = ty.getElementType();
    inputTypes.push_back(ty);
    resultSize += ty.getSize();
  } while (!p.parseOptionalComma());

  resultType = ArrayType::get(elemType, resultSize);
  return success();
}

static void printArrayConcatTypes(OpAsmPrinter &p, Operation *,
                                  TypeRange inputTypes, Type resultType) {
  llvm::interleaveComma(inputTypes, p, [&p](Type t) { p << t; });
}

void ArrayConcatOp::build(OpBuilder &b, OperationState &state,
                          ArrayRef<Value> values) {
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
// StructCreateOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructCreateOp(OpAsmParser &parser,
                                       OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::OperandType, 4> operands;
  StructType declType;

  if (parser.parseLParen() || parser.parseOperandList(operands) ||
      parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();

  llvm::SmallVector<Type, 4> structInnerTypes;
  declType.getInnerTypes(structInnerTypes);
  result.addTypes(declType);

  if (parser.resolveOperands(operands, structInnerTypes, inputOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &printer, rtl::StructCreateOp op) {
  printer << op.getOperationName() << " (";
  printer.printOperands(op.input());
  printer << ")";
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.getType();
}

//===----------------------------------------------------------------------===//
// StructExplodeOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructExplodeOp(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::OperandType operand;
  StructType declType;

  if (parser.parseOperand(operand) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();

  llvm::SmallVector<Type, 4> structInnerTypes;
  declType.getInnerTypes(structInnerTypes);
  result.addTypes(structInnerTypes);

  if (parser.resolveOperand(operand, declType, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &printer, rtl::StructExplodeOp op) {
  printer << op.getOperationName() << " ";
  printer.printOperand(op.input());
  printer.printOptionalAttrDict(op.getAttrs());
  printer << " : " << op.input().getType();
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructExtractOp(OpAsmParser &parser,
                                        OperationState &result) {
  OpAsmParser::OperandType operand;
  StringAttr fieldName;
  StructType declType;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(fieldName, "field", result.attributes) ||
      parser.parseRSquare() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();

  Type resultType = declType.getFieldType(fieldName.getValue());
  if (!resultType) {
    parser.emitError(parser.getNameLoc(), "invalid field name specified");
    return failure();
  }
  result.addTypes(resultType);

  if (parser.resolveOperand(operand, declType, result.operands))
    return failure();
  return success();
}

static void print(OpAsmPrinter &printer, rtl::StructExtractOp op) {
  printer << op.getOperationName() << " ";
  printer.printOperand(op.input());
  printer << "[\"" << op.field() << "\"]";
  printer.printOptionalAttrDict(op.getAttrs(), {"field"});
  printer << " : " << op.input().getType();
}

void StructExtractOp::build(::mlir::OpBuilder &odsBuilder,
                            ::mlir::OperationState &odsState, Value input,
                            StructType::FieldInfo field) {
  build(odsBuilder, odsState, field.type, input, field.name);
}

//===----------------------------------------------------------------------===//
// StructInjectOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructInjectOp(OpAsmParser &parser,
                                       OperationState &result) {
  llvm::SMLoc inputOperandsLoc = parser.getCurrentLocation();
  OpAsmParser::OperandType operand, val;
  StringAttr fieldName;
  StructType declType;

  if (parser.parseOperand(operand) || parser.parseLSquare() ||
      parser.parseAttribute(fieldName, "field", result.attributes) ||
      parser.parseRSquare() || parser.parseComma() ||
      parser.parseOperand(val) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(declType))
    return failure();

  Type resultType = declType.getFieldType(fieldName.getValue());
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

static void print(OpAsmPrinter &printer, rtl::StructInjectOp op) {
  printer << op.getOperationName() << " ";
  printer.printOperand(op.input());
  printer << "[\"" << op.field() << "\"], ";
  printer.printOperand(op.newValue());
  printer.printOptionalAttrDict(op.getAttrs(), {"field"});
  printer << " : " << op.input().getType();
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

void ArrayGetOp::build(OpBuilder &builder, OperationState &result, Value input,
                       Value index) {
  auto resultType = input.getType().cast<ArrayType>().getElementType();
  build(builder, result, resultType, input, index);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/RTL/RTL.cpp.inc"
