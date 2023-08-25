//===- IbisOps.cpp - Implementation of Ibis dialect ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Support/ParsingUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace circt;
using namespace ibis;

template <typename TSymAttr>
ParseResult parseScopeRefFromName(OpAsmParser &parser, Type &scopeRefType,
                                  TSymAttr sym) {
  // Nothing to parse, since this is already encoded in the child symbol.
  scopeRefType = ScopeRefType::get(parser.getContext(), sym);
  return success();
}

template <typename TSymAttr>
void printScopeRefFromName(OpAsmPrinter &p, Operation *op, Type type,
                           TSymAttr sym) {
  // Nothing to print since this information is already encoded in the child
  // symbol.
}

//===----------------------------------------------------------------------===//
// ScopeOpInterface
//===----------------------------------------------------------------------===//

FailureOr<mlir::TypedValue<ScopeRefType>>
circt::ibis::detail::getThisFromScope(Operation *op) {
  auto scopeOp = cast<ScopeOpInterface>(op);
  auto thisOps = scopeOp.getBodyBlock()->getOps<ibis::ThisOp>();
  if (thisOps.empty())
    return op->emitOpError("must contain a 'ibis.this' operation");

  if (std::next(thisOps.begin()) != thisOps.end())
    return op->emitOpError("must contain only one 'ibis.this' operation");

  return {*thisOps.begin()};
}

LogicalResult circt::ibis::detail::verifyScopeOpInterface(Operation *op) {
  if (failed(getThisFromScope(op)))
    return failure();

  if (!isa<SymbolOpInterface>(op))
    return op->emitOpError("must implement 'SymbolOpInterface'");

  return success();
}

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

ParseResult MethodOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Parse the function signature.
  SmallVector<OpAsmParser::Argument, 4> args;
  SmallVector<Attribute> argNames;
  SmallVector<Type> resultTypes;
  TypeAttr functionType;

  using namespace mlir::function_interface_impl;
  auto *context = parser.getContext();

  // Parse the argument list.
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/false))
    return failure();

  // Parse the result type.
  if (succeeded(parser.parseOptionalArrow())) {
    Type resultType;
    if (parser.parseType(resultType))
      return failure();
    resultTypes.push_back(resultType);
  }

  // Process the ssa args for the information we're looking for.
  SmallVector<Type> argTypes;
  for (auto &arg : args) {
    argNames.push_back(parsing_util::getNameFromSSA(context, arg.ssaName.name));
    argTypes.push_back(arg.type);
    if (!arg.sourceLoc)
      arg.sourceLoc = parser.getEncodedSourceLoc(arg.ssaName.location);
  }

  functionType =
      TypeAttr::get(FunctionType::get(context, argTypes, resultTypes));

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  result.addAttribute("argNames", ArrayAttr::get(context, argNames));
  result.addAttribute(MethodOp::getFunctionTypeAttrName(result.name),
                      functionType);

  // Parse the function body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();

  ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void MethodOp::print(OpAsmPrinter &p) {
  FunctionType funcTy = getFunctionType();
  p << ' ';
  p.printSymbolName(getSymName());
  function_interface_impl::printFunctionSignature(
      p, *this, funcTy.getInputs(), /*isVariadic=*/false, funcTy.getResults());
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     getAttributeNames());
  Region &body = getBody();
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

void MethodOp::getAsmBlockArgumentNames(mlir::Region &region,
                                        OpAsmSetValueNameFn setNameFn) {
  if (region.empty())
    return;

  auto func = cast<MethodOp>(region.getParentOp());
  auto argNames = func.getArgNames().getAsRange<StringAttr>();
  auto *block = &region.front();

  for (auto [idx, argName] : llvm::enumerate(argNames))
    if (!argName.getValue().empty())
      setNameFn(block->getArgument(idx), argName);
}

LogicalResult MethodOp::verify() {
  // Check that we have only one return value.
  if (getFunctionType().getNumResults() > 1)
    return failure();
  return success();
}

void ReturnOp::build(OpBuilder &odsBuilder, OperationState &odsState) {}

LogicalResult ReturnOp::verify() {
  // Check that the return operand type matches the function return type.
  auto func = cast<MethodOp>((*this)->getParentOp());
  ArrayRef<Type> resTypes = func.getResultTypes();
  assert(resTypes.size() <= 1);
  assert(getNumOperands() <= 1);

  if (resTypes.empty()) {
    if (getNumOperands() != 0)
      return emitOpError(
          "cannot return a value from a function with no result type");
    return success();
  }

  Value retValue = getRetValue();
  if (!retValue)
    return emitOpError("must return a value");

  Type retType = retValue.getType();
  if (retType != resTypes.front())
    return emitOpError("return type (")
           << retType << ") must match function return type ("
           << resTypes.front() << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto targetClass = getClass(&symbolTable.getSymbolTable(
      getOperation()->getParentOfType<mlir::ModuleOp>()));
  if (!targetClass)
    return emitOpError() << "'" << getTargetName() << "' does not exist";

  return success();
}

ClassOp InstanceOp::getClass(SymbolTable *symbolTable) {
  auto mod = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (symbolTable)
    return dyn_cast<ClassOp>(
        symbolTable->lookupSymbolIn(mod, getTargetNameAttr()));

  return mod.lookupSymbol<ClassOp>(getTargetNameAttr());
}

//===----------------------------------------------------------------------===//
// GetPortOp
//===----------------------------------------------------------------------===//

LogicalResult GetPortOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Lookup the target module type of the instance class reference.
  ModuleOp mod = getOperation()->getParentOfType<ModuleOp>();
  ScopeRefType crt = getInstance().getType().cast<ScopeRefType>();
  // @teqdruid TODO: make this more efficient using
  // innersymtablecollection when that's available to non-firrtl dialects.
  ScopeOpInterface targetScope =
      symbolTable.lookupSymbolIn<ScopeOpInterface>(mod, crt.getScopeRef());
  assert(targetScope && "should have been verified by the type system");
  // @teqdruid TODO: make this more efficient using
  // innersymtablecollection when that's available to non-firrtl dialects.
  Operation *targetOp = targetScope.lookupPort(getPortSymbol());

  if (!targetOp)
    return emitOpError() << "port '" << getPortSymbolAttr()
                         << "' does not exist in "
                         << targetScope.getScopeName();

  auto portOp = dyn_cast<PortOpInterface>(targetOp);
  if (!portOp)
    return emitOpError() << "symbol '" << getPortSymbolAttr()
                         << "' does not refer to a port";

  Type targetPortType = portOp.getPortType();
  Type thisPortType = getType().getPortType();
  if (targetPortType != thisPortType)
    return emitOpError() << "symbol '" << getPortSymbolAttr()
                         << "' refers to a port of type " << targetPortType
                         << ", but this op has type " << thisPortType;

  return success();
}

LogicalResult GetPortOp::canonicalize(GetPortOp op, PatternRewriter &rewriter) {
  // Canonicalize away get_port on %this in favor of using the port SSA value
  // directly.
  // get_port(%this, @P) -> ibis.port.#
  auto parentScope = cast<ScopeOpInterface>(op->getParentOp());
  auto scopeThis = parentScope.getThis();

  if (op.getInstance() == scopeThis) {
    auto definingPort = parentScope.lookupPort(op.getPortSymbol());
    rewriter.replaceOp(op, {definingPort.getPort()});
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ThisOp
//===----------------------------------------------------------------------===//

LogicalResult ThisOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // A thisOp should always refer to the parent operation, which in turn should
  // be an Ibis ScopeOpInterface.
  auto parentScope =
      dyn_cast_or_null<ScopeOpInterface>(getOperation()->getParentOp());
  if (!parentScope)
    return emitOpError() << "thisOp must be nested in a scope op";

  if (parentScope.getScopeName() != getScopeName())
    return emitOpError() << "thisOp refers to a parent scope of name "
                         << getScopeName() << ", but the parent scope is named "
                         << parentScope.getScopeName();

  return success();
}

//===----------------------------------------------------------------------===//
// ContainerInstanceOp
//===----------------------------------------------------------------------===//

ContainerOp ContainerInstanceOp::getContainer(SymbolTable *symbolTable) {
  auto mod = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (symbolTable)
    return dyn_cast<ContainerOp>(
        symbolTable->lookupSymbolIn(mod, getTargetNameAttr()));

  return mod.lookupSymbol<ContainerOp>(getTargetNameAttr());
}

LogicalResult
ContainerInstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto targetContainer = getContainer(&symbolTable.getSymbolTable(
      getOperation()->getParentOfType<mlir::ModuleOp>()));
  if (!targetContainer)
    return emitOpError() << "'" << getTargetName() << "' does not exist";

  return success();
}

//===----------------------------------------------------------------------===//
// PathOp
//===----------------------------------------------------------------------===//

/// Infer the return types of this operation.
LogicalResult PathOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {
  auto path = attrs.get("path").cast<ArrayAttr>();
  if (path.empty())
    return failure();

  auto lastStep = path.getValue().back().cast<PathStepAttr>();
  results.push_back(lastStep.getType());
  return success();
}

//===----------------------------------------------------------------------===//
// PathOp
//===----------------------------------------------------------------------===//

LogicalResult PathStepAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   PathDirection direction, mlir::Type type,
                                   mlir::FlatSymbolRefAttr instance) {
  // 'parent' should never have an instance name specified.
  if (direction == PathDirection::Parent && instance)
    return emitError() << "ibis.step 'parent' may not specify an instance name";

  if (direction == PathDirection::Child && !instance)
    return emitError() << "ibis.step 'child' must specify an instance name";

  // Only allow scoperefs
  auto scoperefType = type.dyn_cast<ScopeRefType>();
  if (!scoperefType)
    return emitError() << "ibis.step type must be an !ibis.scoperef type";

  return success();
}

LogicalResult PathOp::verify() {
  auto pathRange = getPathAsRange();
  if (pathRange.empty())
    return emitOpError() << "ibis.path must have at least one step";

  // Verify that each referenced child symbol actually exists at the module
  // level.
  auto mod = getOperation()->getParentOfType<mlir::ModuleOp>();
  for (PathStepAttr step : getPathAsRange()) {
    auto scoperefType = step.getType().cast<ScopeRefType>();
    FlatSymbolRefAttr scopeRefSym = scoperefType.getScopeRef();
    if (!scopeRefSym)
      continue;

    auto *targetScope = mod.lookupSymbol(scopeRefSym);
    if (!targetScope)
      return emitOpError() << "ibis.step scoperef symbol '" << scopeRefSym
                           << "' does not exist";
  }

  // Verify that the last step is fully typed.
  PathStepAttr lastStep = *std::prev(getPathAsRange().end());
  ScopeRefType lastStepType = lastStep.getType().cast<ScopeRefType>();
  if (!lastStepType.getScopeRef())
    return emitOpError()
           << "last ibis.step in path must specify a symbol for the scoperef";

  return success();
}

LogicalResult PathOp::canonicalize(PathOp op, PatternRewriter &rewriter) {
  // Canonicalize away ibis.path [ibis.child] to just referencing the instance
  // in the current scope.
  auto range = op.getPathAsRange();
  size_t pathSize = std::distance(range.begin(), range.end());
  PathStepAttr firstStep = *range.begin();
  if (pathSize == 1 && firstStep.getDirection() == PathDirection::Child) {
    auto parentScope = cast<ScopeOpInterface>(op->getParentOp());
    Operation *childInstance =
        SymbolTable::lookupSymbolIn(parentScope, firstStep.getChild());
    assert(childInstance && "should have been verified by the op verifier");
    rewriter.replaceOp(op,
                       {cast<ContainerInstanceOp>(childInstance).getResult()});
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisInterfaces.cpp.inc"

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Ibis/Ibis.cpp.inc"
