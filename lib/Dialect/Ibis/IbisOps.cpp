//===- IbisOps.cpp - Implementation of Ibis dialect ops -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/DC/DCTypes.h"
#include "circt/Support/ParsingUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace ibis;

// Looks up a `sym`-symbol defining operation of type T in the `mlir::ModuleOp`
// parent scope of the provided `base` operation.
template <typename T>
static T lookupInModule(Operation *base, FlatSymbolRefAttr sym,
                        const SymbolTable *symbolTable) {
  auto mod = base->getParentOfType<mlir::ModuleOp>();
  if (symbolTable)
    return dyn_cast<T>(symbolTable->lookupSymbolIn(mod, sym));

  return mod.lookupSymbol<T>(sym);
}

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

// Generates a name for Ibis values.
// NOLINTBEGIN(misc-no-recursion)
static llvm::raw_string_ostream &genValueName(llvm::raw_string_ostream &os,
                                              Value value) {
  auto *definingOp = value.getDefiningOp();
  assert(definingOp && "scoperef should always be defined by some op");
  llvm::TypeSwitch<Operation *, void>(definingOp)
      .Case<ThisOp>([&](auto op) { os << "this"; })
      .Case<InstanceOp, ContainerInstanceOp>(
          [&](auto op) { os << op.getInstanceNameAttr().strref(); })
      .Case<PortOpInterface>([&](auto op) { os << op.getPortName().strref(); })
      .Case<PathOp>([&](auto op) {
        llvm::interleave(
            op.getPathAsRange(), os,
            [&](PathStepAttr step) {
              if (step.getDirection() == PathDirection::Parent)
                os << "parent";
              else
                os << step.getChild().getAttr().strref();
            },
            ".");
      })
      .Case<GetPortOp>([&](auto op) {
        genValueName(os, op.getInstance())
            << "." << op.getPortSymbol() << ".ref";
      })
      .Case<PortReadOp>(
          [&](auto op) { genValueName(os, op.getPort()) << ".val"; })
      .Default([&](auto op) {
        op->emitOpError() << "unhandled value type";
        assert(false && "unhandled value type");
      });
  return os;
}
// NOLINTEND(misc-no-recursion)

// Generates a name for Ibis values, and returns a StringAttr.
static StringAttr genValueNameAttr(Value v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  genValueName(os, v);
  return StringAttr::get(v.getContext(), s);
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

template <typename TOp>
ParseResult parseMethodLikeOp(OpAsmParser &parser, OperationState &result) {
  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr))
    return failure();

  result.attributes.append(hw::InnerSymbolTable::getInnerSymbolAttrName(),
                           hw::InnerSymAttr::get(nameAttr));

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

  // Parse the result types
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();

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
  result.addAttribute(TOp::getFunctionTypeAttrName(result.name), functionType);

  // Parse the function body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();

  return success();
}

template <typename TOp>
void printMethodLikeOp(TOp op, OpAsmPrinter &p) {
  FunctionType funcTy = op.getFunctionType();
  p << ' ';
  p.printSymbolName(op.getInnerSym().getSymName());
  Region &body = op.getBody();
  p << "(";
  llvm::interleaveComma(body.getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ") ";
  p.printArrowTypeList(funcTy.getResults());
  p.printOptionalAttrDictWithKeyword(op.getOperation()->getAttrs(),
                                     op.getAttributeNames());
  if (!body.empty()) {
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false,
                  /*printBlockTerminators=*/true);
  }
}

ParseResult MethodOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseMethodLikeOp<MethodOp>(parser, result);
}

void MethodOp::print(OpAsmPrinter &p) { return printMethodLikeOp(*this, p); }

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

//===----------------------------------------------------------------------===//
// DataflowMethodOp
//===----------------------------------------------------------------------===//

ParseResult DataflowMethodOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  return parseMethodLikeOp<DataflowMethodOp>(parser, result);
}

void DataflowMethodOp::print(OpAsmPrinter &p) {
  return printMethodLikeOp(*this, p);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

void ReturnOp::build(OpBuilder &odsBuilder, OperationState &odsState) {}

LogicalResult ReturnOp::verify() {
  // Check that the return operand type matches the function return type.
  auto methodLike = cast<MethodLikeOpInterface>((*this)->getParentOp());
  ArrayRef<Type> resTypes = methodLike.getResultTypes();

  if (getNumOperands() != resTypes.size())
    return emitOpError(
        "must have the same number of operands as the method has results");

  for (auto [arg, resType] : llvm::zip(getOperands(), resTypes))
    if (arg.getType() != resType)
      return emitOpError("operand type (")
             << arg.getType() << ") must match function return type ("
             << resType << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// GetVarOp
//===----------------------------------------------------------------------===//

LogicalResult GetVarOp::verifyInnerRefs(hw::InnerRefNamespace &ns) {
  ScopeRefType parentType = getInstance().getType().cast<ScopeRefType>();
  auto varOp = ns.lookupOp<VarOp>(hw::InnerRefAttr::get(
      parentType.getScopeRef().getAttr(), getVarNameAttr().getAttr()));

  if (!varOp)
    return failure();

  // Ensure that the dereferenced type is the same type as the variable type.
  if (varOp.getType() != getType())
    return emitOpError() << "dereferenced type (" << getType()
                         << ") must match variable type (" << varOp.getType()
                         << ")";

  return success();
}

FailureOr<VarOp> GetVarOp::getTarget(SymbolTable *symbolTable) {
  auto targetClassSym =
      getInstance().getType().cast<ScopeRefType>().getScopeRef();
  auto targetClass =
      lookupInModule<ClassOp>(getOperation(), targetClassSym, symbolTable);

  if (!targetClass)
    return emitOpError() << "'" << targetClassSym << "' does not exist";

  // Lookup the variable inside the class scope.
  auto varName = getVarName();
  // @teqdruid TODO: make this more efficient using
  // innersymtablecollection when that's available to non-firrtl dialects.
  auto var = dyn_cast_or_null<VarOp>(
      symbolTable->lookupSymbolIn(targetClass.getOperation(), varName));
  if (!var)
    return emitOpError() << "'" << varName << "' does not exist in '"
                         << targetClassSym << "'";
  return {var};
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

ClassOp InstanceOp::getClass(const SymbolTable *symbolTable) {
  return lookupInModule<ClassOp>(getOperation(), getTargetNameAttr(),
                                 symbolTable);
}

void InstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), genValueNameAttr(getResult()));
}

//===----------------------------------------------------------------------===//
// GetPortOp
//===----------------------------------------------------------------------===//

LogicalResult GetPortOp::verifyInnerRefs(hw::InnerRefNamespace &ns) {
  // Lookup the target module type of the instance class reference.
  ScopeRefType crt = getInstance().getType().cast<ScopeRefType>();
  Operation *targetOp = ns.lookupOp(hw::InnerRefAttr::get(
      crt.getScopeRef().getAttr(), getPortSymbolAttr().getAttr()));

  if (!targetOp)
    return emitOpError() << "port '" << getPortSymbolAttr()
                         << "' does not exist in " << crt.getScopeRef();

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
  auto parentScope = dyn_cast<ScopeOpInterface>(op->getParentOp());
  if (parentScope) {
    auto scopeThis = parentScope.getThis();
    if (op.getInstance() == scopeThis) {
      auto definingPort = parentScope.lookupPort(op.getPortSymbol());
      rewriter.replaceOp(op, {definingPort.getPort()});
      return success();
    }
  }

  return failure();
}

void GetPortOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), genValueNameAttr(getResult()));
}

//===----------------------------------------------------------------------===//
// ThisOp
//===----------------------------------------------------------------------===//

LogicalResult ThisOp::verifyInnerRefs(hw::InnerRefNamespace &ns) {
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

void ThisOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "this");
}

//===----------------------------------------------------------------------===//
// PortReadOp
//===----------------------------------------------------------------------===//

void PortReadOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), genValueNameAttr(getResult()));
}

//===----------------------------------------------------------------------===//
// ContainerInstanceOp
//===----------------------------------------------------------------------===//

ContainerOpInterface
ContainerInstanceOp::getContainer(const SymbolTable *symbolTable) {
  auto mod = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (symbolTable)
    return dyn_cast_or_null<ContainerOpInterface>(
        symbolTable->lookupSymbolIn(mod, getTargetNameAttr()));

  return mod.lookupSymbol<ContainerOpInterface>(getTargetNameAttr());
}

LogicalResult
ContainerInstanceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto targetContainer = getContainer(&symbolTable.getSymbolTable(
      getOperation()->getParentOfType<mlir::ModuleOp>()));
  if (!targetContainer)
    return emitOpError() << "'" << getTargetName() << "' does not exist";

  return success();
}

void ContainerInstanceOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), genValueNameAttr(getResult()));
}

std::optional<size_t> InnerContainerOp::getTargetResultIndex() {
  return std::nullopt;
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
    auto childInstance = dyn_cast_or_null<ContainerInstanceOp>(
        parentScope.lookupInnerSym(firstStep.getChild().getValue()));
    assert(childInstance && "should have been verified by the op verifier");
    rewriter.replaceOp(op, {childInstance.getResult()});
    return success();
  }

  return failure();
}

void PathOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), genValueNameAttr(getResult()));
}

//===----------------------------------------------------------------------===//
// OutputPortOp
//===----------------------------------------------------------------------===//

LogicalResult OutputPortOp::canonicalize(OutputPortOp op,
                                         PatternRewriter &rewriter) {
  // Replace any reads of an output port op that is written to from the same
  // scope, with the value that is written to it.
  PortWriteOp writer;
  llvm::SmallVector<PortReadOp, 4> readers;
  for (auto *user : op.getResult().getUsers()) {
    if (auto read = dyn_cast<PortReadOp>(user)) {
      readers.push_back(read);
    } else if (auto write = dyn_cast<PortWriteOp>(user);
               write && write.getPort() == op.getPort()) {
      assert(!writer && "should only have one writer");
      writer = write;
    }
  }

  if (!readers.empty()) {
    for (auto reader : readers)
      rewriter.replaceOp(reader, writer.getValue());
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// InputWireOp
//===----------------------------------------------------------------------===//

LogicalResult InputWireOp::canonicalize(InputWireOp op,
                                        PatternRewriter &rewriter) {
  // Canonicalize away wires which are assigned within this scope.
  auto portUsers = op.getPort().getUsers();
  size_t nPortUsers = std::distance(portUsers.begin(), portUsers.end());
  for (auto *portUser : op.getPort().getUsers()) {
    auto writer = dyn_cast<PortWriteOp>(portUser);
    if (writer && writer.getPort() == op.getPort() && nPortUsers == 1) {
      rewriter.replaceAllUsesWith(op.getOutput(), writer.getValue());
      rewriter.eraseOp(writer);
      rewriter.eraseOp(op);
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// OutputWireOp
//===----------------------------------------------------------------------===//

LogicalResult OutputWireOp::canonicalize(OutputWireOp op,
                                         PatternRewriter &rewriter) {
  // Canonicalize away wires which are read (and nothing else) within this
  // scope. Assume that duplicate reads have been CSE'd away and just look
  // for a single reader.
  auto portUsers = op.getPort().getUsers();
  size_t nPortUsers = std::distance(portUsers.begin(), portUsers.end());
  for (auto *portUser : op.getPort().getUsers()) {
    auto reader = dyn_cast<PortReadOp>(portUser);
    if (reader && reader.getPort() == op.getPort() && nPortUsers == 1) {
      rewriter.replaceOp(reader, op.getInput());
      rewriter.eraseOp(op);
      return success();
    }
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// StaticBlockOp
//===----------------------------------------------------------------------===//

template <typename TOp>
static ParseResult parseBlockLikeOp(
    OpAsmParser &parser, OperationState &result,
    llvm::function_ref<ParseResult(OpAsmParser::Argument &)> argAdjuster = {}) {
  // Parse the argument initializer list.
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> inputOperands;
  llvm::SmallVector<OpAsmParser::Argument> inputArguments;
  llvm::SmallVector<Type> inputTypes;
  ArrayAttr inputNames;
  if (parsing_util::parseInitializerList(parser, inputArguments, inputOperands,
                                         inputTypes, inputNames))
    return failure();

  // Parse the result types.
  llvm::SmallVector<Type> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  result.addTypes(resultTypes);

  // Parse the attribute dict.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  // All operands have been parsed - resolve.
  if (parser.resolveOperands(inputOperands, inputTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  // If the user provided an arg adjuster, apply it to each argument.
  if (argAdjuster) {
    for (auto &arg : inputArguments)
      if (failed(argAdjuster(arg)))
        return failure();
  }

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, inputArguments))
    return failure();

  TOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

template <typename T>
static void printBlockLikeOp(T op, OpAsmPrinter &p) {
  p << ' ';
  parsing_util::printInitializerList(p, op.getInputs(),
                                     op.getBodyBlock()->getArguments());
  p.printOptionalArrowTypeList(op.getResultTypes());
  p.printOptionalAttrDictWithKeyword(op.getOperation()->getAttrs());
  p << ' ';
  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult StaticBlockOp::verify() {
  if (getInputs().size() != getBodyBlock()->getNumArguments())
    return emitOpError("number of inputs must match number of block arguments");

  for (auto [arg, barg] :
       llvm::zip(getInputs(), getBodyBlock()->getArguments())) {
    if (arg.getType() != barg.getType())
      return emitOpError("block argument type must match input type");
  }

  return success();
}

ParseResult StaticBlockOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBlockLikeOp<StaticBlockOp>(parser, result);
}

void StaticBlockOp::print(OpAsmPrinter &p) {
  return printBlockLikeOp(*this, p);
}

//===----------------------------------------------------------------------===//
// IsolatedStaticBlockOp
//===----------------------------------------------------------------------===//

LogicalResult IsolatedStaticBlockOp::verify() {
  if (getInputs().size() != getBodyBlock()->getNumArguments())
    return emitOpError("number of inputs must match number of block arguments");

  for (auto [arg, barg] :
       llvm::zip(getInputs(), getBodyBlock()->getArguments())) {
    if (arg.getType() != barg.getType())
      return emitOpError("block argument type must match input type");
  }

  return success();
}

ParseResult IsolatedStaticBlockOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  return parseBlockLikeOp<IsolatedStaticBlockOp>(parser, result);
}

void IsolatedStaticBlockOp::print(OpAsmPrinter &p) {
  return printBlockLikeOp(*this, p);
}

//===----------------------------------------------------------------------===//
// DCBlockOp
//===----------------------------------------------------------------------===//

void DCBlockOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                      TypeRange outputs, ValueRange inputs,
                      IntegerAttr maxThreads) {
  odsState.addOperands(inputs);
  if (maxThreads)
    odsState.addAttribute(getMaxThreadsAttrName(odsState.name), maxThreads);
  auto *region = odsState.addRegion();
  llvm::SmallVector<Type> resTypes;
  for (auto output : outputs) {
    dc::ValueType dcType = output.dyn_cast<dc::ValueType>();
    assert(dcType && "DCBlockOp outputs must be dc::ValueType");
    resTypes.push_back(dcType);
  }
  odsState.addTypes(resTypes);
  ensureTerminator(*region, odsBuilder, odsState.location);
  llvm::SmallVector<Location> argLocs;
  llvm::SmallVector<Type> argTypes;
  for (auto input : inputs) {
    argLocs.push_back(input.getLoc());
    dc::ValueType dcType = input.getType().dyn_cast<dc::ValueType>();
    assert(dcType && "DCBlockOp inputs must be dc::ValueType");
    argTypes.push_back(dcType.getInnerType());
  }
  region->front().addArguments(argTypes, argLocs);
}

LogicalResult DCBlockOp::verify() {
  if (getInputs().size() != getBodyBlock()->getNumArguments())
    return emitOpError("number of inputs must match number of block arguments");

  for (auto [arg, barg] :
       llvm::zip(getInputs(), getBodyBlock()->getArguments())) {
    dc::ValueType dcType = arg.getType().dyn_cast<dc::ValueType>();
    if (!dcType)
      return emitOpError("DCBlockOp inputs must be dc::ValueType but got ")
             << arg.getType();

    if (dcType.getInnerType() != barg.getType())
      return emitOpError("block argument type must match input type. Got ")
             << barg.getType() << " expected " << dcType.getInnerType();
  }

  return success();
}

ParseResult DCBlockOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseBlockLikeOp<DCBlockOp>(
      parser, result, [&](OpAsmParser::Argument &arg) -> LogicalResult {
        dc::ValueType valueType = arg.type.dyn_cast<dc::ValueType>();
        if (!valueType)
          return parser.emitError(parser.getCurrentLocation(),
                                  "DCBlockOp inputs must be dc::ValueType");
        arg.type = valueType.getInnerType();
        return success();
      });
}

void DCBlockOp::print(OpAsmPrinter &p) { return printBlockLikeOp(*this, p); }

//===----------------------------------------------------------------------===//
// BlockReturnOp
//===----------------------------------------------------------------------===//

LogicalResult BlockReturnOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  auto parentBlock = dyn_cast<BlockOpInterface>(parent);
  if (!parentBlock)
    return emitOpError("must be nested in a block");

  if (getNumOperands() != parent->getNumResults())
    return emitOpError("number of operands must match number of block outputs");

  for (auto [op, out] :
       llvm::zip(getOperands(), parentBlock.getInternalResultTypes())) {
    if (op.getType() != out)
      return emitOpError(
                 "operand type must match parent block output type. Expected ")
             << out << " got " << op.getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// InlineStaticBlockEndOp
//===----------------------------------------------------------------------===//

InlineStaticBlockBeginOp InlineStaticBlockEndOp::getBeginOp() {
  auto curr = getOperation()->getReverseIterator();
  Operation *firstOp = &getOperation()->getBlock()->front();
  while (true) {
    if (auto beginOp = dyn_cast<InlineStaticBlockBeginOp>(*curr))
      return beginOp;
    if (curr.getNodePtr() == firstOp)
      break;
    ++curr;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// InlineStaticBlockBeginOp
//===----------------------------------------------------------------------===//

InlineStaticBlockEndOp InlineStaticBlockBeginOp::getEndOp() {
  auto curr = getOperation()->getIterator();
  auto end = getOperation()->getBlock()->end();
  while (curr != end) {
    if (auto endOp = dyn_cast<InlineStaticBlockEndOp>(*curr))
      return endOp;

    ++curr;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// TableGen generated logic
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisInterfaces.cpp.inc"

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/Ibis/Ibis.cpp.inc"
