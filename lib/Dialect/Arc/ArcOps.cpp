//===- ArcOps.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace arc;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static LogicalResult verifyTypeListEquivalence(Operation *op,
                                               TypeRange expectedTypeList,
                                               TypeRange actualTypeList,
                                               StringRef elementName) {
  if (expectedTypeList.size() != actualTypeList.size())
    return op->emitOpError("incorrect number of ")
           << elementName << "s: expected " << expectedTypeList.size()
           << ", but got " << actualTypeList.size();

  for (unsigned i = 0, e = expectedTypeList.size(); i != e; ++i) {
    if (expectedTypeList[i] != actualTypeList[i]) {
      auto diag = op->emitOpError(elementName)
                  << " type mismatch: " << elementName << " #" << i;
      diag.attachNote() << "expected type: " << expectedTypeList[i];
      diag.attachNote() << "  actual type: " << actualTypeList[i];
      return diag;
    }
  }

  return success();
}

static LogicalResult verifyArcSymbolUse(Operation *op, TypeRange inputs,
                                        TypeRange results,
                                        SymbolTableCollection &symbolTable) {
  // Check that the arc attribute was specified.
  auto arcName = op->getAttrOfType<FlatSymbolRefAttr>("arc");
  // The arc attribute is verified by the tablegen generated verifier as it is
  // an ODS defined attribute.
  assert(arcName && "FlatSymbolRefAttr called 'arc' missing");
  DefineOp arc = symbolTable.lookupNearestSymbolFrom<DefineOp>(op, arcName);
  if (!arc)
    return op->emitOpError() << "`" << arcName.getValue()
                             << "` does not reference a valid `arc.define`";

  // Verify that the operand and result types match the arc.
  auto type = arc.getFunctionType();
  if (failed(
          verifyTypeListEquivalence(op, type.getInputs(), inputs, "operand")))
    return failure();

  if (failed(
          verifyTypeListEquivalence(op, type.getResults(), results, "result")))
    return failure();

  return success();
}

static bool isSupportedModuleOp(Operation *moduleOp) {
  return llvm::isa<arc::ModelOp, hw::HWModuleLike>(moduleOp);
}

/// Fetches the operation pointed to by `pointing` with name `symbol`, checking
/// that it is a supported model operation for simulation.
static Operation *getSupportedModuleOp(SymbolTableCollection &symbolTable,
                                       Operation *pointing, StringAttr symbol) {
  Operation *moduleOp = symbolTable.lookupNearestSymbolFrom(pointing, symbol);
  if (!moduleOp) {
    pointing->emitOpError("model not found");
    return nullptr;
  }

  if (!isSupportedModuleOp(moduleOp)) {
    pointing->emitOpError("model symbol does not point to a supported model "
                          "operation, points to ")
        << moduleOp->getName() << " instead";
    return nullptr;
  }

  return moduleOp;
}

static std::optional<hw::ModulePort> getModulePort(Operation *moduleOp,
                                                   StringRef portName) {
  auto findRightPort = [&](auto ports) -> std::optional<hw::ModulePort> {
    const hw::ModulePort *port = llvm::find_if(
        ports, [&](hw::ModulePort port) { return port.name == portName; });
    if (port == ports.end())
      return std::nullopt;
    return *port;
  };

  return TypeSwitch<Operation *, std::optional<hw::ModulePort>>(moduleOp)
      .Case<arc::ModelOp>(
          [&](arc::ModelOp modelOp) -> std::optional<hw::ModulePort> {
            return findRightPort(modelOp.getIo().getPorts());
          })
      .Case<hw::HWModuleLike>(
          [&](hw::HWModuleLike moduleLike) -> std::optional<hw::ModulePort> {
            return findRightPort(moduleLike.getPortList());
          })
      .Default([](Operation *) { return std::nullopt; });
}

//===----------------------------------------------------------------------===//
// DefineOp
//===----------------------------------------------------------------------===//

ParseResult DefineOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void DefineOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, "function_type", getArgAttrsAttrName(),
      getResAttrsAttrName());
}

LogicalResult DefineOp::verifyRegions() {
  // Check that the body does not contain any side-effecting operations. We can
  // simply iterate over the ops directly within the body; operations with
  // regions, like scf::IfOp, implement the `HasRecursiveMemoryEffects` trait
  // which causes the `isMemoryEffectFree` check to already recur into their
  // regions.
  for (auto &op : getBodyBlock()) {
    if (isMemoryEffectFree(&op))
      continue;

    // We don't use a op-error here because that leads to the whole arc being
    // printed. This can be switched of when creating the context, but one
    // might not want to switch that off for other error messages. Here it's
    // definitely not desirable as arcs can be very big and would fill up the
    // error log, making it hard to read. Currently, only the signature (first
    // line) of the arc is printed.
    auto diag = mlir::emitError(getLoc(), "body contains non-pure operation");
    diag.attachNote(op.getLoc()).append("first non-pure operation here: ");
    return diag;
  }
  return success();
}

bool DefineOp::isPassthrough() {
  if (getNumArguments() != getNumResults())
    return false;

  return llvm::all_of(
      llvm::zip(getArguments(), getBodyBlock().getTerminator()->getOperands()),
      [](const auto &argAndRes) {
        return std::get<0>(argAndRes) == std::get<1>(argAndRes);
      });
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify() {
  auto *parent = (*this)->getParentOp();
  TypeRange expectedTypes = parent->getResultTypes();
  if (auto defOp = dyn_cast<DefineOp>(parent))
    expectedTypes = defOp.getResultTypes();

  TypeRange actualTypes = getOperands().getTypes();
  return verifyTypeListEquivalence(*this, expectedTypes, actualTypes, "output");
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

LogicalResult StateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyArcSymbolUse(*this, getInputs().getTypes(),
                            getResults().getTypes(), symbolTable);
}

LogicalResult StateOp::verify() {
  if (getLatency() < 1)
    return emitOpError("latency must be a positive integer");

  if (!getOperation()->getParentOfType<ClockDomainOp>() && !getClock())
    return emitOpError("outside a clock domain requires a clock");

  if (getOperation()->getParentOfType<ClockDomainOp>() && getClock())
    return emitOpError("inside a clock domain cannot have a clock");

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyArcSymbolUse(*this, getInputs().getTypes(),
                            getResults().getTypes(), symbolTable);
}

bool CallOp::isClocked() { return false; }

Value CallOp::getClock() { return Value{}; }

void CallOp::eraseClock() {}

uint32_t CallOp::getLatency() { return 0; }

//===----------------------------------------------------------------------===//
// MemoryWritePortOp
//===----------------------------------------------------------------------===//

SmallVector<Type> MemoryWritePortOp::getArcResultTypes() {
  auto memType = cast<MemoryType>(getMemory().getType());
  SmallVector<Type> resultTypes{memType.getAddressType(),
                                memType.getWordType()};
  if (getEnable())
    resultTypes.push_back(IntegerType::get(getContext(), 1));
  if (getMask())
    resultTypes.push_back(memType.getWordType());
  return resultTypes;
}

LogicalResult
MemoryWritePortOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyArcSymbolUse(*this, getInputs().getTypes(), getArcResultTypes(),
                            symbolTable);
}

LogicalResult MemoryWritePortOp::verify() {
  if (getLatency() < 1)
    return emitOpError("latency must be at least 1");

  if (!getOperation()->getParentOfType<ClockDomainOp>() && !getClock())
    return emitOpError("outside a clock domain requires a clock");

  if (getOperation()->getParentOfType<ClockDomainOp>() && getClock())
    return emitOpError("inside a clock domain cannot have a clock");

  return success();
}

//===----------------------------------------------------------------------===//
// ClockDomainOp
//===----------------------------------------------------------------------===//

LogicalResult ClockDomainOp::verifyRegions() {
  return verifyTypeListEquivalence(*this, getBodyBlock().getArgumentTypes(),
                                   getInputs().getTypes(), "input");
}

//===----------------------------------------------------------------------===//
// RootInputOp
//===----------------------------------------------------------------------===//

void RootInputOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buf("in_");
  buf += getName();
  setNameFn(getState(), buf);
}

//===----------------------------------------------------------------------===//
// RootOutputOp
//===----------------------------------------------------------------------===//

void RootOutputOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buf("out_");
  buf += getName();
  setNameFn(getState(), buf);
}

//===----------------------------------------------------------------------===//
// ModelOp
//===----------------------------------------------------------------------===//

LogicalResult ModelOp::verify() {
  if (getBodyBlock().getArguments().size() != 1)
    return emitOpError("must have exactly one argument");
  if (auto type = getBodyBlock().getArgument(0).getType();
      !isa<StorageType>(type))
    return emitOpError("argument must be of storage type");
  for (const hw::ModulePort &port : getIo().getPorts())
    if (port.dir == hw::ModulePort::Direction::InOut)
      return emitOpError("inout ports are not supported");
  return success();
}

LogicalResult ModelOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto fnAttrs = std::array{getInitialFnAttr(), getFinalFnAttr()};
  auto nouns = std::array{"initializer", "finalizer"};
  for (auto [fnAttr, noun] : llvm::zip(fnAttrs, nouns)) {
    if (!fnAttr)
      continue;
    auto fn = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
    if (!fn)
      return emitOpError() << noun << " '" << fnAttr.getValue()
                           << "' does not reference a valid function";
    if (!llvm::equal(fn.getArgumentTypes(), getBody().getArgumentTypes())) {
      auto diag = emitError() << noun << " '" << fnAttr.getValue()
                              << "' arguments must match arguments of model";
      diag.attachNote(fn.getLoc()) << noun << " declared here:";
      return diag;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LutOp
//===----------------------------------------------------------------------===//

LogicalResult LutOp::verify() {
  Location firstSideEffectOpLoc = UnknownLoc::get(getContext());
  const WalkResult result = getBody().walk([&](Operation *op) {
    if (auto memOp = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
      memOp.getEffects(effects);

      if (!effects.empty()) {
        firstSideEffectOpLoc = memOp->getLoc();
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return emitOpError("no operations with side-effects allowed inside a LUT")
               .attachNote(firstSideEffectOpLoc)
           << "first operation with side-effects here";

  return success();
}

//===----------------------------------------------------------------------===//
// VectorizeOp
//===----------------------------------------------------------------------===//

LogicalResult VectorizeOp::verify() {
  if (getInputs().empty())
    return emitOpError("there has to be at least one input vector");

  if (!llvm::all_equal(llvm::map_range(
          getInputs(), [](OperandRange range) { return range.size(); })))
    return emitOpError("all input vectors must have the same size");

  for (OperandRange range : getInputs()) {
    if (!llvm::all_equal(range.getTypes()))
      return emitOpError("all input vector lane types must match");

    if (range.empty())
      return emitOpError("input vector must have at least one element");
  }

  if (getResults().empty())
    return emitOpError("must have at least one result");

  if (!llvm::all_equal(getResults().getTypes()))
    return emitOpError("all result types must match");

  if (getResults().size() != getInputs().front().size())
    return emitOpError("number results must match input vector size");

  return success();
}

static FailureOr<unsigned> getVectorWidth(Type base, Type vectorized) {
  if (isa<VectorType>(base))
    return failure();

  if (auto vectorTy = dyn_cast<VectorType>(vectorized)) {
    if (vectorTy.getElementType() != base)
      return failure();

    return vectorTy.getDimSize(0);
  }

  if (vectorized.getIntOrFloatBitWidth() < base.getIntOrFloatBitWidth())
    return failure();

  if (vectorized.getIntOrFloatBitWidth() % base.getIntOrFloatBitWidth() == 0)
    return vectorized.getIntOrFloatBitWidth() / base.getIntOrFloatBitWidth();

  return failure();
}

LogicalResult VectorizeOp::verifyRegions() {
  auto returnOp = cast<VectorizeReturnOp>(getBody().front().getTerminator());
  TypeRange bodyArgTypes = getBody().front().getArgumentTypes();

  if (bodyArgTypes.size() != getInputs().size())
    return emitOpError(
        "number of block arguments must match number of input vectors");

  // Boundary and body are vectorized, or both are not vectorized
  if (returnOp.getValue().getType() == getResultTypes().front()) {
    for (auto [i, argTy] : llvm::enumerate(bodyArgTypes))
      if (argTy != getInputs()[i].getTypes().front())
        return emitOpError("if terminator type matches result type the "
                           "argument types must match the input types");

    return success();
  }

  // Boundary is vectorized, body is not
  if (auto width = getVectorWidth(returnOp.getValue().getType(),
                                  getResultTypes().front());
      succeeded(width)) {
    for (auto [i, argTy] : llvm::enumerate(bodyArgTypes)) {
      Type inputTy = getInputs()[i].getTypes().front();
      FailureOr<unsigned> argWidth = getVectorWidth(argTy, inputTy);
      if (failed(argWidth))
        return emitOpError("block argument must be a scalar variant of the "
                           "vectorized operand");

      if (*argWidth != width)
        return emitOpError("input and output vector width must match");
    }

    return success();
  }

  // Body is vectorized, boundary is not
  if (auto width = getVectorWidth(getResultTypes().front(),
                                  returnOp.getValue().getType());
      succeeded(width)) {
    for (auto [i, argTy] : llvm::enumerate(bodyArgTypes)) {
      Type inputTy = getInputs()[i].getTypes().front();
      FailureOr<unsigned> argWidth = getVectorWidth(inputTy, argTy);
      if (failed(argWidth))
        return emitOpError(
            "block argument must be a vectorized variant of the operand");

      if (*argWidth != width)
        return emitOpError("input and output vector width must match");

      if (getInputs()[i].size() > 1 && argWidth != getInputs()[i].size())
        return emitOpError(
            "when boundary not vectorized the number of vector element "
            "operands must match the width of the vectorized body");
    }

    return success();
  }

  return returnOp.emitOpError(
      "operand type must match parent op's result value or be a vectorized or "
      "non-vectorized variant of it");
}

bool VectorizeOp::isBoundaryVectorized() {
  return getInputs().front().size() == 1;
}
bool VectorizeOp::isBodyVectorized() {
  auto returnOp = cast<VectorizeReturnOp>(getBody().front().getTerminator());
  if (isBoundaryVectorized() &&
      returnOp.getValue().getType() == getResultTypes().front())
    return true;

  if (auto width = getVectorWidth(getResultTypes().front(),
                                  returnOp.getValue().getType());
      succeeded(width))
    return *width > 1;

  return false;
}

//===----------------------------------------------------------------------===//
// SimInstantiateOp
//===----------------------------------------------------------------------===//

void SimInstantiateOp::print(OpAsmPrinter &p) {
  BlockArgument modelArg = getBody().getArgument(0);
  auto modelType = cast<SimModelInstanceType>(modelArg.getType());

  p << " " << modelType.getModel() << " as ";
  p.printRegionArgument(modelArg, {}, true);

  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());

  p << " ";

  p.printRegion(getBody(), false);
}

ParseResult SimInstantiateOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  StringAttr modelName;
  if (failed(parser.parseSymbolName(modelName)))
    return failure();

  if (failed(parser.parseKeyword("as")))
    return failure();

  OpAsmParser::Argument modelArg;
  if (failed(parser.parseArgument(modelArg, false, false)))
    return failure();

  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  MLIRContext *ctxt = result.getContext();
  modelArg.type =
      SimModelInstanceType::get(ctxt, FlatSymbolRefAttr::get(ctxt, modelName));

  std::unique_ptr<Region> body = std::make_unique<Region>();
  if (failed(parser.parseRegion(*body, {modelArg})))
    return failure();

  result.addRegion(std::move(body));
  return success();
}

LogicalResult SimInstantiateOp::verifyRegions() {
  Region &body = getBody();
  if (body.getNumArguments() != 1)
    return emitError("entry block of body region must have the model instance "
                     "as a single argument");
  if (!llvm::isa<SimModelInstanceType>(body.getArgument(0).getType()))
    return emitError("entry block argument type is not a model instance");
  return success();
}

LogicalResult
SimInstantiateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *moduleOp = getSupportedModuleOp(
      symbolTable, getOperation(),
      llvm::cast<SimModelInstanceType>(getBody().getArgument(0).getType())
          .getModel()
          .getAttr());
  if (!moduleOp)
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// SimSetInputOp
//===----------------------------------------------------------------------===//

LogicalResult
SimSetInputOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *moduleOp = getSupportedModuleOp(
      symbolTable, getOperation(),
      llvm::cast<SimModelInstanceType>(getInstance().getType())
          .getModel()
          .getAttr());
  if (!moduleOp)
    return failure();

  std::optional<hw::ModulePort> port = getModulePort(moduleOp, getInput());
  if (!port)
    return emitOpError("port not found on model");

  if (port->dir != hw::ModulePort::Direction::Input &&
      port->dir != hw::ModulePort::Direction::InOut)
    return emitOpError("port is not an input port");

  if (port->type != getValue().getType())
    return emitOpError(
               "mismatched types between value and model port, port expects ")
           << port->type;

  return success();
}

//===----------------------------------------------------------------------===//
// SimGetPortOp
//===----------------------------------------------------------------------===//

LogicalResult
SimGetPortOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *moduleOp = getSupportedModuleOp(
      symbolTable, getOperation(),
      llvm::cast<SimModelInstanceType>(getInstance().getType())
          .getModel()
          .getAttr());
  if (!moduleOp)
    return failure();

  std::optional<hw::ModulePort> port = getModulePort(moduleOp, getPort());
  if (!port)
    return emitOpError("port not found on model");

  if (port->type != getValue().getType())
    return emitOpError(
               "mismatched types between value and model port, port expects ")
           << port->type;

  return success();
}

//===----------------------------------------------------------------------===//
// SimStepOp
//===----------------------------------------------------------------------===//

LogicalResult SimStepOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  Operation *moduleOp = getSupportedModuleOp(
      symbolTable, getOperation(),
      llvm::cast<SimModelInstanceType>(getInstance().getType())
          .getModel()
          .getAttr());
  if (!moduleOp)
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ExecuteOp
//===----------------------------------------------------------------------===//

LogicalResult ExecuteOp::verifyRegions() {
  return verifyTypeListEquivalence(*this, getInputs().getTypes(),
                                   getBody().getArgumentTypes(), "input");
}

#include "circt/Dialect/Arc/ArcInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "circt/Dialect/Arc/Arc.cpp.inc"
