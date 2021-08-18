//===- FSMOps.cpp - Implement the FSM operations --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FSM/FSMOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace circt;
using namespace fsm;

//===----------------------------------------------------------------------===//
// MachineOp
//===----------------------------------------------------------------------===//

void MachineOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                      Type stateType, FunctionType type,
                      ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute("stateType", TypeAttr::get(stateType));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_like_impl::addArgAndResultAttrs(builder, state, argAttrs,
                                           /*resultAttrs=*/llvm::None);
}

/// Get the default state of the machine.
StateOp MachineOp::getDefaultState() { return *getOps<StateOp>().begin(); }

/// Get the port information of the machine.
void MachineOp::getHWPortInfo(
    SmallVectorImpl<circt::hw::ModulePortInfo> &ports) {
  ports.clear();
  auto machineType = getType();
  auto builder = Builder(*this);

  for (unsigned i = 0, e = machineType.getNumInputs(); i < e; ++i) {
    circt::hw::ModulePortInfo port;
    port.name = builder.getStringAttr("in" + std::to_string(i));
    port.direction = circt::hw::PortDirection::INPUT;
    port.type = machineType.getInput(i);
    port.argNum = i;
    ports.push_back(port);
  }

  for (unsigned i = 0, e = machineType.getNumResults(); i < e; ++i) {
    circt::hw::ModulePortInfo port;
    port.name = builder.getStringAttr("out" + std::to_string(i));
    port.direction = circt::hw::PortDirection::OUTPUT;
    port.type = machineType.getResult(i);
    port.argNum = i;
    ports.push_back(port);
  }
}

static ParseResult parseMachineOp(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes,
                          ArrayRef<Type> results,
                          function_like_impl::VariadicFlag, std::string &) {
    return builder.getFunctionType(argTypes, results);
  };

  return function_like_impl::parseFunctionLikeOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

static void print(MachineOp op, OpAsmPrinter &p) {
  FunctionType fnType = op.getType();
  function_like_impl::printFunctionLikeOp(
      p, op, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

static LogicalResult verify(MachineOp op) {
  // If this function is external there is nothing to do.
  if (op.isExternal())
    return success();

  if (!op.stateType().isa<IntegerType>())
    return op.emitOpError("state must be integer type");

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  auto fnInputTypes = op.getType().getInputs();
  Block &entryBlock = op.front();
  for (unsigned i = 0, e = entryBlock.getNumArguments(); i != e; ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "machine signature(" << fnInputTypes[i] << ')';

  // Verify that the machine only has one block terminated with OutputOp. The
  // SingleBlockImplicitTerminator trait has duplicated definitions of
  // the `front()` method with the FunctionLike interface.
  if (!llvm::hasSingleElement(op))
    return op.emitOpError("must only have a single block");

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

/// Lookup the machine for the symbol.  This returns null on invalid IR.
MachineOp InstanceOp::getReferencedMachine() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;

  auto referencedMachine = topLevelModuleOp.lookupSymbol(machine());
  if (!referencedMachine)
    return nullptr;

  return dyn_cast<MachineOp>(referencedMachine);
}

static LogicalResult verifyInstanceOp(InstanceOp op) {
  auto referencedMachine = op.getReferencedMachine();
  if (referencedMachine == nullptr)
    return op.emitError("Cannot find machine definition '")
           << op.machine() << "'";

  return success();
}

void InstanceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(instance(), sym_name());
}

//===----------------------------------------------------------------------===//
// TriggerOp
//===----------------------------------------------------------------------===//

template <typename OpType>
static LogicalResult verifyCallerTypes(OpType op) {
  auto referencedMachine = op.getReferencedMachine();

  // Check operand types first.
  auto numOperands = op.inputs().size();
  auto expectedOperandTypes = referencedMachine.getType().getInputs();

  if (expectedOperandTypes.size() != numOperands) {
    auto diag = op.emitOpError()
                << "has a wrong number of operands; expected "
                << expectedOperandTypes.size() << " but got " << numOperands;
    diag.attachNote(referencedMachine->getLoc())
        << "original machine declared here";

    return failure();
  }

  for (size_t i = 0; i != numOperands; ++i) {
    auto expectedType = expectedOperandTypes[i];
    auto operandType = op.inputs()[i].getType();
    if (operandType != expectedType) {
      auto diag = op.emitOpError()
                  << "#" << i << " operand type must be " << expectedType
                  << ", but got " << operandType;

      diag.attachNote(referencedMachine->getLoc())
          << "original machine declared here";
      return failure();
    }
  }

  // Check result types.
  auto numResults = op->getNumResults();
  auto expectedResultTypes = referencedMachine.getType().getResults();

  if (expectedResultTypes.size() != numResults) {
    auto diag = op.emitOpError()
                << "has a wrong number of results; expected "
                << expectedResultTypes.size() << " but got " << numResults;
    diag.attachNote(referencedMachine->getLoc())
        << "original machine declared here";

    return failure();
  }

  for (size_t i = 0; i != numResults; ++i) {
    auto expectedType = expectedResultTypes[i];
    auto resultType = op.getResult(i).getType();
    if (resultType != expectedType) {
      auto diag = op.emitOpError()
                  << "#" << i << " result type must be " << expectedType
                  << ", but got " << resultType;

      diag.attachNote(referencedMachine->getLoc())
          << "original machine declared here";
      return failure();
    }
  }

  return success();
}

/// Lookup the machine for the symbol.  This returns null on invalid IR.
MachineOp TriggerOp::getReferencedMachine() {
  auto instanceOp = dyn_cast<InstanceOp>(instance().getDefiningOp());
  if (!instanceOp)
    return nullptr;

  return instanceOp.getReferencedMachine();
}

static LogicalResult verifyTriggerOp(TriggerOp op) {
  return verifyCallerTypes(op);
}

//===----------------------------------------------------------------------===//
// HWInstanceOp
//===----------------------------------------------------------------------===//

/// Lookup the machine for the symbol.  This returns null on invalid IR.
MachineOp HWInstanceOp::getReferencedMachine() {
  auto topLevelModuleOp = (*this)->getParentOfType<ModuleOp>();
  if (!topLevelModuleOp)
    return nullptr;

  auto referencedMachine = topLevelModuleOp.lookupSymbol(machine());
  if (!referencedMachine)
    return nullptr;

  return dyn_cast<MachineOp>(referencedMachine);
}

static LogicalResult verifyHWInstanceOp(HWInstanceOp op) {
  return verifyCallerTypes(op);
}

//===----------------------------------------------------------------------===//
// StateOp
//===----------------------------------------------------------------------===//

LogicalResult StateOp::canonicalize(StateOp op, PatternRewriter &rewriter) {
  bool hasAlwaysTakenTransition = false;
  SmallVector<TransitionOp, 4> transitionsToErase;
  // Remove all transitions after an "always-taken" transition.
  for (auto transition : op.transitions().getOps<TransitionOp>()) {
    if (!hasAlwaysTakenTransition)
      hasAlwaysTakenTransition = transition.isAlwaysTaken();
    else
      transitionsToErase.push_back(transition);
  }

  for (auto transition : transitionsToErase)
    rewriter.eraseOp(transition);

  return failure(transitionsToErase.empty());
}

static LogicalResult verifyStateOp(StateOp op) {
  if (op.transitions().front().getTerminator()->getNumOperands() != 0)
    return op.emitOpError("transitions region should not return any value");

  return success();
}

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyOutputOp(OutputOp op) {
  if (op->getParentRegion() != &op->getParentOfType<StateOp>().output())
    return success();

  auto machine = op->getParentOfType<MachineOp>();

  // Verify that the result list of the machine and the operand list of the
  // OutputOp line up.
  if (op.getNumOperands() != machine.getNumResults())
    return op.emitOpError("the number of operands doesn't match with the "
                          "result number of the machine");

  auto outputTypes = machine.getType().getResults();
  for (unsigned i = 0, e = machine.getNumResults(); i != e; ++i)
    if (outputTypes[i] != op.getOperand(i).getType())
      return op.emitOpError("type of the terminator #")
             << i << '(' << op.getOperand(i).getType()
             << ") must match the type of the corresponding result in machine "
             << "signature(" << outputTypes[i] << ')';

  return success();
}

//===----------------------------------------------------------------------===//
// TransitionOp
//===----------------------------------------------------------------------===//

/// Lookup the next state for the symbol. This returns null on invalid IR.
StateOp TransitionOp::getReferencedNextState() {
  auto machineOp = (*this)->getParentOfType<MachineOp>();
  if (!machineOp)
    return nullptr;

  auto referencedNextState = machineOp.lookupSymbol(nextState());
  if (!referencedNextState)
    return nullptr;

  return dyn_cast<StateOp>(referencedNextState);
}

bool TransitionOp::isAlwaysTaken() {
  auto guardReturn = getGuardReturn();
  if (guardReturn.getNumOperands() == 0)
    return true;
  else if (auto definingOp = guardReturn.getOperand(0).getDefiningOp()) {
    if (auto constantOp = dyn_cast<mlir::ConstantOp>(definingOp)) {
      return constantOp.value().cast<BoolAttr>().getValue();
    }
  }
  return false;
}

LogicalResult TransitionOp::canonicalize(TransitionOp op,
                                         PatternRewriter &rewriter) {
  auto guardReturn = op.getGuardReturn();
  if (guardReturn.getNumOperands() == 1) {
    if (auto definingOp = guardReturn.getOperand(0).getDefiningOp()) {
      // Simplify the transition op when the guard region returns a constant
      // value.
      if (auto constantOp = dyn_cast<mlir::ConstantOp>(definingOp)) {
        if (constantOp.value().cast<BoolAttr>().getValue()) {
          // Replace the original return op with a new one without any operands
          // if the constant is TRUE.
          rewriter.setInsertionPoint(guardReturn);
          rewriter.create<fsm::ReturnOp>(guardReturn.getLoc());
          rewriter.eraseOp(guardReturn);
        } else {
          // Erase the whole transition op if the constant is FALSE, because the
          // transition will never be taken.
          rewriter.eraseOp(op);
        }
        return success();
      }
    }
  }
  return failure();
}

static LogicalResult verifyTransitionOp(TransitionOp op) {
  if (!op.getReferencedNextState())
    return op.emitOpError("cannot find the definition of the next state `")
           << op.nextState() << "`";

  // Verify the action and guard region.
  if (op.action().front().getTerminator()->getNumOperands() != 0)
    return op.emitOpError("action region should not return any value");

  auto guardReturn = op.getGuardReturn();
  if (guardReturn.getNumOperands() > 1)
    return op.emitOpError("guard region should only return zero or one result");

  auto isBoolType = [&](Type type) {
    if (auto intTy = type.dyn_cast<IntegerType>())
      return intTy.getWidth() == 1 && intTy.isSignless();
    return false;
  };

  if (guardReturn.getNumOperands() == 1)
    if (!isBoolType(guardReturn.getOperandTypes().front()))
      return op.emitOpError("guard region should return bool result");

  // Verify the transition is located in the correct region.
  if (op->getParentRegion() != &op.getCurrentState().transitions())
    return op.emitOpError("should only be located in the transitions region");

  return success();
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

void VariableOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(result(), name());
}

//===----------------------------------------------------------------------===//
// UpdateOp
//===----------------------------------------------------------------------===//

/// Get the targeted variable operation. This returns null on invalid IR.
VariableOp UpdateOp::getDstVariable() {
  auto dstDefOp = dst().getDefiningOp();
  if (!dstDefOp)
    return nullptr;

  return dyn_cast<VariableOp>(dstDefOp);
}

static LogicalResult verifyUpdateOp(UpdateOp op) {
  if (!op.getDstVariable())
    return op.emitOpError("destination is not a variable operation");

  if (op->getParentRegion() != &op->getParentOfType<TransitionOp>().action())
    return op.emitOpError("should only be located in the action region");

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/FSM/FSM.cpp.inc"
#undef GET_OP_CLASSES

#include "circt/Dialect/FSM/FSMDialect.cpp.inc"
