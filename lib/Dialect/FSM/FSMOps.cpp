//===- FSMOps.cpp - Implementation of FSM dialect operations --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FSM/FSMOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  function_interface_impl::addArgAndResultAttrs(builder, state, argAttrs,
                                                /*resultAttrs=*/llvm::None);
}

/// Get the default state of the machine.
StateOp MachineOp::getDefaultState() { return *getOps<StateOp>().begin(); }

/// Get the port information of the machine.
void MachineOp::getHWPortInfo(SmallVectorImpl<hw::PortInfo> &ports) {
  ports.clear();
  auto machineType = getFunctionType();
  auto builder = Builder(*this);

  for (unsigned i = 0, e = machineType.getNumInputs(); i < e; ++i) {
    hw::PortInfo port;
    port.name = builder.getStringAttr("in" + std::to_string(i));
    port.direction = circt::hw::PortDirection::INPUT;
    port.type = machineType.getInput(i);
    port.argNum = i;
    ports.push_back(port);
  }

  for (unsigned i = 0, e = machineType.getNumResults(); i < e; ++i) {
    hw::PortInfo port;
    port.name = builder.getStringAttr("out" + std::to_string(i));
    port.direction = circt::hw::PortDirection::OUTPUT;
    port.type = machineType.getResult(i);
    port.argNum = i;
    ports.push_back(port);
  }
}

ParseResult MachineOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, buildFuncType);
}

void MachineOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false);
}

static LogicalResult compareTypes(TypeRange rangeA, TypeRange rangeB) {
  if (rangeA.size() != rangeB.size())
    return failure();

  int64_t index = 0;
  for (auto zip : llvm::zip(rangeA, rangeB)) {
    if (std::get<0>(zip) != std::get<1>(zip))
      return failure();
    ++index;
  }

  return success();
}

LogicalResult MachineOp::verify() {
  // If this function is external there is nothing to do.
  if (isExternal())
    return success();

  if (!stateType().isa<IntegerType>())
    return emitOpError("state must be integer type");

  // Verify that the argument list of the function and the arg list of the entry
  // block line up.  The trait already verified that the number of arguments is
  // the same between the signature and the block.
  if (failed(compareTypes(getArgumentTypes(), front().getArgumentTypes())))
    return emitOpError(
        "entry block argument types must match the machine input types");

  // Verify that the machine only has one block terminated with OutputOp.
  if (!llvm::hasSingleElement(*this))
    return emitOpError("must only have a single block");

  return success();
}

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

/// Lookup the machine for the symbol.  This returns null on invalid IR.
MachineOp InstanceOp::getMachine() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  return module.lookupSymbol<MachineOp>(machine());
}

LogicalResult InstanceOp::verify() {
  auto m = getMachine();
  if (!m)
    return emitError("cannot find machine definition '") << machine() << "'";

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
  auto machine = op.getMachine();
  if (!machine)
    return op.emitError("cannot find machine definition");

  // Check operand types first.
  if (failed(
          compareTypes(machine.getArgumentTypes(), op.inputs().getTypes()))) {
    auto diag =
        op.emitOpError("operand types must match the machine input types");
    diag.attachNote(machine->getLoc()) << "original machine declared here";
    return failure();
  }

  // Check result types.
  if (failed(compareTypes(machine.getResultTypes(), op.outputs().getTypes()))) {
    auto diag =
        op.emitOpError("result types must match the machine output types");
    diag.attachNote(machine->getLoc()) << "original machine declared here";
    return failure();
  }

  return success();
}

/// Lookup the machine for the symbol.  This returns null on invalid IR.
MachineOp TriggerOp::getMachine() {
  auto instanceOp = instance().getDefiningOp<InstanceOp>();
  if (!instanceOp)
    return nullptr;

  return instanceOp.getMachine();
}

LogicalResult TriggerOp::verify() { return verifyCallerTypes(*this); }

//===----------------------------------------------------------------------===//
// HWInstanceOp
//===----------------------------------------------------------------------===//

/// Lookup the machine for the symbol.  This returns null on invalid IR.
MachineOp HWInstanceOp::getMachine() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  return module.lookupSymbol<MachineOp>(machine());
}

LogicalResult HWInstanceOp::verify() { return verifyCallerTypes(*this); }

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

//===----------------------------------------------------------------------===//
// OutputOp
//===----------------------------------------------------------------------===//

LogicalResult OutputOp::verify() {
  if ((*this)->getParentRegion() ==
      &(*this)->getParentOfType<StateOp>().transitions()) {
    if (getNumOperands() != 0)
      emitOpError("transitions region must not output any value");
    return success();
  }

  // Verify that the result list of the machine and the operand list of the
  // OutputOp line up.
  auto machine = (*this)->getParentOfType<MachineOp>();
  if (failed(compareTypes(machine.getResultTypes(), getOperandTypes())))
    return emitOpError("operand types must match the machine output types");

  return success();
}

//===----------------------------------------------------------------------===//
// TransitionOp
//===----------------------------------------------------------------------===//

/// Lookup the next state for the symbol. This returns null on invalid IR.
StateOp TransitionOp::getNextState() {
  auto machineOp = (*this)->getParentOfType<MachineOp>();
  if (!machineOp)
    return nullptr;

  return machineOp.lookupSymbol<StateOp>(nextState());
}

bool TransitionOp::isAlwaysTaken() {
  auto guardReturn = getGuardReturn();
  if (guardReturn.getNumOperands() == 0)
    return true;

  if (auto constantOp =
          guardReturn.getOperand(0).getDefiningOp<mlir::arith::ConstantOp>())
    return constantOp.getValue().cast<BoolAttr>().getValue();

  return false;
}

LogicalResult TransitionOp::canonicalize(TransitionOp op,
                                         PatternRewriter &rewriter) {
  auto guardReturn = op.getGuardReturn();
  if (guardReturn.getNumOperands() == 1)
    if (auto constantOp = guardReturn.getOperand(0)
                              .getDefiningOp<mlir::arith::ConstantOp>()) {
      // Simplify when the guard region returns a constant value.
      if (constantOp.getValue().cast<BoolAttr>().getValue()) {
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

  return failure();
}

LogicalResult TransitionOp::verify() {
  if (!getNextState())
    return emitOpError("cannot find the definition of the next state `")
           << nextState() << "`";

  // Verify the action region.
  if (action().front().getTerminator()->getNumOperands() != 0)
    return emitOpError("action region must not return any value");

  // Verify the transition is located in the correct region.
  if ((*this)->getParentRegion() != &getCurrentState().transitions())
    return emitOpError("must only be located in the transitions region");

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
VariableOp UpdateOp::getVariable() {
  return variable().getDefiningOp<VariableOp>();
}

LogicalResult UpdateOp::verify() {
  if (!getVariable())
    return emitOpError("destination is not a variable operation");

  if (!(*this)->getParentOfType<TransitionOp>().action().isAncestor(
          (*this)->getParentRegion()))
    return emitOpError("must only be located in the action region");

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
