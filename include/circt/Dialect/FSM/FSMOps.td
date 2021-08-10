//===- FSMOps.td - FSM dialect operations definition file -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FSM_FSMOPS_TD
#define CIRCT_DIALECT_FSM_FSMOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def MachineOp : FSMOp<"machine",
      [IsolatedFromAbove, FunctionLike, Symbol, SymbolTable,
       HasParent<"mlir::ModuleOp">]> {
  let summary = "Define a finite-state machine";
  let description = [{
    `fsm.machine` represents a finite-state machine, including a machine name
    and the types of inputs and outputs. This op also includes a `$body` region
    that contains the variables and states.
  }];

  let arguments = (ins StrAttr:$sym_name, TypeAttr:$stateType,
                       TypeAttr:$type);
  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "Type":$stateType, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

  let extraClassDeclaration = [{
    StateOp getDefaultState();
    void getHWPortInfo(SmallVectorImpl<circt::hw::ModulePortInfo> &);

  private:
    /// This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<MachineOp>;

    /// Returns the number of arguments. This is a hook for
    /// OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type. Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];

  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  let verifier = [{ return ::verify(*this); }];
}

def InstanceOp : FSMOp<"instance",
      [Symbol, DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "Create an instance of a machine";
  let description = [{
    `fsm.instance` represents an instance of a machine, including an instance
    name and a symbol reference of the machine.
  }];

  let arguments = (ins StrAttr:$sym_name, FlatSymbolRefAttr:$machine);
  let results = (outs InstanceType:$instance);

  let extraClassDeclaration = [{
    /// Lookup the machine for the symbol. This returns null on invalid IR.
    MachineOp getReferencedMachine();
  }];

  let assemblyFormat = [{ $sym_name $machine attr-dict }];

  let verifier = "return ::verify$cppClass(*this);";
}

def TriggerOp : FSMOp<"trigger", []> {
  let summary = "Trigger an instance";
  let description = [{
    `fsm.trigger` triggers a machine instance. The inputs and outputs are
    correponding to the inputs and outputs of the instance.
  }];

  let arguments = (ins InstanceType:$instance, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);

  let extraClassDeclaration = [{
    /// Lookup the machine for the symbol. This returns null on invalid IR.
    MachineOp getReferencedMachine();
  }];

  let assemblyFormat = [{
    $instance attr-dict `(` $inputs `)` `:` functional-type($inputs, $outputs)
  }];

  let verifier = "return ::verify$cppClass(*this);";
}

def HWInstanceOp : FSMOp<"hw_instance", [Symbol]> {
  let summary = "Create a hardware-style instance of a machine";
  let description = [{
    `fsm.hw_instance` represents a hardware-style instance of a machine,
    including an instance name and a symbol reference of the machine. The inputs
    and outputs are correponding to the inputs and outputs of the referenced
    machine.
  }];

  let arguments = (ins StrAttr:$sym_name, FlatSymbolRefAttr:$machine,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);

  let extraClassDeclaration = [{
    /// Lookup the machine for the symbol. This returns null on invalid IR.
    MachineOp getReferencedMachine();
  }];

  let assemblyFormat = [{
    $sym_name $machine attr-dict `(` $inputs `)` `:`
    functional-type($inputs, $outputs)
  }];

  let verifier = "return ::verify$cppClass(*this);";
}

def StateOp : FSMOp<"state",
      [Symbol, HasParent<"MachineOp">,
       SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "Represent a state of a machine";
  let description = [{
    `fsm.state` represents a state of a machine. This op includes an `entry`
    region and an `exit` region that represent the actions to be executed when
    entering and exiting the state. This op also includes a `transitions` region
    that contains all the transitions of the state.
  }];

  let arguments = (ins StrAttr:$sym_name);
  let regions = (region SizedRegion<1>:$entry, SizedRegion<1>:$exit,
                        SizedRegion<1>:$transitions);

  let hasCanonicalizeMethod = true;

  let assemblyFormat = [{
    $sym_name attr-dict `entry` $entry `exit` $exit `transitions` $transitions
  }];

  let verifier = "return ::verify$cppClass(*this);";
}

def TransitionOp : FSMOp<"transition",
      [HasParent<"StateOp">, SingleBlockImplicitTerminator<"ReturnOp">]> {
  let summary = "Represent a transition of a state";
  let description = [{
    `fsm.transition` represents a transition of a state, including a symbol
    reference of the next state. This op also includes a `$guard` region that
    represents the guard condition and an `$action` region that represents the
    actions to be executed when this transition is triggered.
  }];

  let arguments = (ins FlatSymbolRefAttr:$nextState);
  let regions = (region SizedRegion<1>:$guard, SizedRegion<1>:$action);

  let extraClassDeclaration = [{
    /// Lookup the next state for the symbol. This returns null on invalid IR.
    StateOp getReferencedNextState();

    /// Get the current state, this should never fail.
    StateOp getCurrentState() {
      return (*this)->getParentOfType<StateOp>();
    }

    /// Get the return operation of the guard region, this should never fail.
    ReturnOp getGuardReturn() {
      return cast<ReturnOp>(guard().front().getTerminator());
    }

    bool isAlwaysTaken();
  }];

  let hasCanonicalizeMethod = true;

  let assemblyFormat = [{
    $nextState attr-dict `guard` $guard `action` $action
  }];

  let verifier = "return ::verify$cppClass(*this);";
}

def VariableOp : FSMOp<"variable", 
      [HasParent<"MachineOp">, FirstAttrDerivedResultType,
       DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "Declare a variable in a machine";
  let description = [{
    `fsm.variable` represents a variable in a machine.
  }];

  let arguments = (ins AnyAttr:$initValue, StrAttr:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{ $name attr-dict `:` type($result) }];
}

def UpdateOp : FSMOp<"update", [SameTypeOperands]> {
  let summary = "Update a variable in a machine";
  let description = [{
    `fsm.update` updates the value of `$dst` with `$src`. The definition op of
    `$dst` should be an `fsm.variable`.
  }];

  let arguments = (ins AnyType:$dst, AnyType:$src);

  let extraClassDeclaration = [{
    /// Get the targeted variable operation. This returns null on invalid IR.
    VariableOp getDstVariable();
  }];

  let assemblyFormat = [{
    attr-dict $dst `,` $src `:` type($dst)
  }];

  let verifier = "return ::verify$cppClass(*this);";
}

def OutputOp : FSMOp<"output",
      [HasParent<"MachineOp">, ReturnLike, Terminator]> {
  let summary = "Output values from a machine";
  let description = [{
    "fsm.output" outputs the value of variables in a machine. The types of
    `$operands` should be consistant with the output types of the machine.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [ OpBuilder<(ins), "build($_builder, $_state, llvm::None);"> ];

  let assemblyFormat = [{ attr-dict ($operands^ `:` type($operands))? }];
}

def ReturnOp : FSMOp<"return",
      [ParentOneOf<["StateOp", "TransitionOp"]>, ReturnLike, Terminator]> {
  let summary = "Return values from a region";
  let description = [{
    "fsm.return" marks the end of a region of `fsm.state` or `fsm.transition`
    and return values.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [ OpBuilder<(ins), "build($_builder, $_state, llvm::None);"> ];

  let assemblyFormat = [{ attr-dict ($operands^ `:` type($operands))? }];
}

#endif // CIRCT_DIALECT_FSM_FSMOPS_TD
