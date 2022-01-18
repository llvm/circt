//===- GAAOps.td - GAA Dialect Operators -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the GAA dialect operators.
//
//===----------------------------------------------------------------------===//
#ifndef CIRCT_DIALECT_GAA_GAAOPS_TD
#define CIRCT_DIALECT_GAA_GAAOPS_TD

// TODO: add this to HW interface
//       and do the failing check.
def isHWModuleSymbol : AttrConstraint<
    CPred<"!::mlir::SymbolTable::lookupNearestSymbolFrom("
            "&$_op, $_self.cast<::mlir::FlatSymbolRefAttr>()) || "
          "hw::isAnyModule(::mlir::SymbolTable::lookupNearestSymbolFrom("
            "&$_op, $_self.cast<::mlir::FlatSymbolRefAttr>()))"
    >, "is module like">;

def CircuitOp : GAAOp<
  "circuit", [
  IsolatedFromAbove,
  SingleBlock,
  Symbol,
  SymbolTable,
  NoTerminator,
  NoRegionArguments
  ]> {
  let summary = "represent a GAA circuit.";
  let description = [{
    Represent a GAA circuit. name of circuit is the top module, this generates
    one scheduler circuit.
  }];
  let regions = (region SizedRegion<1>:$body);
}

def ExtModuleOp : GAAOp<
  "module.extern", [
  HasParent<"CircuitOp">,
  IsolatedFromAbove,
  SingleBlock,
  Symbol,
  SymbolTable,
  NoTerminator,
  DeclareOpInterfaceMethods<GAAModuleLike>,
  DeclareOpInterfaceMethods<HWModuleLike>
  ]> {
  let summary = "Binding a `hw.module` to GAA ExtModule.";
  let description = [{
    Defining the primitive GAA Module which binds `hw.module` to `gaa.module`,
    In the attributes definition, it should also define the conflict matrix.
  }];
  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins Confined<FlatSymbolRefAttr, [isHWModuleSymbol]>:$extModuleName
                  // TODO: represent conflict conflictFree sequenceBefore sequenceAfter
                  );
}

def BindBareOp : GAAOp<
  "bind.bare", [
  HasParent<"ExtModuleOp">
  ]> {
  let summary = "Binding a signal to a port";
  let description = [{
    barely bare port symbol to gaa external module.
  }];
  let arguments = (ins HWIntegerType:$signal);
}

def BindValueOp : GAAOp<
  "bind.value", [
  HasParent<"ExtModuleOp">,
  DeclareOpInterfaceMethods<GAAFunctionLike>,
  DeclareOpInterfaceMethods<GAAValueLike>,
  Symbol
  ]> {
  let summary = "Binding a signal to a port";
  let description = [{
    barely bare port symbol to gaa external module.
  }];
}

def BindMethodOp : GAAOp<
  "bind.method", [
  HasParent<"ExtModuleOp">,
  DeclareOpInterfaceMethods<GAAFunctionLike>,
  DeclareOpInterfaceMethods<GAAMethodLike>,
  Symbol,
  NoTerminator,
  IsolatedFromAbove
  ]> {
  let regions = (region SizedRegion<1>:$body);
}

def ModuleOp : GAAOp<
  "module", [
  HasParent<"CircuitOp">,
  IsolatedFromAbove,
  SingleBlock,
  Symbol,
  SymbolTable,
  NoTerminator,
  DeclareOpInterfaceMethods<GAAModuleLike>,
  DeclareOpInterfaceMethods<HWModuleLike>
  ]> {
  let summary = "Defining a GAA Module.";
  let description = [{
    This operation defines the a GAA module. A GAA Module can contain rules,
    methods and values.
  }];
  let regions = (region SizedRegion<1>:$body);
}

def InstanceOp : GAAOp<
  "instance", [
  HasParent<"ModuleOp">,
  Symbol,
  DeclareOpInterfaceMethods<HWInstanceLike, ["instanceName"]>
  ]> {
  let summary = "Instantiate a GAA Module or GAA ExtModule";
  let description = [{
    It instantiates modules to construct the module hierarchy, this module can
    invoke instance module with `gaa.call`.
  }];
  // arguments which need to pass to non-scheduler modules.
  let arguments = (ins Variadic<HWIntegerType>:$operands,
                       FlatSymbolRefAttr:$moduleName
  );
}
def RuleOp : GAAOp<
  "rule", [
  HasParent<"ModuleOp">,
  Symbol,
  DeclareOpInterfaceMethods<GAARuleLike>,
  SingleBlockImplicitTerminator<"RuleReturnOp">
  ]> {
  let summary = "Define a rule";
  let description = [{
    Scheduling base element.
  }];
  let regions = (region SizedRegion<1>:$body);
}

// use DynInstDataOpInterface after migrate it into HW Dialect.
def RuleRefOp : GAAOp<
  "ruleref", [
  HasParent<"CircuitOp">,
  Symbol
  ]> {
  let summary = "Define a global reference to a rule";
  let description = [{
  }];
  let arguments = (ins StrAttr:$rule, FlatSymbolRefAttr:$ref);
}

def RuleReturnOp : GAAOp<
  "rule.return", [
  Terminator,
  ReturnLike,
  HasParent<"RuleOp">,
  ]> {
  let summary = "Return ready to scheduler.";
  let description = [{
    The terminator to `RuleOp`, ready `ready`, which will be used by scheduler
    The first argument is `ready`, indicates all `ready` signal propagated from
    downstream methods are `ready`.
  }];

  let arguments = (ins HWIntegerType:$ready, Optional<HWIntegerType>:$data);
}

def ValueOp : GAAOp<
  "value", [
  HasParent<"ModuleOp">,
  Symbol,
  DeclareOpInterfaceMethods<GAAFunctionLike>,
  DeclareOpInterfaceMethods<GAAValueLike>,
  SingleBlockImplicitTerminator<"ValueReturnOp">
  ]> {
  let summary = "Define a value, can be used in both GAA ExtModule and Module.";
  let description = [{
    Defining interface which doesn't need enable signal. It doesn't need
    `enable` signal as input argument, but it has a `ready` as output signal,
    which is the first output as well.
    In side the region of the ValueOp, it has Comb Dialect operators and
    ValueCallOp, but MethodCallOp is not allowed, since `enable` signal cannot
    bypass.
    It should be terminated by ValueReturnOp.
  }];
  let regions = (region SizedRegion<1>:$body);
}

def ValueCallOp : GAAOp<
  "value.call", [
  ParentOneOf<["RuleOp", "MethodOp", "ValueOp"]>,
  DeclareOpInterfaceMethods<GAACallLike>
  ]> {
  let summary = "Call Value.";
  let description = [{
    Call a `ValueOp` in `RuleOp`, `MethodOp` and `ValueOp`.
    Has no arguments.
    The first output is `ready` signal, and second output is data.
  }];
  let results = (outs HWIntegerType:$ready, HWIntegerType:$data);

  let builders = [
    OpBuilder<(ins "FlatSymbolRefAttr":$instanceName,
                   "FlatSymbolRefAttr":$functionName)>
  ];
}

def ValueReturnOp : GAAOp<
  "value.return", [
  Terminator,
  ReturnLike,
  HasParent<"ValueOp">,
  ]> {
  let summary = "Return signals.";
  let description = [{
    The terminator to `ValueOp`, return `ready` and `data` to `rule`, `method`
    or `value`.
    The first argument is `ready`, the second argument is the optional field
    data.
  }];

  let arguments = (ins HWIntegerType:$ready, Optional<HWIntegerType>:$data);
}

def MethodOp : GAAOp<
  "method", [
  HasParent<"ModuleOp">,
  DeclareOpInterfaceMethods<GAAFunctionLike>,
  DeclareOpInterfaceMethods<GAAMethodLike>,
  Symbol,
  SingleBlockImplicitTerminator<"MethodReturnOp">,
  FunctionOpInterface
  ]> {
  let summary = "Define a method, can be used in both GAA ExtModule and Module.";
  let description = [{
    Defining interface which has the `ready`-`enable` contract. The first
    argument is `enable` signal, auto generated by scheduler, which should be
    propagated from rule down through method call. The first output is `ready`
    signal, which should be propagated from primitive modules up to rule as
    the input to its own scheduling group.
    In side the region of the MethodOp, it has Comb Dialect operators,
    ValueCallOp and MethodCallOp.
    It should be terminated by ReturnMethodOp, with the first operator as
    `ready` signal.
  }];
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{

    // methods need to be defined by FunctionOpInterface

    using mlir::detail::FunctionOpInterfaceTrait<MethodOp>::front;
    using mlir::detail::FunctionOpInterfaceTrait<MethodOp>::getBody;

    /// Returns the type of this function.
    FunctionType getFunctionType() {
      return getFunctionTypeAttr().getValue().cast<FunctionType>();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getFunctionTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];
}

def MethodCallOp : GAAOp<
  "method.call", [
  ParentOneOf<["MethodOp", "RuleOp"]>,
  DeclareOpInterfaceMethods<GAACallLike>
  ]> {
  let summary = "Call Method.";
  let description = [{
    Call a `MethodOp` in `RuleOp`, `MethodOp`.
    The first argument is `enable`, and second argument is optional input
    `data`.
    The first output is `ready` signal, and second output is optional `data`.
  }];
  let arguments = (ins HWIntegerType:$enable, Optional<HWIntegerType>:$operands);
  let results = (outs HWIntegerType:$ready, Optional<HWIntegerType>:$data);

  let builders = [
    OpBuilder<(ins "FlatSymbolRefAttr":$instanceName,
                   "FlatSymbolRefAttr":$functionName)>
  ];
}

def MethodReturnOp : GAAOp<
  "method.return", [
  Terminator,
  ReturnLike,
  HasParent<"MethodOp">,
  ]> {
  let summary = "Return signals.";
  let description = [{
    call in a `rule`: return `ready` to scheduler.
    call in a `method`: return `ready` and optional `data` to `rule` or `method`
    call in a `value`: return `ready` and `data` to `rule`, `method` or `value`.
  }];

  let arguments = (ins HWIntegerType:$ready, Optional<HWIntegerType>:$operands);
}

#endif // CIRCT_DIALECT_GAA_GAAOPS_TD
