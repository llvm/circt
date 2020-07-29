//===- StaticLogic.td - StaticLogic Definitions ------------*- tablegen -*-===//
//
// Copyright 2020 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#ifdef STATICLOGIC_OPS
#else
#define STATICLOGIC_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def StaticLogic_Dialect : Dialect {
  let name = "staticlogic";
  let cppNamespace = "";
}

def PipelineOp : Op<StaticLogic_Dialect, "pipeline", [FunctionLike, Symbol,
  NativeOpTrait<"IsIsolatedFromAbove">, RegionKindInterface
]> {
  let summary = "pipeline operation";
  let description = [{
    The "staticlogic.pipeline" operation represents a statically scheduled 
    pipeline stucture which can contain multiple pipeline stages.
  }];

  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion: $body);

  let skipDefaultBuilders = 1;

  let builders = [OpBuilder<"OpBuilder &builder, OperationState &odsState, "
      "StringRef name, FunctionType type, " 
      "ArrayRef<NamedAttribute> attributes",
  [{
    odsState.addAttribute(SymbolTable::getSymbolAttrName(),
                          builder.getStringAttr(name));
    odsState.addAttribute(getTypeAttrName(), TypeAttr::get(type));
    odsState.addAttributes(attributes);
    for (unsigned i = 0; i != 1; ++i)
      (void)odsState.addRegion();
  }]>];

  let extraClassDeclaration = [{
    /// Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<PipelineOp>;

    /// Returns the number of arguments, implementing OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }
    /// Returns the number of results, implementing OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
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

  let printer = [{
    auto type = getAttrOfType<TypeAttr>(getTypeAttrName()).getValue()
                .cast<FunctionType>();
    impl::printFunctionLikeOp(p, *this, type.getInputs(), 
                              /*isVariadic=*/true, type.getResults());
  }];
}

def InstanceOp : Op<StaticLogic_Dialect, "instance", [CallOpInterface]> {
  let summary = "module instantiate operation";
  let description = [{
    The `staticlogic.instance` operation represents the instantiation of a 
	  StaticLogic module which is similar to a function call.
  }];

  let arguments = (ins FlatSymbolRefAttr: $module, 
                       Variadic<AnyType>: $operands);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    StringRef getModule() { return module(); }
    FunctionType getModuleType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the module of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<SymbolRefAttr>("module");
    }
  }];
}

def ReturnOp : Op<StaticLogic_Dialect, "return", [Terminator]> {
  let summary = "StaticLogic dialect return.";
  let description = [{
    The "handshake.return" operation represents a terminator of a statically 
    scheduled module, which is similar to a standard return operation.
  }];

  let arguments = (ins Variadic<AnyType>: $operands);
}

#endif // STATICLOGIC_OPS