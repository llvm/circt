//===- IbisOps.td - Definition of Ibis dialect operations -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_IBISOPS_TD
#define CIRCT_DIALECT_IBIS_IBISOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/Ibis/IbisInterfaces.td"
include "circt/Dialect/Ibis/IbisTypes.td"

class IbisOp<string mnemonic, list<Trait> traits = []> :
    Op<IbisDialect, mnemonic, traits>;

def HasCustomSSAName :
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>;

def ClassOp : IbisOp<"class", [
    IsolatedFromAbove, RegionKindInterface,
    Symbol, SymbolTable, SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    HasParent<"mlir::ModuleOp">]> {

  let summary = "Ibis class";
  let description = [{
    Ibis has the notion of a class which can contain methods and member
    variables.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    // Returns the inner 'this' argument of the class.
    BlockArgument getThis() { return getBody().front().getArgument(0); }
  }];
}

def InstanceOp : IbisOp<"instance", [
    Symbol,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Ibis class instance";
  let description = [{
    Instantiates an Ibis class.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$className);
  let results = (outs ClassRefType:$classRef);
  let assemblyFormat = [{
    $sym_name `,` $className attr-dict
    custom<ClassRefFromName>(type($classRef), ref($className))
  }];

  let extraClassDeclaration = [{
    // Return the class this instance is instantiating.
    ClassOp getClass(SymbolTableCollection *symbolTable = nullptr) {
      auto mod = getOperation()->getParentOfType<mlir::ModuleOp>();
      if(symbolTable)
        return symbolTable->lookupSymbolIn<ClassOp>(mod, getClassNameAttr());

      return mod.lookupSymbol<ClassOp>(getClassNameAttr());
    }
  }];
}

def MethodOp : IbisOp<"method", [
      IsolatedFromAbove, RegionKindInterface,
      Symbol, FunctionOpInterface,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
      SingleBlockImplicitTerminator<"ReturnOp">,
      HasParent<"ClassOp">]> {

  let summary = "Ibis method";
  let description = [{
    Ibis functions are a lot like software functions: a list of named arguments
    and one unnamed return value.

    Can only live inside of classes.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       ArrayAttr:$argNames,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph;}

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}

def ReturnOp : IbisOp<"return", [
      Pure, ReturnLike, Terminator, HasParent<"MethodOp">]> {
  let summary = "Ibis function terminator";

  let arguments = (ins Optional<AnyType>:$retValue);
  let assemblyFormat = "($retValue^)? attr-dict (`:` type($retValue)^)?";
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins)>,
  ];
}

def CallOp : IbisOp<"call", [CallOpInterface]> {
  let summary = "Ibis method call";
  let description = [{
    Dispatch a call to an Ibis method.
  }];

  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    /// Get the argument operands to the called method.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(), callee.get<mlir::SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def GetParentOfRefOp : IbisOp<"get_parent_of_ref"> {
  let summary = "Ibis get classref parent op";
  let description = [{
    Given an Ibis class reference, returns a class reference to the parent of
    said class. The returned `ibis.classref` will be an opaque type, given that
    any given Ibis class may be instantiated by multiple different parent classes.
  }];

  let arguments = (ins AnyClassRefType:$src);
  let results = (outs OpaqueClassRefType:$parent);

  let assemblyFormat = [{
    $src `:` qualified(type($src)) attr-dict
  }];
}

def GetInstanceInRefOp : IbisOp<"get_instance_in_ref"> {
  let summary = "Ibis get instance";
  let description = [{
    Given an Ibis class reference, returns a instance of said class. The instance
    is specified by the symbol name of the target child instance in the referenced
    class instance. In case of an opaque reference to the containing class, no type
    validation is performed, and it is up to lowering passes to determine whether
    the instance of the given classref actually exists.
    The returned classref, however, is typed, given that one must specify the
    symbol of the class which the instance is an instance of.
  }];

  let arguments = (ins
    AnyClassRefType:$src,
    FlatSymbolRefAttr:$instanceName,
    FlatSymbolRefAttr:$instanceSymbol);
  let results = (outs ClassRefType:$instance);
  let assemblyFormat = [{
    $instanceName `:` $instanceSymbol `in` $src `:` qualified(type($src)) attr-dict custom<ClassRefFromName>(type($instance), ref($instanceSymbol))
  }];
}

// ===---------------------------------------------------------------------===//
// Low-level Ibis operations
// ===---------------------------------------------------------------------===//

def ContainerOp : IbisOp<"container", [
  Symbol, SymbolTable, SingleBlock,
  NoTerminator, NoRegionArguments,
  HasParent<"ClassOp">
]> {
  let summary = "Ibis container";
  let description = [{
    An ibis container describes a collection of logic nested within an Ibis class.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];
}

class PortLikeOp<string mnemonic, list<Trait> traits = []> :
    IbisOp<mnemonic, !listconcat(traits, [
      PortOpInterface,
      HasParent<"ClassOp">
  ])> {
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyType>:$type);
  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
  }];

  let extraClassDeclaration = [{
    Type getPortType() {
      return getTypeAttr().getValue();
    }
  }];
}

def GetPortOp : IbisOp<"get_port", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Ibis get port";
  let description = [{
    Given an Ibis class reference, returns a port of said class. The port
    is specified by the symbol name of the port in the referenced class.
  }];

  let arguments = (ins
    ClassRefType:$instance,
    FlatSymbolRefAttr:$portSymbol);
  let results = (outs PortRefType:$port);
  let assemblyFormat = [{
    $instance `,` $portSymbol `:` qualified(type($instance)) `->` qualified(type($port)) attr-dict
  }];
}

def InputPortOp : PortLikeOp<"port.input"> {
  let summary = "Ibis input port";
}

def OutputPortOp : PortLikeOp<"port.output"> {
  let summary = "Ibis output port";
}

class PortRefToInnerTypeConstraint<string lhs, string rhs>
  : TypesMatchWith<"the lhs portref dictates the rhs type",
    lhs, rhs, "$_self.cast<PortRefType>().getPortType()">;

def PortReadOp : IbisOp<"port.read", [
      PortRefToInnerTypeConstraint<"port", "output">
]> {
  let summary = "Ibis port read";
  let description = [{
    Read the value of a local port (i.e. a port of 'this' class).
  }];

  let arguments = (ins PortRefType:$port);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    $port attr-dict `:` qualified(type($port))
  }];
}

class InnerTypeToPortRefTypeConstraint<string lhs, string rhs>
  : TypesMatchWith<"the type of the lhs value dictates the rhs portref type",
    lhs, rhs, "PortRefType::get($_ctxt, $_self)">;

def PortWriteOp : IbisOp<"port.write", [
      InnerTypeToPortRefTypeConstraint<"value", "port">
]> {
  let summary = "Ibis port write";
  let description = [{
    Write a value to a local port (i.e. a port of 'this' class).
  }];

  let arguments = (ins PortRefType:$port, AnyType:$value);
  let assemblyFormat = [{
    $port `,` $value attr-dict `:` type($value)
  }];
}

#endif // CIRCT_DIALECT_IBIS_IBISOPS_TD
