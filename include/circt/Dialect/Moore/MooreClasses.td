#ifndef CIRCT_DIALECT_MOORE_MOORECLASSES
#define CIRCT_DIALECT_MOORE_MOORECLASSES

include "circt/Dialect/Moore/MooreAttributes.td"
include "circt/Dialect/Moore/MooreDialect.td"
include "circt/Dialect/Moore/MooreTypes.td"
include "circt/Dialect/Moore/MooreOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def ClassPropertyDeclOp
    : MooreOp<"class.propertydecl", [Symbol, HasParent<"ClassDeclOp">]> {
  let summary = "Declare a class property (variable)";

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);

  let results = (outs);
  // Printed as:  moore.class.propertydecl @foo : !moore.ref<!moore.i32> { ... }
  let assemblyFormat = [{
     $sym_name `:` $type  attr-dict
  }];

  let extraClassDeclaration = [{
    // Convenience typed accessor for the stored type.
    ::mlir::Type getPropertyType() { return getTypeAttr().getValue(); }
  }];
}

def ClassMethodDeclOp
    : MooreOp<"class.methoddecl", [Symbol, HasParent<"ClassDeclOp">]> {
  let summary = "Declare a class method";

  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type);

  let results = (outs);
  let assemblyFormat = [{
      $sym_name `:` $function_type  attr-dict
  }];
}

def ClassBodyEndOp
    : MooreOp<"class.declend", [Terminator, HasParent<"ClassDeclOp">]> {
  let summary = "end of classdecl region";
  let assemblyFormat = "attr-dict";
}

def ClassDeclOp
    : MooreOp<"class.classdecl", [Symbol, SymbolTable, IsolatedFromAbove,
                                       SingleBlockImplicitTerminator<
                                           "circt::moore::ClassBodyEndOp">]> {
  let summary = "Class declaration";
  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<SymbolRefAttr>:$base,
      OptionalAttr<SymbolRefArrayAttr>:$implementedInterfaces);
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $sym_name
    (`extends` $base^)?
    (`implements` $implementedInterfaces^)?
    `:` $body attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Core Class Ops
//===----------------------------------------------------------------------===//

def ClassNewOp
    : MooreOp<
          "class.new", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Allocate a new class instance";
  let description = [{
    Allocates a new instance of class @C. This op does not call the constructor.
    The result is a non-null `!moore.class.object<@C>` handle.
  }];

  let arguments = (ins);
  let results = (outs ClassHandleType:$result);

  let assemblyFormat = [{
     `:` type($result) attr-dict
  }];
  let hasVerifier = 1;
}

def ClassPropertyRefOp : MooreOp<"class.property_ref", [Pure]> {
  let summary = "Get a !moore.ref to a class property";
  let arguments = (ins ClassHandleRefType:$instance,
      FlatSymbolRefAttr:$property);
  let results = (outs RefType:$propertyRef);
  let assemblyFormat = "$instance`[`$property`]` `:` type($instance) `->` "
                       "type($propertyRef) attr-dict";
  let hasVerifier = 1;
}

def ClassUpcastOp : MooreOp<"class.upcast", [Pure]> {
  let summary = "Upcast a derived handle to a base handle (zero-cost)";
  let arguments = (ins ClassHandleRefType:$instance);
  let results = (outs ClassHandleRefType:$result);
  let assemblyFormat =
      "$instance `:` type($instance) `to` type($result) attr-dict";
  let hasVerifier = 1;
}

// moore.class.call @C::m(%inst, args) : (types) -> (results)
// Direct, non-virtual call. Uses CallOpInterface.

def ClassCallOp
    : MooreOp<"class.call", [CallOpInterface, ArgAndResultAttrsOpInterface]> {
  let summary = "Direct call to a (non-virtual) class method";
  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $operands `)` `:` `(`type($operands)`)` "
                       "`->` `(` type($results) `)` attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getOperands();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return CallInterfaceCallable(getCalleeAttr());
    }

    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
         if (auto sym = callee.dyn_cast<mlir::SymbolRefAttr>())
            setCalleeAttr(sym);
         else
             llvm_unreachable("Expected SymbolRefAttr for callee");
    }

    /// Stub implementations for ArgumentAttributesMethods. If desired,
    /// implement these by defining arg_attrs and res_attrs as arguments to the
    /// operation as OptionalAttr<DictArrayAttr>.
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr args) {}
    void setResAttrsAttr(mlir::ArrayAttr args) {}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];
}

#endif // CIRCT_DIALECT_MOORE_MOORECLASSES
