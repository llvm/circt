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

#endif // CIRCT_DIALECT_MOORE_MOORECLASSES
