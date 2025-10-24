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

#endif // CIRCT_DIALECT_MOORE_MOORECLASSES
