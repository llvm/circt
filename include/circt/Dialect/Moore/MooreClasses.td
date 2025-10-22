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

// Base class for the operations in this dialect.
class MooreClassOp<string mnemonic, list<Trait> traits = []> :
  Op<MooreDialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// Class Attributes
//===----------------------------------------------------------------------===//

def ClassFlags : I32Enum<"ClassFlags", "class flags", [
  I32BitEnumAttrCase<"Concrete",    0x0>,
  I32BitEnumAttrCase<"Abstract",    0x1>,
  I32BitEnumAttrCase<"Interface",   0x2>,
  I32BitEnumAttrCase<"Final",       0x3>
]> {
  let cppNamespace = "circt::moore";
}

def FieldVisibility : I32Enum<"FieldVisibility", "class field visibility", [
  I32EnumAttrCase<"Public",    0>,
  I32EnumAttrCase<"Protected", 1>,
  I32EnumAttrCase<"Local",     2>
]> {
  let cppNamespace = "circt::moore";
}

def FieldLifetime : I32Enum<"FieldLifetime", "class field lifetime", [
  I32EnumAttrCase<"Automatic", 0>,
  I32EnumAttrCase<"Static",    1>
]> {
  let cppNamespace = "circt::moore";
}

def ClassFlagsAttr      : EnumAttr<MooreDialect, ClassFlags, "class_flags">;
def FieldVisibilityAttr : EnumAttr<MooreDialect, FieldVisibility, "visibility">;
def FieldLifetimeAttr   : EnumAttr<MooreDialect, FieldLifetime, "lifetime">;

//===----------------------------------------------------------------------===//
// Core Class Ops
//===----------------------------------------------------------------------===//

// moore.class.new @C(args) : (arg_types) -> !moore.class<@C>
// Allocates and constructs a new object. The constructor is implied by @C$ctor.
// Lowering chooses heap allocation and writes the vptr. Side-effecting.

def ClassNewOp : MooreClassOp<"class.new"> {
  let summary = "Allocate and construct a new class instance";
  let description = [{
    Allocates a new instance of class @C and invokes its constructor with the
    given arguments. The result is a non-null `!moore.class<@C>` handle with
    refcount initialized to 1 (see retain/release ops).
  }];

  let arguments = (ins);
  let results   = (outs ClassHandleType:$result);

  let assemblyFormat = [{
     `:` type($result) attr-dict
  }];
  let hasVerifier = 1;
}

// moore.class.field_ref %inst, @field : -> !moore.ref<T>
// Produces a reference to a field inside the object. No side effects.

def ClassFieldRefOp : MooreClassOp<"class.field_ref", [Pure]> {
  let summary = "Get a !moore.ref to a class field";
  let arguments = (ins ClassHandleRefType:$instance, FlatSymbolRefAttr:$field);
  let results   = (outs RefType:$fieldRef);
  let assemblyFormat = "$instance `:` type($instance) `,` $field `:` type($fieldRef) attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Class Declarations (metadata)
// These are module-level ops that hold metadata for verification and codegen.
//===----------------------------------------------------------------------===//

def ClassFieldDeclOp : MooreClassOp < "class.propertydecl",
  [Symbol, HasParent<"ClassDeclOp">]> {
  let summary = "Declare a class property (variable)";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    FieldVisibilityAttr:$vis,
    FieldLifetimeAttr:$lifetime
  );

  let results = (outs);
  // Printed as:  moore.class.fielddecl @foo : !moore.ref<!moore.i32> { ... }
  let assemblyFormat = [{
      `[` $vis `,` $lifetime `]` $sym_name `:` $type  attr-dict
  }];

  let extraClassDeclaration = [{
    // Convenience typed accessor for the stored type.
    ::mlir::Type getFieldType() { return getTypeAttr().getValue(); }
  }];
}

def ClassMethodDeclOp : MooreClassOp < "class.methoddecl",
  [Symbol, HasParent<"ClassDeclOp">]> {
  let summary = "Declare a class method";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    FieldVisibilityAttr:$vis,
    FieldLifetimeAttr:$lifetime
  );

  let results = (outs);
  let assemblyFormat = [{
      `[` $vis `,` $lifetime `]` $sym_name `:` $function_type  attr-dict
        }];
}

def ClassBodyEndOp : MooreClassOp<"class.declend", [Terminator, HasParent<"ClassDeclOp">]> {
  let summary = "end of classdecl region";
  let assemblyFormat = "attr-dict";
}

def ClassDeclOp : MooreClassOp < "class.classdecl",
    [Symbol, SymbolTable, IsolatedFromAbove,
     SingleBlockImplicitTerminator<"circt::moore::ClassBodyEndOp">]> {
  let summary = "Class declaration";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ClassFlagsAttr:$flag,
    OptionalAttr<SymbolRefAttr>:$base,
    OptionalAttr<SymbolRefArrayAttr>:$implementedInterfaces
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $flag $sym_name
    (`extends` $base^)?
    (`implements` $implementedInterfaces^)?
    `:` $body attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_MOORE_MOORECLASSES
