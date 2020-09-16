//===- TypeDecl.td - SV type declaration ops ---------------*- tablegen -*-===//
//
// This describes the ops for SystemVerilog type declartions.
//
//===----------------------------------------------------------------------===//

def HasTypeDeclTerminator :
    SingleBlockImplicitTerminator<"TypeDeclTerminatorOp">;

//===----------------------------------------------------------------------===//
// Interface operations
//===----------------------------------------------------------------------===//

def InterfaceOp : SVOp<"interface",
    [Symbol, SymbolTable, HasTypeDeclTerminator]> {
  let summary = "Operation to define a SystemVerilog interface";

  let description = [{
    This operation defines a named interface. Its name is a symbol that can
    be looked up when declared inside a SymbolTable operation. This operation is
    also a SymbolTable itself, so the symbols in its region can be looked up.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    attr-dict $sym_name $body
  }];
}

// For now, IntegerType is a placeholder for future SV dialect types
def SignalTypeAttr : TypeAttrBase<"IntegerType", "Integer type attribute">;

def InterfaceSignalOp : SVOp<"interface.signal",
    [Symbol, HasParent<"InterfaceOp">]> {
  let summary = "Operation to define a SystemVerilog signal for interfaces";

  let description = [{
    This operation defines a named signal within an interface. Its type is
    specified in an attribute, and currently supports IntegerTypes.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    SignalTypeAttr:$type
  );

  let assemblyFormat = [{
    attr-dict $sym_name `:` $type
  }];
}

def ModportDirectionInput : StrEnumAttrCase<"input">;
def ModportDirectionOutput : StrEnumAttrCase<"output">;
def ModportDirectionInOut : StrEnumAttrCase<"inout">;

def ModportDirectionAttr : StrEnumAttr<"ModportDirectionAttr",
  "Defines direction in a modport",
  [ModportDirectionInput, ModportDirectionOutput, ModportDirectionInOut]>;

def ModportDirectionField : StructFieldAttr<"direction", ModportDirectionAttr>;

def ModportSignalField : StructFieldAttr<"signal", FlatSymbolRefAttr>;

def ModportStructAttr : StructAttr<"ModportStructAttr", SVDialect,
  [ModportDirectionField, ModportSignalField]>;

def ModportStructArrayAttr : TypedArrayAttrBase<ModportStructAttr,
  "array of modport structs">;

def InterfaceModportOp : SVOp<"interface.modport",
    [Symbol, HasParent<"InterfaceOp">]> {
  let summary = "Operation to define a SystemVerilog modport for interfaces";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ModportStructArrayAttr:$ports
  );

  let assemblyFormat = [{
    attr-dict $sym_name custom<ModportStructs>($ports)
  }];
}

//===----------------------------------------------------------------------===//
// Other operations
//===----------------------------------------------------------------------===//

def TypeDeclTerminatorOp : SVOp<"type_decl_terminator",
    [NoSideEffect, Terminator, ParentOneOf<["InterfaceOp"]>]> {
  let summary = "Terminator for type declaration operations";

  let arguments = (ins);
}
