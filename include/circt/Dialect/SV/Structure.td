//===- Structure.td - SV structure ops -------------------*- tablegen -*-===//
//
// This describes the ops for SystemVerilog structure.
//
//===----------------------------------------------------------------------===//

def HasStructureTerminator :
    SingleBlockImplicitTerminator<"StructureTerminatorOp">;

//===----------------------------------------------------------------------===//
// Interface operations
//===----------------------------------------------------------------------===//

def InterfaceOp : SVOp<"interface",
    [Symbol, SymbolTable, HasStructureTerminator]>, HasCustomParserPrinter {
  let summary = "Operation to define a SystemVerilog interface";

  let description = [{
    This operation defines a named interface. Its name is a symbol that can
    be looked up when declared inside a SymbolTable operation. This operation is
    also a SymbolTable itself, so the symbols in its region can be looked up.
  }];

  let regions = (region SizedRegion<1>:$body);
}

// For now, IntegerType is a placeholder for future SV dialect types
def SignalTypeAttr : TypeAttrBase<"IntegerType", "Integer type attribute">;

def InterfaceSignalOp : SVOp<"interface.signal", [Symbol, HasParent<"InterfaceOp">]>,
    HasCustomParserPrinter {
  let summary = "Operation to define a SystemVerilog signal for interfaces";

  let description = [{
    This operation defines a named signal within an interface. Its type is
    specified in an attribute, and currently supports IntegerTypes.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    SignalTypeAttr:$type
  );
}

def ModPortDirectionInput : StrEnumAttrCase<"input">;
def ModPortDirectionOutput : StrEnumAttrCase<"output">;
def ModPortDirectionInout : StrEnumAttrCase<"inout">;

def ModPortDirectionAttr : StrEnumAttr<"ModPortDirectionAttr",
  "Defines direction in a modport",
  [ModPortDirectionInput, ModPortDirectionOutput, ModPortDirectionInout]>;

def ModPortDirectionField : StructFieldAttr<"direction", ModPortDirectionAttr>;

def ModPortSignalField : StructFieldAttr<"signal", FlatSymbolRefAttr>;

def ModPortStructAttr : StructAttr<"ModPortStructAttr", SVDialect,
  [ModPortDirectionField, ModPortSignalField]>;

def ModPortStructArrayAttr : TypedArrayAttrBase<ModPortStructAttr,
  "array of modport structs">;

def InterfaceModPortOp : SVOp<"interface.modport", [Symbol, HasParent<"InterfaceOp">]>,
    HasCustomParserPrinter {
  let summary = "Operation to define a SystemVerilog modport for interfaces";

  let arguments = (ins
    StrAttr:$sym_name,
    ModPortStructArrayAttr:$ports
  );

  let verifier = [{
    return ::verify$cppClass(*this);
  }];
}

//===----------------------------------------------------------------------===//
// Other operations
//===----------------------------------------------------------------------===//

def StructureTerminatorOp : SVOp<"structure_terminator",
    [NoSideEffect, Terminator, ParentOneOf<["InterfaceOp"]>]> {
  let summary = "Terminator for structure operations";

  let arguments = (ins);
}
