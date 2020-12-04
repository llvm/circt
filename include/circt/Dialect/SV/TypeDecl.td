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

    Example:

    ```mlir
    sv.interface @myinterface {
      sv.interface.signal @data : i32
      sv.interface.modport @input_port ("input" @data)
      sv.interface.modport @output_port ("output" @data)
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    attr-dict $sym_name $body
  }];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &body().front(); }
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

    Example:

    ```mlir
    sv.interface.signal @data : i32
    ```
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

  let description = [{
    This operation defines a named modport within an interface. Its name is a
    symbol that can be looked up inside its parent interface. There is an array
    of structs that contains two fields: an enum to indicate the direction of
    the signal in the modport, and a symbol reference to refer to the signal.

    Example:

    ```mlir
    sv.interface.modport @input_port ("input" @data)
    sv.interface.modport @output_port ("output" @data)
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ModportStructArrayAttr:$ports
  );

  let assemblyFormat = [{
    attr-dict $sym_name custom<ModportStructs>($ports)
  }];
}

//===----------------------------------------------------------------------===//
// Types for integration in to type-based dialects.
//===----------------------------------------------------------------------===//

def InterfaceType : SVType<"Interface"> {
  let summary = "SystemVerilog interface type pointing to an InterfaceOp";
  let description = [{
    A MLIR type for the SV dialect's `InterfaceOp` to allow instances in any
    dialect with an open type system. Points at the InterfaceOp which defines
    the SystemVerilog interface.
  }];

  let mnemonic = "interface";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$interface);
}

def InterfaceModportType : SVType<"InterfaceModport"> {
  let summary = "SystemVerilog type pointing to an InterfaceModportOp";
  let description = [{
    A MLIR type for the SV dialect's `InterfaceModportOp` to allow
    interactions with any open type system dialect. Points at the
    InterfaceModportOp which defines the SystemVerilog interface's modport.
  }];

  let mnemonic = "modport";
  let parameters = (ins "::mlir::SymbolRefAttr":$modport);
}

//===----------------------------------------------------------------------===//
// Other operations
//===----------------------------------------------------------------------===//

def InterfaceTypeCond :
  Type<CPred<"$_self.isa<::circt::sv::InterfaceType>()">, "sv.interface">;

def InterfaceInstanceOp : SVOp<"interface.instance", [NoSideEffect]> {
  let summary = "Instantiate an interface";
  let description = [{
    Use this to declare an instance of an interface:
    ```
    %iface = sv.interface.instance : !sv.interface<@handshake_example>
    ```
  }];

  let arguments = (ins);
  let results = (outs InterfaceTypeCond : $result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];

  let verifier = [{
    return ::verifyInterfaceInstanceOp(*this);
  }];
}

def GetModportOp: SVOp<"modport.get", [NoSideEffect]> {
  let summary = "Get a modport out of an interface instance";
  let description = [{
    Use this to extract a modport view to an instantiated interface. For
    example, to get the 'dataflow_in' modport on the 'handshake_example'
    interface:
    ```
    %ifaceModport = sv.modport.get @dataflow_in %iface : !sv.interface<@handshake_example> -> !sv.modport<@handshake_example::@dataflow_in>
    ```
  }];

  let arguments = (ins InterfaceTypeCond:$iface, FlatSymbolRefAttr:$field);
  let results = (outs
    Type<CPred<"$_self.isa<::circt::sv::InterfaceModportType>()">,
      "sv.modport">:$result);

  let assemblyFormat = [{
    $iface $field attr-dict `:` type($iface) `->` type($result)
  }];

  let verifier = [{
    return ::verifyGetModportOp(*this);
  }];
}

def TypeDeclTerminatorOp : SVOp<"type_decl_terminator",
    [NoSideEffect, Terminator, ParentOneOf<["InterfaceOp"]>]> {
  let summary = "Terminator for type declaration operations";

  let arguments = (ins);
}
