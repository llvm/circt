//===- SVTypeDecl.td - SV type declaration ops -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemVerilog type declartions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Interface operations
//===----------------------------------------------------------------------===//

def InterfaceOp : SVOp<"interface",
    [SingleBlock, NoTerminator, Symbol, SymbolTable]> {
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

  let assemblyFormat = "attr-dict $sym_name $body";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name,
               CArg<"std::function<void()>", "{}">:$body)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &body().front(); }

    ModportType getModportType(::llvm::StringRef);
    InterfaceType getInterfaceType() {
      return InterfaceType::get(getContext(),
        ::mlir::SymbolRefAttr::get(getContext(), sym_name()));
    }
    Type getSignalType(StringRef signalName);

    // Get the interface's symbolic name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the interface's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }
  }];
}

def SignalTypeAttr : TypeAttrBase<"::mlir::Type", "Any SV/HW type"> {
  let predicate = And<[
      CPred<"$_self.isa<::mlir::TypeAttr>()">,
      CPred<"::circt::hw::getBitWidth(" #
              "$_self.cast<::mlir::TypeAttr>().getValue()) != -1">
  ]>;
}

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

  let assemblyFormat = "attr-dict $sym_name `:` $type";

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name, "::mlir::Type":$type)>
  ];
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

  let assemblyFormat = "attr-dict $sym_name custom<ModportStructs>($ports)";

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name,
                      "ArrayRef<::llvm::StringRef>":$inputs,
                      "ArrayRef<::llvm::StringRef>":$outputs)>
  ];
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

def ModportType : SVType<"Modport"> {
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

def InterfaceInstanceOp : SVOp<"interface.instance", []> {
  let summary = "Instantiate an interface";
  let description = [{
    Use this to declare an instance of an interface:
    ```
    %iface = sv.interface.instance : !sv.interface<@handshake_example>
    ```
  }];

  let arguments = (ins StrAttr:$name,
                       OptionalAttr<SymbolNameAttr>:$sym_name);
  let results = (outs InterfaceType : $result);

  let assemblyFormat =
    "(`sym` $sym_name^)? custom<ImplicitSSAName>(attr-dict) `:` type($result)";

  let builders = [
    OpBuilder<(ins "::mlir::Type":$result), [{
      return build($_builder, $_state, result, None);
    }]>
  ];

  let verifier = "return ::verifyInterfaceInstanceOp(*this);";

  let extraClassDeclaration = [{
    InterfaceType getInterfaceType() { return getType().cast<InterfaceType>(); }

    /// Lookup the interface that this is an instance of.  This returns null on
    /// invalid IR.
    Operation *getReferencedInterface(const hw::SymbolCache *cache = nullptr);
  }];
}

def GetModportOp: SVOp<"modport.get", [NoSideEffect]> {
  let summary = "Get a modport out of an interface instance";
  let description = [{
    Use this to extract a modport view to an instantiated interface. For
    example, to get the 'dataflow_in' modport on the 'handshake_example'
    interface:
    ```
    %ifaceModport = sv.modport.get @dataflow_in %iface :
      !sv.interface<@handshake_example> ->
      !sv.modport<@handshake_example::@dataflow_in>
    ```
  }];

  let arguments = (ins InterfaceType:$iface, FlatSymbolRefAttr:$field);
  let results = (outs ModportType:$result);

  let assemblyFormat =
    "$iface $field attr-dict `:` type($iface) `->` type($result)";

  let verifier = "return ::verifyGetModportOp(*this);";

  let builders = [
    OpBuilder<(ins "Value":$value, "::llvm::StringRef":$field)>
  ];
}

def AssignInterfaceSignalOp : SVOp<"interface.signal.assign", []> {
  let summary = "Assign an interfaces signal to some other signal.";
  let description = [{
    Use this to continuously assign a signal inside an interface to a
    value or other signal.

    ```
      sv.interface.signal.assign %iface(@handshake_example::@data)
        = %zero32 : i32
    ```

    Would result in the following SystemVerilog:
    ```
      assign iface.data = zero32;
    ```
  }];

  let arguments = (ins InterfaceType:$iface, FlatSymbolRefAttr:$signalName,
    AnyType:$rhs);

  let assemblyFormat = [{
    $iface `(` custom<IfaceTypeAndSignal>(type($iface), $signalName) `)`
        `=` $rhs attr-dict `:` type($rhs)
  }];

  let verifier = "return ::verifySignalExists(iface(), signalNameAttr());";
}

def ReadInterfaceSignalOp : SVOp<"interface.signal.read", [NoSideEffect]> {
  let summary = "Access the data in an interface's signal.";
  let description = [{
    This is an expression to access a signal inside of an interface.

    ```
      %ifaceData = sv.interface.signal.read %iface
          (@handshake_example::@data) : i32
    ```

    Could result in the following SystemVerilog:
    ```
      wire [31:0] ifaceData = iface.data;
    ```
  }];

  let arguments = (ins InterfaceType:$iface, FlatSymbolRefAttr:$signalName);
  let results = (outs AnyType:$signalData);

  let assemblyFormat = [{
    $iface `(` custom<IfaceTypeAndSignal>(type($iface), $signalName) `)`
        attr-dict `:` type($signalData)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$iface, "::llvm::StringRef":$signalName)>
  ];

  let verifier = "return ::verifySignalExists(iface(), signalNameAttr());";
}
