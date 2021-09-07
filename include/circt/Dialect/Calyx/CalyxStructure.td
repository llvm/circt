//===- CalyxStructure.td - Calyx Structure -------------*- tablegen -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the Calyx structures.
//
//===----------------------------------------------------------------------===//

def UndefinedOp : CalyxOp<"undef", [
    NoSideEffect
  ]> {
  let summary = "Calyx Undefined Value";
  let description = [{
    The "undef" operation represents an undefined value
    that may be used when a specific source or destination
    does not have an assignment yet. This is used to avoid
    pulling in the entire LLVMIR dialect for a single
    operation.

    ```mlir
      %0 = calyx.undef : i1
    ```
  }];
  let results = (outs AnyType:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

def ProgramOp : CalyxContainer<"program", [
    IsolatedFromAbove,
    SymbolTable
  ]> {
  let summary = "Calyx Program";
  let description = [{
    The "calyx.program" operation represents an overall Calyx program,
    containing a list of Calyx components. This must include a "main"
    component, the entry point of the program.
  }];
  let arguments = (ins);

  let extraClassDeclaration = [{
    /// Returns the main component, representing the
    /// entry point of the Calyx program.
    ComponentOp getMainComponent() { return lookupSymbol<ComponentOp>("main"); }
  }];
  let verifier = "return ::verify$cppClass(*this);";
}

def ComponentOp : CalyxOp<"component", [
    HasParent<"ProgramOp">,
    Symbol,
    FunctionLike,
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Calyx Component";
  let description = [{
    The "calyx.component" operation represents an overall
    Calyx component containing:
    (1) In- and output port definitions
        that define the interface.
    (2) The cells, wires, and control schedule.

    A Calyx component requires the following ports with bit width 1:
    - Input ports: `go`, `clk`, and `reset`.
    - Output ports: `done`.

    ```mlir
      calyx.component @MyComponent(%in: i32, %go: i1, %clk: i1, %reset: i1) -> (%out: i16, %done: i1) {
        ...
        calyx.wires { ... }
        calyx.control { ... }
      }
    ```
  }];

  // TODO(Calyx): Allow explicit port naming?
  let arguments = (ins
    ArrayAttr:$portNames
  );
  let results = (outs);
  let regions = (region SizedRegion<1>: $body);

  let builders = [
    OpBuilder<(ins "StringAttr":$name, "ArrayRef<PortInfo>":$ports)>
  ];

  let extraClassDeclaration = [{
    // Necessary to avoid name clashing with `front`.
    using FunctionLike::front;

    private:
      // This trait needs access to the hooks defined below.
      friend class OpTrait::FunctionLike<ComponentOp>;

      /// Hooks for the input/output type enumeration in FunctionLike.
      unsigned getNumFuncArguments() { return getType().getNumInputs(); }
      unsigned getNumFuncResults() { return getType().getNumResults(); }
    public:
      /// Returns the body of a Calyx component.
      Block *getBody() { return &getOperation()->getRegion(0).front(); }

      /// Returns the WiresOp of a Calyx Component.
      WiresOp getWiresOp();

      /// Returns the ControlOp of a Calyx Component.
      ControlOp getControlOp();

      /// Gets the Done port of a Calyx component.
      Value getDonePort();

      /// Gets the Go port of a Calyx component.
      Value getGoPort();
  }];

  let verifier = "return ::verify$cppClass(*this);";
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def WiresOp : CalyxContainer<"wires", [
    HasParent<"ComponentOp">,
    SymbolTable
  ]> {
  let summary = "Calyx Wires";
  let description = [{
    The "calyx.wires" operation represents a set of
    guarded connections between component instances,
    which may be placed within groups.

    ```mlir
      calyx.wires {
        calyx.group @A { ... }
        calyx.assign %1 = %2 : i16
      }
    ```
  }];
  let verifier = "return ::verify$cppClass(*this);";
}

def InstanceOp : CalyxCell<"instance", [
  ]> {
  let summary = "Calyx Component Instance";
  let description = [{
    Represents an instance of a Calyx component,
    which may include state.

    ```mlir
      %name.in, %name.out = calyx.instance "name" @MyComponent : i64, i16
    ```
  }];

  let extraClassDeclaration = [{
    /// Lookup the component for the symbol. This returns null on
    /// invalid IR.
    ComponentOp getReferencedComponent();
  }];

  let arguments = (ins
    StrAttr:$instanceName,
    FlatSymbolRefAttr:$componentName
  );
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $instanceName $componentName attr-dict (`:` type($results)^)?
  }];
  let verifier = "return ::verify$cppClass(*this);";
}

def GroupOp : CalyxOp<"group", [
    HasParent<"WiresOp">,
    NoRegionArguments,
    RegionKindInterface,
    SingleBlock,
    Symbol
  ]> {
  let summary = "Calyx Group";
  let description = [{
    Represents a Calyx group, which is a collection
    of assignments that are only active when the group
    is run from the control execution schedule. A group
    signifies its termination with a special port named
    a "done" port.

    ```mlir
      calyx.group @MyGroup {
        calyx.assign %1 = %2 : i32
        calyx.group_done %3 : i1
      }
    ```
  }];

  let arguments = (ins SymbolNameAttr: $sym_name);

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the GroupGoOp for this group.
    GroupGoOp getGoOp();

    /// Returns the GroupDoneOp for this group.
    GroupDoneOp getDoneOp();
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $body attr-dict";
}

def AssignOp : CalyxOp<"assign", [
    SameTypeConstraint<"dest", "src">,
    ParentOneOf<["GroupOp", "WiresOp"]>
  ]> {
  let summary = "Calyx Assignment";
  let description = [{
    The "calyx.assign" operation represents a non-blocking
    assignment. An assignment may optionally be guarded,
    which controls when the assignment should be active.
    This operation should only be instantiated in the
    "calyx.wires" section or a "calyx.group".

    ```mlir
      calyx.assign %1 = %2 : i16
      calyx.assign %1 = %2, %guard ? : i16
    ```
  }];
  let arguments = (ins
    AnyType:$dest,
    AnyType:$src,
    Optional<I1>:$guard
  );
  let verifier = "return ::verify$cppClass(*this);";

  // TODO(Calyx): Calyx IR typically represents a
  // guarded assignments as `%dest = %guard ? %src`
  let assemblyFormat = [{
    $dest `=` $src (`,` $guard^ `?`)? attr-dict `:` type($dest)
  }];
}

def GroupDoneOp : CalyxGroupPort<"group_done", [
    Terminator
  ]> {
  let summary = "Calyx Group Done Port";
  let description = [{
    The "calyx.group_done" operation represents a port on a
    Calyx group that signifies when the group is finished.
    A done operation may optionally be guarded, which controls
    when the group's done operation should be active.

    ```mlir
      calyx.group_done %v1 : i1
      calyx.group_done %v2, %guard ? : i1
    ```
  }];
  let results = (outs);
}

def GroupGoOp : CalyxGroupPort<"group_go", [
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
]> {
  let summary = "Calyx Group Go Port";
  let description = [{
    The "calyx.group_go" operation represents a port on a
    Calyx group that signifies when the group begins.
    A go operation may optionally be guarded, which
    controls when the group's go operation should be
    active. The go operation should only be inserted
    during the Go Insertion pass. It does not
    receive a source until the Compile Control pass.

    ```mlir
      %group_name1.go = calyx.group_go %0 : i1
      %group_name2.go = calyx.group_go %3, %guard ? : i1
    ```
  }];
  let results = (outs I1);
  let builders = [
    OpBuilder<(ins "Value":$src, CArg<"Value", "{}">:$guard), [{
      $_state.addTypes($_builder.getI1Type());
      $_state.addOperands(src);
      if (guard)
        $_state.addOperands(guard);
    }]>
  ];
}
