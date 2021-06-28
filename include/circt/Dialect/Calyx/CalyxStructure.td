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
    // TODO(Calyx): Need to eventually expose the output ports.
    NoTerminator,
    SingleBlock,
    SymbolTable
  ]> {
  let summary = "Calyx Component";
  let description = [{
    The "calyx.component" operation represents an overall
    Calyx component containing:
    (1) In- and output port definitions
        that define the interface.
    (2) The cells, wires, and control schedule.
    ```mlir
      calyx.component @MyComponent(%in1: i32) -> (%out1: i8) {
        calyx.cells { ... }
        calyx.wires { ... }
        calyx.control { ... }
      }
    ```
  }];

  // TODO(Calyx): Allow explicit port naming?
  let arguments = (ins
    ArrayAttr:$inPortNames,
    ArrayAttr:$outPortNames
  );
  let results = (outs);

  let regions = (region SizedRegion<1>: $body);

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
  }];

  let verifier = "return ::verify$cppClass(*this);";
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def CellsOp : CalyxContainer<"cells", [
    HasParent<"ComponentOp">,
    IsolatedFromAbove
  ]> {
  let summary = "Calyx Cells";
  let description = [{
    The "calyx.cells" operation represents a container for
    the sub-components that are used within the parent
    component. This may only contain `calyx.cell` ops.

    ```mlir
      calyx.cells {
        %in0 = calyx.cell "a0" @A : i64
        %in1, %out0 = calyx.cell "b0" @B : i132, i8
      }
    ```
  }];

  let verifier = "return ::verify$cppClass(*this);";
}

def WiresOp : CalyxContainer<"wires", [
    HasParent<"ComponentOp">
  ]> {
  let summary = "Calyx Wires";
  let description = [{
    The "calyx.wires" operation represents a set of
    guarded connections between component instances,
    which may be placed within groups.

    ```mlir
      calyx.wires {
        calyx.group @A { ... }
      }
    ```
  }];
}

// TODO1(Calyx): The terminator for the control flow
// schedule should define that of the component.
// TODO2(Calyx): While it currently carries the
// SingleBlock trait, this may change depending
// on how we decide to represent Groups in the
// dialect.
def ControlOp : CalyxContainer<"control", [
    HasParent<"ComponentOp">
  ]> {
  let summary = "Calyx Control";
  let description = [{
    The "calyx.control" operation represents the
    execution schedule defined for the given
    component, i.e. when each group executes.

    ```mlir
      calyx.control {
        // TODO(Calyx): Add `control` examples
        // once control flow ops are added.
      }
    ```
  }];
}

def CellOp : CalyxOp<"cell", [
    HasParent<"CellsOp">
  ]> {
  let summary = "Calyx Cell";
  let description = [{
    Represents a cell (or instance) of a Calyx component or
    primitive, which may include state. Some cells may
    optionally have parameters attributed to them.

    ```mlir
      %in1, %out1 = calyx.cell "name" @MyComponent : i64, i16
    ```
  }];

  let extraClassDeclaration = [{
    /// Lookup the component for the symbol. This returns null on
    /// invalid IR.
    ComponentOp getReferencedComponent();
  }];

  // TODO(Calyx): Add `parameters` attribute for
  // SystemVerilog-wrapped primitives.
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
    RegionKindInterface,
    // TODO(Calyx): When assignments are added:
    // SingleBlockImplicitTerminator<>
    SingleBlock,
    NoTerminator,
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
        // TODO(Calyx): Add assignments when
        // they are in the Calyx dialect.
      }
    ```
  }];

  let arguments = (ins SymbolNameAttr: $sym_name);

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }

    /// Returns the body of a Calyx group.
    Block *getBody() { return &getOperation()->getRegion(0).front(); }
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $body attr-dict";
}
