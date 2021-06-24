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
    Operation *getMainComponent() { return lookupSymbol("main"); }
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
    SingleBlock
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
    component.

    ```mlir
      calyx.cells {
       // TODO(Calyx): Add `cell` examples once
       // CellOps are added.
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
        // TODO(Calyx): Add `group` examples once
        // GroupOps are added.
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
    primitive, which may include state. Some instances may
    optionally have parameters.

    ```mlir
      %out1 = calyx.cell "foo" @MyComponent(%in0) : i32 -> i64
    ```
  }];

  let extraClassDeclaration = [{
    /// Lookup the component for the symbol. This returns null on
    /// invalid IR.
    Operation *getReferencedComponent();
  }];

  // TODO(Calyx): Add parameters attribute for
  // SystemVerilog-wrapped primitives.
  let arguments = (ins
    StrAttr:$instanceName,
    FlatSymbolRefAttr:$componentName,
    Variadic<AnyType>:$inputs
  );
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $instanceName $componentName `(` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];
  let verifier = "return ::verify$cppClass(*this);";
}
