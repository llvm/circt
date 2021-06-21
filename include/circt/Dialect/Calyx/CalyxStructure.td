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

def ComponentOp : CalyxOp<"component", [
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
      component MyComponent(in1: 32) -> (out1: 32) {
        cells { ... }
        wires { ... }
        control { ... }
      }
    ```
  }];

  // TODO(Calyx): Allow explicit port naming?
  let arguments = (ins);
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

  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let verifier = "return ::verify$cppClass(*this);";
}

def CellsOp : CalyxOp<"cells", [
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    HasParent<"ComponentOp">
  ]> {
  let summary = "Calyx Cells";
  let description = [{
    The "calyx.cells" operation represents a container for
    the sub-components that are used within the parent
    component.

    ```mlir
      cells {
        adder = std_add(32);
        myComp = MyComponent();
      }
    ```
  }];
  let regions = (region SizedRegion<1>: $body);
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def WiresOp : CalyxOp<"wires", [
    NoTerminator,
    HasParent<"ComponentOp">
  ]> {
  let summary = "Calyx Wires";
  let description = [{
    The "calyx.wires" operation represents a set of
    guarded connections between component instances,
    which may be placed within groups.

    ```mlir
      wires {
        group compute_add { ... };
        ...
        out1 = adder.out;
      }
    ```
  }];
  let regions = (region SizedRegion<1>: $body);
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}

def ControlOp : CalyxOp<"control", [
    IsolatedFromAbove,
    // TODO(Calyx): The terminator for the control flow
    // schedule should define that of the component.
    NoTerminator,
    HasParent<"ComponentOp">
  ]> {
  let summary = "Calyx Control";
  let description = [{
    The "calyx.control" operation represents the
    execution schedule defined for the given
    component, i.e. when each group executes.

    ```mlir
      control {
        seq { compute_add; }
      }
    ```
  }];
  let regions = (region SizedRegion<1>: $body);
  let printer = "return ::print$cppClass(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
}
