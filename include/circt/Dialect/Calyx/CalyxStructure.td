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
    ```mlir
      calyx.component @MyComponent(%in1: i32) -> (%out1: i8) {
        %in, %out = "c0" @SomeComponent : i32, i32

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

      /// This is a helper function that should only be used to get the WiresOp or
      /// ControlOp of a ComponentOp, which are guaranteed to exist and generally at
      /// the end of a component's body. In the worst case, this will run in linear
      /// time with respect to the number of instances within the cell.
      template <typename Op>
      Operation* getOpWithType() {
        auto body = &getOperation()->getRegion(0).front();;
        auto opIt = llvm::find_if(llvm::reverse(*body),
                                  [](auto &&op) { return isa<Op>(op); });
        assert(opIt != body->rend() &&
         "A component should have a WiresOp and ControlOp.");
        return cast<Op>(*opIt);
      }
    public:
      /// Returns the body of a Calyx component.
      Block *getBody() { return &getOperation()->getRegion(0).front(); }

      /// Returns the WiresOp of a Calyx Component.
      Operation* getWiresOp() { return getOpWithType<WiresOp>(); }

      /// Returns the ControlOp of a Calyx Component.
      Operation* getControlOp() { return getOpWithType<ControlOp>(); }
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
    HasParent<"ComponentOp">
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
        calyx.done %3 : i1
      }
    ```
  }];

  let arguments = (ins SymbolNameAttr: $sym_name);

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $body attr-dict";
}

class SameTypeConstraint<string lhs, string rhs>
 : TypesMatchWith<"lhs and rhs types should be equivalent",
                   lhs, rhs, [{ $_self }]>;

def AssignOp : CalyxOp<"assign", [
    SameTypeConstraint<"dest", "src">
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

def DoneOp : CalyxOp<"done", [
    HasParent<"GroupOp">,
    Terminator
  ]> {
    let summary = "Calyx Done Port";
    let description = [{
      The "calyx.done" operation represents a port on a
      Calyx group that signifies when the group is finished.
      A done operation may optionally be guarded, which controls
      when the group's done operation should be active.

      ```mlir
        calyx.done %0 : i1
        calyx.done %0, %guard ? : i1
      ```
    }];

    let arguments = (ins
      I1:$src,
      Optional<I1>:$guard
    );
    let assemblyFormat = "$src (`,` $guard^ `?`)? attr-dict `:` type($src)";
}

def GoOp : CalyxOp<"go", [
    HasParent<"GroupOp">
  ]> {
    let summary = "Calyx Go Port";
    let description = [{
      The "calyx.go" operation represents a port on a
      Calyx group that signifies when the group begins.
      A go operation may optionally be guarded, which
      controls when the group's go operation should be
      active. The go operation should only be inserted
      during the Go Insertion pass. It does not
      receive a value until the Compile Control pass.

      ```mlir
        %1 = calyx.go %0 : i1
        %2 = calyx.go %3, %guard ? : i1
      ```
    }];
    let results = (outs I1);

    let arguments = (ins
      I1:$src,
      Optional<I1>:$guard
    );
    let builders = [ OpBuilder<(ins "Value":$src)> ];
    let assemblyFormat = "$src (`,` $guard^ `?`)? attr-dict `:` type($src)";
}
