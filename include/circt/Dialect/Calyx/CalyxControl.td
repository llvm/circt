//===- CalyxControl.td - Calyx Control -------------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the operations used for the Calyx control execution schedule.
//
//===----------------------------------------------------------------------===//

def ControlLike : NativeOpTrait<"ControlLike"> {
  let cppNamespace = "::circt::calyx";
}

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
        calyx.seq {
          calyx.enable @GroupA
        }
      }
    ```
  }];
  let verifier = "return ::verify$cppClass(*this);";
}

def IfOp : CalyxContainer<"if", [
    ControlLike
  ]> {
  let summary = "Calyx If";
  let arguments = (ins
    I1:$cond,
    OptionalAttr<FlatSymbolRefAttr>:$groupName
    );
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
  let description = [{
    The "calyx.if" operation represents and if-then-else construct for
    conditionally executing two Calyx groups. The operands to an if operation is
    a 1-bit port and an optional combinational group under which this port is driven.

    Note: The native and CIRCT Calyx IRs may diverge wrt. 'with' execution, see:
    https://github.com/cucapra/calyx/discussions/588

    ```mlir
      calyx.if %1 with @G1 {
        calyx.enable @G2
        ...
      } else {
        calyx.enable @G3
        ...
      }
      calyx.if %1 {
        calyx.enable @G2
        ...
      }
    ```
  }];

  let assemblyFormat = "$cond (`with` $groupName^)? $thenRegion (`else` $elseRegion^)? attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
  let hasCanonicalizeMethod = true;
  let extraClassDeclaration = [{
    /// Checks whether the `then` region exists.
    bool thenRegionExists() { return !getOperation()->getRegion(0).empty(); }
    /// Checks whether the `else` region exists.
    bool elseRegionExists() { return !getOperation()->getRegion(1).empty(); }
    /// Gets the single basic block representing the `then` region.
    Block *getThenBody() {
      return &getOperation()->getRegion(0).front();
    }
    /// Gets the single basic block representing the `else` region.
    Block *getElseBody() {
      assert(elseRegionExists() && "Else region does not exist.");
      return &getOperation()->getRegion(1).front();
    }
  }];
}

def SeqOp : CalyxContainer<"seq", [
    ControlLike
  ]> {
  let summary = "Calyx Sequential";
  let description = [{
    The "calyx.seq" operation executes the
    control within its region sequentially.

    ```mlir
      calyx.seq {
        // G2 will not begin execution until G1 is done.
        calyx.enable @G1
        calyx.enable @G2
      }
    ```
  }];
  let hasCanonicalizeMethod = true;
}

def ParOp : CalyxContainer<"par", [
    ControlLike
  ]> {
  let summary = "Calyx Parallel";
  let description = [{
    The "calyx.par" operation executes the
    control within its region in parallel.

    ```mlir
      calyx.par {
        // G1 and G2 will execute in parallel.
        // The region is complete when both
        // G1 and G2 are done.
        calyx.enable @G1
        calyx.enable @G2
      }
    ```
  }];
  let hasCanonicalizeMethod = true;
  let verifier = "return ::verify$cppClass(*this);";
}

def EnableOp : CalyxOp<"enable", [
    ControlLike
  ]> {
  let summary = "Calyx Enable";
  let description = [{
    The "calyx.enable" operation represents the execution of
    a group defined explicitly in the "calyx.wires" section.

    The 'compiledGroups' attribute is used in the Compile
    Control pass to track which groups are compiled within
    the new compilation group.

    ```mlir
      calyx.enable @SomeGroup
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$groupName,
    OptionalAttr<ArrayAttr>:$compiledGroups
  );
  let builders = [
    OpBuilder<(ins "StringRef":$groupName), [{
      $_state.addAttribute(
        "groupName",
        FlatSymbolRefAttr::get($_builder.getContext(), groupName)
      );
    }]>
  ];
  let assemblyFormat = "$groupName attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
}

def WhileOp : CalyxContainer<"while", [
    ControlLike
  ]> {
  let summary = "Calyx While";
  let arguments = (ins
    I1:$cond,
    OptionalAttr<FlatSymbolRefAttr>:$groupName
  );
  let description = [{
    The "calyx.while" operation represents a construct for continuously
    executing the inner groups of the 'while' operation while the condition port
    evaluates to true. The operands to a while operation is a 1-bit port and an
    optional combinational group under which this port is driven.

    Note: The native and CIRCT Calyx IRs may diverge wrt. 'with' execution, see:
    https://github.com/cucapra/calyx/discussions/588

    ```mlir
      calyx.while %1 with @G1 {
        calyx.enable @G2
        ...
      }
      calyx.while %1 {
        calyx.enable @G2
        ...
      }
    ```
  }];

  let assemblyFormat = "$cond (`with` $groupName^)? $body attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
}
