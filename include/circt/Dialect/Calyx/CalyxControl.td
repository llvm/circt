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
    FlatSymbolRefAttr:$groupName
    );
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
  let description = [{
    The "calyx.if" operation represents and if-then-else construct for
    conditionally executing two calyx groups. The operands to an if operation
    is a boolean value and the group under which this value is evaluated.

    Note: The native and CIRCT calyx IRs may diverge wrt. 'with' execution, see:
    https://github.com/cucapra/calyx/discussions/588

    ```mlir
      calyx.if %1 with @G1 {
        calyx.enable @G2
        ...
      } else {
        calyx.enable @G3
        ...
      }
    ```
  }];

  let assemblyFormat = "$cond `with` $groupName $thenRegion (`else` $elseRegion^)? attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
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
  let assemblyFormat = "$groupName attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
}
