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
    HasParent<"ComponentOp">,
    ControlLike
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
}

def SeqOp : CalyxContainer<"seq", [
    ParentOneOf<["ControlOp", "SeqOp"]>,
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
    ParentOneOf<["ControlOp", "SeqOp"]>
  ]> {
  let summary = "Calyx Enable";
  let description = [{
    The "calyx.enable" operation represents the execution of
    a group defined explicitly in the "calyx.wires" section.

    ```mlir
      calyx.enable @SomeGroup
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$groupName);
  let assemblyFormat = "$groupName attr-dict";
  let verifier = "return ::verify$cppClass(*this);";
}

