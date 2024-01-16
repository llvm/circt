//===- ArcInterfaces.td - Interfaces used in Arc -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCINTERFACES_TD
#define CIRCT_DIALECT_ARC_ARCINTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"

def ClockedOpInterface : OpInterface<"ClockedOpInterface"> {
  let description = [{
    This interface should be implemented by operations that have clocked
    behavior. Don't use this interface for operations that are not themselves
    clocked but only define a clocked region.
  }];
  let cppNamespace = "::circt::arc";

  let methods = [
    StaticInterfaceMethod<[{
        Allows non-clocked counterparts to clocked operations (e.g., `arc.call`)
        to implement this interface to simplify the implementation of some
        passes.
      }],
      "bool", "isClocked", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return true; }]>,
    InterfaceMethod<[{
        Returns the SSA value representing the clock signal. It is valid to
        return a null value if the operation is inside a clocked region and thus
        the clock is defined by the operation with the clocked region, or if the
        operation is not clocked as determined by the `isClocked` static
        function.
      }],
      "::mlir::Value", "getClock">,
    InterfaceMethod<[{
        Removes the clock value, e.g., used when moving a clocked operation into
        a clocked region. If the operation already does not have a clock, this
        should be a nop.
      }],
      "void", "eraseClock", (ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return $_op.getClockMutable().clear(); }]>,
    InterfaceMethod<[{
        Returns the latency w.r.t. to the clock returned by the `getClock`
        function.
      }],
      "uint32_t", "getLatency">,
  ];
}

#endif // CIRCT_DIALECT_ARC_ARCINTERFACES_TD
