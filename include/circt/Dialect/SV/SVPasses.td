//===-- SVPasses.td - SV pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for passes that work on the SV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SV_SVPASSES
#define CIRCT_DIALECT_SV_SVPASSES

include "mlir/Pass/PassBase.td"

def AlwaysFusion : Pass<"sv-always-fusion", "rtl::RTLModuleOp"> {
  let summary = "Combine AlwaysFF blocks with the same predicate";
  let description = [{
      This pass finds all AlwaysFF operations with the same condition, and
      combines them in to a single operation. For example:
      ```mlir
      // Input
      rtl.module() {
        sv.alwaysff(posedge %clock)  {
          sv.passign %counter, %1 : i8
        }
        sv.alwaysff(posedge %clock)  {
          sv.passign %param, %2 : i16
        }
      }
  
      // Output
      rtl.module() {
        sv.alwaysff(posedge %clock)  {
          sv.passign %counter, %1 : i8
          sv.passign %param, %2 : i16
        }
      }
      ```
  }];

  let constructor = "circt::sv::createAlwaysFusionPass()";
}

#endif // CIRCT_DIALECT_SV_SVPASSES
