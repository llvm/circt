//===- RTGInterfaces.td - Interfaces used in RTG -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_RTGINTERFACES_TD
#define CIRCT_DIALECT_RTG_RTGINTERFACES_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"

def InstructionOpInterface : OpInterface<"InstructionOpInterface"> {
  let description = [{
    This interface should be implemented by operations that represent
    instructions, the base unit for random test generation.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    InterfaceMethod<[{
        Returns a binary representation of the instruction as an APInt. 
      }],
      "::llvm::APInt", "getBinary",
      (ins "llvm::ArrayRef<llvm::APInt>":$operands)>,
    InterfaceMethod<[{
        Emits the assembly representation of the instruction to the provideed
        stream. The format should be the one understood by common assembler
        tools.
      }],
      "void", "printAssembly", (ins "llvm::raw_ostream &":$stream)>,
  ];
}

#endif // CIRCT_DIALECT_RTG_RTGINTERFACES_TD
