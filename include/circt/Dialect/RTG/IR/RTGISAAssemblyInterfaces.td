//===- RTGISAAssemblyInterfaces.td - ISA Assembly RTG ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGISAASSEMBLYINTERFACES_TD
#define CIRCT_DIALECT_RTG_IR_RTGISAASSEMBLYINTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/OpBase.td"

def RegisterOpInterface : OpInterface<"RegisterOpInterface"> {
  let description = [{
    This interface should be implemented by operations that represent
    ISA registers. It is used for register allocation, emission, etc.

    Virtual registers are indicated by a register index of ~0.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    InterfaceMethod<[{
        Returns the bitvector of the Dialect's canonical register indices for
        which this operation allows.
      }],
      "::llvm::BitVector", "getAllowedRegs">,
    InterfaceMethod<[{
        Returns the Dialect's canonical register index of the register if it
        selects a specific one, or ~0 if it isn't fixed yet.
      }],
      "unsigned", "getFixedReg">,
    InterfaceMethod<[{
        Sets this operation to use a specific register given by the Dialect's
        canonical register index.
      }],
      "void", "setFixedReg", (ins "unsigned":$reg)>,
    InterfaceMethod<[{
        Returns the class specific index of the register. This translates from
        the flat internal representation to the architectural representation.
      }],
      "unsigned", "getClassIndex">,
    InterfaceMethod<[{
        Returns the class specific index of the register. This translates from
        the flat internal representation to the architectural representation.
        This returns an APInt for the common binary encoding.
      }],
      "llvm::APInt", "getClassIndexBinary">,
    InterfaceMethod<[{
        Returns a suitable string for use in assembly format.
      }],
      "std::string", "getRegisterAssembly">,
  ];
}

#endif // CIRCT_DIALECT_RTG_IR_RTGISAASSEMBLYINTERFACES_TD
