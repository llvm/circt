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
      "void", "printAssembly",
      (ins "llvm::raw_ostream &":$stream,
           "llvm::function_ref<void(mlir::Value)>":$print), [{}], [{
        stream << $_op->getName().stripDialect() << " ";
        llvm::interleaveComma($_op->getOperands(), stream, [&](Value val){
          print(val);
        });
      }]>,
  ];
}

def ResourceOpInterface : OpInterface<"ResourceOpInterface"> {
  let description = [{
    This interface should be implemented by operations that represent
    resources.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    InterfaceMethod<[{
        Returns a binary representation of the resource as an APInt. 
      }],
      "::llvm::APInt", "getBinary">,
    InterfaceMethod<[{
        Emits the assembly representation of the resource to the provided
        stream. The format should be the one understood by common assembler
        tools.
      }],
      "void", "printAssembly", (ins "llvm::raw_ostream &":$stream)>,
  ];
}

def ContextResourceOpInterface : OpInterface<"ContextResourceOpInterface"> {
  let description = [{
    This interface should be implemented by operations that represent
    context resources.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    InterfaceMethod<"", "size_t", "getResourceId">,
  ];
}

def ResourceTypeInterface : TypeInterface<"ResourceTypeInterface"> {
  let description = [{
    This interface should be implemented by types that represent resources.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    InterfaceMethod<[{
        Returns the name of the resource.
      }],
      "std::string", "getName">,
  ];
}

def RegisterOpInterface : OpInterface<"RegisterOpInterface"> {
  let description = [{
    This interface should be implemented by operations that represent
    register.

    Virtual registers are indicated by a register and index of ~0.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    InterfaceMethod<[{
        Returns the bitvector of the Dialect's canonical register indexes for
        which this operation allows.
      }],
      "::llvm::BitVector", "getAllowedRegs">,
    InterfaceMethod<[{
        Returns if this selects a specirfic register.
      }],
      "unsigned", "getFixedReg">,
    InterfaceMethod<[{
        Sets this operation to use a specirfic register.
      }],
      "void", "setFixedReg", (ins "unsigned":$reg)>,
    InterfaceMethod<[{
        Returns the class specific index of the register.  This translates from
        the flat internal representation to the architectural representation.
      }],
      "APInt", "getClassIndex">,
    InterfaceMethod<[{
        Returns a suitable string for use in assembly format.
      }],
      "std::string", "getRegisterAssembly">,
  ];
}


#endif // CIRCT_DIALECT_RTG_RTGINTERFACES_TD
