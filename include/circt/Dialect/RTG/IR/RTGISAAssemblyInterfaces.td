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
include "mlir/IR/BuiltinAttributeInterfaces.td"

def RegisterAttrInterface : AttrInterface<"RegisterAttrInterface", [TypedAttrInterface]> {
  let description = [{
    This interface should be implemented by attributes that represent
    ISA registers. It is used for elaboration, register allocation, emission,
    etc.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    InterfaceMethod<[{
        Returns the class specific index of the register. This translates from
        the flat internal representation to the architectural representation.
      }],
      "unsigned", "getClassIndex">,
    InterfaceMethod<[{
        Returns a suitable string for use in assembly format.
      }],
      "llvm::StringLiteral", "getRegisterAssembly">,
  ];
}

def RegisterTypeInterface : TypeInterface<"RegisterTypeInterface"> {
  let description = [{
    This interface should be implemented by types that represent
    ISA registers. It is used for elaboration, register allocation, emission,
    etc.
  }];
  let cppNamespace = "::circt::rtg";
}

def RegisterAllocationOpInterface : OpInterface<"RegisterAllocationOpInterface"> {
  let description = [{
    This interface should be implemented by operations that read or write 
    registers. It is primarily used by the register allocation pass.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    InterfaceMethod<[{
        Returns the registers read by this operation. It should return all
        read registers, including implicit uses as well as those passed
        explicitly as operands. The method provides the fixed registers
        selected for the operands as arguments and the choice of implicit
        registers may depend on them.
      }],
      "void", "getReadRegisters",
      (ins "llvm::ArrayRef<rtg::RegisterAttrInterface>":$operands,
           "llvm::SmallVectorImpl<rtg::RegisterAttrInterface> &":$readRegisters),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        // Usually operands contain read and written registers, i.e., this is
        // an overapproximation and should be refined by an operation-specific
        // implementation.
        for (auto reg : operands)
          readRegisters.push_back(reg);
      }]>,

    InterfaceMethod<[{
        Returns the registers written by this operation. It should return all
        written registers, including implicit uses as well as those provided
        explicitly as operands or returned as results. The method provides the
        fixed registers selected for the operands as arguments and the choice
        of implicit registers may depend on them.
      }],
      "void", "getWrittenRegisters",
      (ins "llvm::ArrayRef<rtg::RegisterAttrInterface>":$operands,
           "llvm::SmallVectorImpl<rtg::RegisterAttrInterface> &":$writtenRegisters),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        // Usually operands contain read and written registers, i.e., this is
        // an overapproximation and should be refined by an operation-specific
        // implementation.
        for (auto reg : operands)
          writtenRegisters.push_back(reg);
      }]>,
  ];
}

def InstructionOpInterface : OpInterface<"InstructionOpInterface", [
  RegisterAllocationOpInterface
]> {
  let description = [{
    This interface should be implemented by operations that represent
    ISA instructions.
  }];
  let cppNamespace = "::circt::rtg";

  let methods = [
    StaticInterfaceMethod<[{
        Emits a binary representation of the instruction compatible with the
        ISA specification.
      }],
      "void", "printInstructionBinary",
      (ins "llvm::raw_ostream &":$os,
           "llvm::ArrayRef<mlir::Attribute>":$operands)>,
    StaticInterfaceMethod<[{
        Emits the ISA assembly representation of the instruction to the provided
        stream. The format should be the one understood by common assembler
        tools.
      }],
      "void", "printInstructionAssembly",
      (ins "llvm::raw_ostream &":$os,
           "llvm::ArrayRef<mlir::Attribute>":$operands)>,
  ];
}

def InstructionOpAdaptorTrait : NativeOpTrait<"InstructionOpAdaptorTrait"> {
  let extraConcreteClassDeclaration = [{
    static void printInstructionBinary(llvm::raw_ostream &os,
                                       FoldAdaptor adaptor);

    static void printInstructionAssembly(llvm::raw_ostream &os,
                                         FoldAdaptor adaptor);
  }];

  let extraConcreteClassDefinition = [{
    void $cppClass::printInstructionBinary(
        llvm::raw_ostream &os, llvm::ArrayRef<mlir::Attribute> operands) {
      printInstructionBinary(os, FoldAdaptor(operands));
    }

    void $cppClass::printInstructionAssembly(
        llvm::raw_ostream &os, llvm::ArrayRef<mlir::Attribute> operands) {
      printInstructionAssembly(os, FoldAdaptor(operands));
    }
  }];

  let cppNamespace = "::circt::rtg";
}

def InstructionOpAdaptor : TraitList<[
  DeclareOpInterfaceMethods<InstructionOpInterface>,
  InstructionOpAdaptorTrait,
]>;

#endif // CIRCT_DIALECT_RTG_IR_RTGISAASSEMBLYINTERFACES_TD
