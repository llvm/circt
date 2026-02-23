//===- RTGTestOps.td - RTGTest operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTGTest MLIR operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "circt/Dialect/RTG/IR/RTGTypes.td"
include "circt/Dialect/RTG/IR/RTGInterfaces.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"

// Base class for the operation in this dialect.
class RTGTestOp<string mnemonic, list<Trait> traits = []> :
  Op<RTGTestDialect, mnemonic, traits>;


def ConstantTestOp : RTGTestOp<"constant_test", [
  Pure, ConstantLike,
]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "type($result) attr-dict";
  let hasFolder = 1;
}

def GetHartIdOp : RTGTestOp<"get_hartid", [Pure]> {
  let arguments = (ins CPUType:$cpu);
  let results = (outs Index:$hartid);

  let assemblyFormat = "$cpu attr-dict";
  let hasFolder = 1;
}

def ImplicitConstraintTestOp : RTGTestOp<"implicit_constraint_op", [
  DeclareOpInterfaceMethods<ImplicitConstraintOpInterface>,
]> {
  let arguments = (ins UnitAttr:$implicitConstraint);
  let assemblyFormat = "(`implicit_constraint` $implicitConstraint^)? attr-dict";
}

//===- Instructions -------------------------------------------------------===//

def TwoRegisterInstrOp : RTGTestOp<"two_register_instr", [
  InstructionOpAdaptor
]> {
  let arguments = (ins IntegerRegisterType:$rd,
                       IntegerRegisterType:$rs);

  let assemblyFormat = "$rd `,` $rs attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      auto binary = llvm::APInt(12, 0) 
        .concat(APInt(5, cast<rtg::RegisterAttrInterface>(
          adaptor.getRs()).getClassIndex()))
        .concat(APInt(3, 0))
        .concat(APInt(5, cast<rtg::RegisterAttrInterface>(
          adaptor.getRd()).getClassIndex()))
        .concat(APInt(7, 0));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << "two_register_instr "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRd())
              .getRegisterAssembly()
         << ", "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRs())
              .getRegisterAssembly();
    }
  }];
}

def ThreeRegisterInstrOp : RTGTestOp<"three_register_instr", [
  InstructionOpAdaptor
]> {
  let arguments = (ins IntegerRegisterType:$rd,
                       IntegerRegisterType:$rs1,
                       IntegerRegisterType:$rs2);

  let assemblyFormat = "$rd `,` $rs1 `,` $rs2 attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      auto binary = llvm::APInt(7, 0) 
        .concat(APInt(5, cast<rtg::RegisterAttrInterface>(
          adaptor.getRs2()).getClassIndex()))
        .concat(APInt(5, cast<rtg::RegisterAttrInterface>(
          adaptor.getRs1()).getClassIndex()))
        .concat(APInt(3, 0))
        .concat(APInt(5, cast<rtg::RegisterAttrInterface>(
          adaptor.getRd()).getClassIndex()))
        .concat(APInt(7, 0));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << "three_register_instr "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRd())
              .getRegisterAssembly()
         << ", "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRs1())
              .getRegisterAssembly()
         << ", "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRs2())
              .getRegisterAssembly();
    }
  }];
}

def ZeroOperandInstrOp : RTGTestOp<"zero_operand_instr", [
  InstructionOpAdaptor
]> {
  let assemblyFormat = "attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      os << "12345678";
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << "zero_operand_instr";
    }
  }];
}

def ImmediateInstrOp : RTGTestOp<"immediate_instr", [InstructionOpAdaptor]> {
  let arguments = (ins IntegerRegisterType:$rd,
                       ImmediateOfWidth<32>:$imm);

  let assemblyFormat = "$rd `,` $imm attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      auto binary = cast<rtg::ImmediateAttr>(adaptor.getImm())
        .getValue().extractBits(20, 12)
        .concat(APInt(5, cast<rtg::RegisterAttrInterface>(
          adaptor.getRd()).getClassIndex()))
        .concat(APInt(7, 0));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << "immediate_instr "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRd())
              .getRegisterAssembly()
         << ", "
         << cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();
    }
  }];
}

def MemoryInstrOp : RTGTestOp<"memory_instr", [InstructionOpAdaptor]> {
  let arguments = (ins IntegerRegisterType:$rd, ISAMemory:$mem);

  let assemblyFormat = "$rd `,` $mem `:` type($mem) attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      assert(false && "binary not supported");
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << "memory_instr "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRd())
              .getRegisterAssembly()
         << ", ";

      if (auto label = dyn_cast<rtg::LabelAttr>(adaptor.getMem())) {
        os << label.getName();
        return;
      }
      
      os << cast<rtg::ImmediateAttr>(adaptor.getMem()).getValue();
    }
  }];
}
