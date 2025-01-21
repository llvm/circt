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
include "circt/Dialect/RTG/IR/RTGInterfaces.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"

// Base class for the operation in this dialect.
class RTGTestOp<string mnemonic, list<Trait> traits = []> :
  Op<RTGTestDialect, mnemonic, traits>;


def CPUDeclOp : RTGTestOp<"cpu_decl", [
  Pure,
  ConstantLike,
  ContextResourceDefining,
  FirstAttrDerivedResultType,
]> {
  let summary = "declare a CPU";
  let description = [{
    This operation is used to test the `ContextResourceOpInterface` and passes
    taking advantage of it.
  }];

  let arguments = (ins CPUAttr:$id);
  let results = (outs CPUType:$cpu);

  let assemblyFormat = "$id attr-dict";
  let hasFolder = 1;
}

def ImmediateOp : RTGTestOp<"immediate", [
  Pure,
  ConstantLike,
  FirstAttrDerivedResultType,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let summary = "declare an immediate value";

  let arguments = (ins AnyAttrOf<[Imm12, Imm21, Imm32]>:$imm);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$imm attr-dict";
  let hasFolder = 1;
}

def ConstantTestOp : RTGTestOp<"constant_test", [
  Pure, ConstantLike,
]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "type($result) attr-dict";
  let hasFolder = 1;
}

//===- Instruction Formats -------------------------------------------------===//

class InstFormatIOpBase<string mnemonic, int opcode7, int funct3>
  : RTGTestOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins IntegerRegisterType:$rd,
                       IntegerRegisterType:$rs,
                       Imm12Type:$imm);

  let assemblyFormat = "$rd `,` $rs `,` $imm attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      auto binary = APInt(12, cast<Imm12Attr>(adaptor.getImm()).getValue())
        .concat(APInt(5, cast<rtg::RegisterAttrInterface>(
          adaptor.getRs()).getClassIndex()))
        .concat(APInt(3, }] # funct3 # [{))
        .concat(APInt(5, cast<rtg::RegisterAttrInterface>(
          adaptor.getRd()).getClassIndex()))
        .concat(APInt(7, }] # opcode7 # [{));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << getOperationName().rsplit('.').second << " "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRd())
              .getRegisterAssembly()
         << ", "
         << cast<Imm12Attr>(adaptor.getImm()).getValue()
         << "("
         << cast<rtg::RegisterAttrInterface>(adaptor.getRs())
              .getRegisterAssembly()
         << ")";
    }
  }];
}

class InstFormatIImmOpBase<string mnemonic, int opcode7, int funct12>
  : RTGTestOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let assemblyFormat = "attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      auto binary = APInt(12, }] # funct12 # [{)
      .concat(APInt(13, 0))
      .concat(llvm::APInt(7, }] # opcode7 # [{));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << getOperationName().rsplit('.').second;
    }
  }];
}

//===- Instructions -------------------------------------------------------===//

def RV32I_JALROp    : InstFormatIOpBase<"jalr",  0b1100111, 0b000>;

def RV32I_LBOp      : InstFormatIOpBase<"lb",  0b0000011, 0b000>;
def RV32I_LHOp      : InstFormatIOpBase<"lh",  0b0000011, 0b001>;
def RV32I_LWOp      : InstFormatIOpBase<"lw",  0b0000011, 0b010>;
def RV32I_LBUOp     : InstFormatIOpBase<"lbu", 0b0000011, 0b100>;
def RV32I_LHUOp     : InstFormatIOpBase<"lhu", 0b0000011, 0b101>;

def RV32I_ECALLOp   : InstFormatIImmOpBase<"ecall", 0b1110011, 0b000000000000>;
def RV32I_EBREAKOp  : InstFormatIImmOpBase<"ebreak", 0b1110011, 0b000000000001>;
