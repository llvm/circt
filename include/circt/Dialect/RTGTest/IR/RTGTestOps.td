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

//===- Instruction Formats -------------------------------------------------===//

class InstFormatIOpBase<string mnemonic, int opcode7, int funct3>
  : RTGTestOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins IntegerRegisterType:$rd,
                       IntegerRegisterType:$rs,
                       ImmediateOfWidth<12>:$imm);

  let assemblyFormat = "$rd `,` $rs `,` $imm attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      auto binary = cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue()
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
         // The assembler only accepts signed values here.
         << cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue()
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

class InstFormatBOpBase<string mnemonic, int opcode7, int funct3>
  : RTGTestOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins IntegerRegisterType:$rs1,
                       IntegerRegisterType:$rs2,
                       AnyTypeOf<[ImmediateOfWidth<13>, LabelType]>:$imm);

  let assemblyFormat = [{
    $rs1 `,` $rs2 `,` $imm `:` qualified(type($imm)) attr-dict
  }];

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      assert (isa<rtg::ImmediateAttr>(adaptor.getImm()) &&
              "binary of labels not supported");

      auto imm = cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();
      auto rs1 = cast<rtg::RegisterAttrInterface>(adaptor.getRs1());
      auto rs2 = cast<rtg::RegisterAttrInterface>(adaptor.getRs2());

      auto binary = imm.extractBits(1, 12)
      .concat(imm.extractBits(6, 5))
      .concat(llvm::APInt(5, rs2.getClassIndex()))
      .concat(llvm::APInt(5, rs1.getClassIndex()))
      .concat(llvm::APInt(3, }] # funct3 # [{))
      .concat(imm.extractBits(4, 1))
      .concat(imm.extractBits(1, 11))
      .concat(llvm::APInt(7, }] # opcode7 # [{));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << getOperationName().rsplit('.').second
         << " "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRs1())
            .getRegisterAssembly()
         << ", "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRs2())
            .getRegisterAssembly()
         << ", ";

      if (auto label = dyn_cast<StringAttr>(adaptor.getImm())) {
        os << label.getValue();
        return;      
      }

      // The assembler is fine with unsigned and signed values here.
      cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue().print(os, /*isSigned=*/false);
    }
  }];
}

class InstFormatROpBase<string mnemonic, int opcode7, int funct3, int funct7>
  : RTGTestOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins IntegerRegisterType:$rd,
                       IntegerRegisterType:$rs1,
                       IntegerRegisterType:$rs2);

  let assemblyFormat = "$rd `,` $rs1 `,` $rs2 attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                         FoldAdaptor adaptor) {
      auto rd = cast<rtg::RegisterAttrInterface>(adaptor.getRd());
      auto rs1 = cast<rtg::RegisterAttrInterface>(adaptor.getRs1());
      auto rs2 = cast<rtg::RegisterAttrInterface>(adaptor.getRs2());

      auto binary = llvm::APInt(7, }] # funct7 # [{)
      .concat(llvm::APInt(5, rs2.getClassIndex()))
      .concat(llvm::APInt(5, rs1.getClassIndex()))
      .concat(llvm::APInt(3, }] # funct3 # [{))
      .concat(llvm::APInt(5, rd.getClassIndex()))
      .concat(llvm::APInt(7, }] # opcode7 # [{));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << getOperationName().rsplit('.').second
         << " "
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

class InstFormatSOpBase<string mnemonic, int opcode7, int funct3>
  : RTGTestOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins IntegerRegisterType:$rs1,
                       IntegerRegisterType:$rs2,
                       ImmediateOfWidth<12>:$imm);

  let assemblyFormat = "$rs1 `,` $rs2 `,` $imm attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      auto rs1 = cast<rtg::RegisterAttrInterface>(adaptor.getRs1());
      auto rs2 = cast<rtg::RegisterAttrInterface>(adaptor.getRs2());
      auto imm = cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();

      auto binary = imm.extractBits(7, 5)
      .concat(llvm::APInt(5, rs2.getClassIndex()))
      .concat(llvm::APInt(5, rs1.getClassIndex()))
      .concat(llvm::APInt(3, }] # funct3 # [{))
      .concat(imm.extractBits(5, 0))
      .concat(llvm::APInt(7, }] # opcode7 # [{));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << getOperationName().rsplit('.').second
         << " "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRs1())
              .getRegisterAssembly()
         << ", "
         // The assembler only accepts signed values here.
         << cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue()
         << "("
         << cast<rtg::RegisterAttrInterface>(adaptor.getRs2())
              .getRegisterAssembly()
         << ")";
    }
  }];
}

class InstFormatUOpBase<string mnemonic, int opcode7>
  : RTGTestOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins IntegerRegisterType:$rd,
                       AnyTypeOf<[ImmediateOfWidth<32>, LabelType]>:$imm);

  let assemblyFormat = "$rd `,` $imm `:` type($imm) attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      assert (isa<rtg::ImmediateAttr>(adaptor.getImm()) &&
              "binary of labels not supported");

      auto rd = cast<rtg::RegisterAttrInterface>(adaptor.getRd());
      auto imm = cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();

      auto binary = imm.extractBits(20, 12)
      .concat(llvm::APInt(5, rd.getClassIndex()))
      .concat(llvm::APInt(7, }] # opcode7 # [{));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << getOperationName().rsplit('.').second
         << " "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRd())
              .getRegisterAssembly()
         << ", ";

      if (auto label = dyn_cast<StringAttr>(adaptor.getImm())) {
        os << label.getValue();
        return;      
      }

      // The assembler wants an unsigned value here.
      cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue().print(os, /*isSigned=*/false);
    }
  }];
}

class InstFormatJOpBase<string mnemonic, int opcode7>
  : RTGTestOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins IntegerRegisterType:$rd,
                       AnyTypeOf<[ImmediateOfWidth<21>, LabelType]>:$imm);

  let assemblyFormat = "$rd `,` $imm `:` type($imm) attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      assert (isa<rtg::ImmediateAttr>(adaptor.getImm()) &&
              "binary of labels not supported");

      auto rd = cast<rtg::RegisterAttrInterface>(adaptor.getRd());
      auto imm = cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();

      auto binary = imm.extractBits(1, 20)
      .concat(imm.extractBits(10, 1))
      .concat(imm.extractBits(1, 1))
      .concat(imm.extractBits(8, 12))
      .concat(llvm::APInt(5, rd.getClassIndex()))
      .concat(llvm::APInt(7, }] # opcode7 # [{));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << getOperationName().rsplit('.').second
         << " "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRd())
              .getRegisterAssembly()
         << ", ";

      if (auto label = dyn_cast<StringAttr>(adaptor.getImm())) {
        os << label.getValue();
        return;      
      }

      // The assembler is fine with signed and unsigned values here.
      os << cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();
    }
  }];
}

class InstFormatIAOpBase<string mnemonic, int opcode7, int funct3>
  : RTGTestOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins IntegerRegisterType:$rd,
                       IntegerRegisterType:$rs,
                       ImmediateOfWidth<12>:$imm);

  let assemblyFormat = "$rd `,` $rs `,` $imm attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      auto rd = cast<rtg::RegisterAttrInterface>(adaptor.getRd());
      auto rs = cast<rtg::RegisterAttrInterface>(adaptor.getRs());
      auto imm = cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();

      auto binary = imm
      .concat(llvm::APInt(5, rs.getClassIndex()))
      .concat(llvm::APInt(3, }] # funct3 # [{))
      .concat(llvm::APInt(5, rd.getClassIndex()))
      .concat(llvm::APInt(7, }] # opcode7 # [{));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << getOperationName().rsplit('.').second
         << " "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRd())
              .getRegisterAssembly()
         << ", "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRs())
              .getRegisterAssembly()
         << ", "
         // The assembler only accepts signed values here.
         << cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();
    }
  }];
}

class InstFormatShiftOpBase<string mnemonic, int opcode7,
                            int funct3, int funct7>
  : RTGTestOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins IntegerRegisterType:$rd,
                       IntegerRegisterType:$rs,
                       ImmediateOfWidth<5>:$imm);

  let assemblyFormat = "$rd `,` $rs `,` $imm attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      auto rd = cast<rtg::RegisterAttrInterface>(adaptor.getRd());
      auto rs = cast<rtg::RegisterAttrInterface>(adaptor.getRs());
      auto imm = cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue();

      auto binary = llvm::APInt(7, }] # funct7 # [{)
      .concat(imm.extractBits(5, 0))
      .concat(llvm::APInt(5, rs.getClassIndex()))
      .concat(llvm::APInt(3, }] # funct3 # [{))
      .concat(llvm::APInt(5, rd.getClassIndex()))
      .concat(llvm::APInt(7, }] # opcode7 # [{));

      SmallVector<char> str;
      binary.toStringUnsigned(str, 16);
      os << str;
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << getOperationName().rsplit('.').second
         << " "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRd())
              .getRegisterAssembly()
         << ", "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRs())
              .getRegisterAssembly()
         << ", ";
      // The assembler only accepts an unsigned value here.
      cast<rtg::ImmediateAttr>(adaptor.getImm()).getValue().print(os, /*isSigned=*/false);
    }
  }];
}

//===- Instructions -------------------------------------------------------===//

def RV32I_LA : RTGTestOp<"rv32i.la", [InstructionOpAdaptor]> {

  let arguments = (ins IntegerRegisterType:$rd, ISAMemory:$mem);

  let assemblyFormat = "$rd `,` $mem `:` type($mem) attr-dict";

  let extraClassDefinition = [{
    void $cppClass::printInstructionBinary(llvm::raw_ostream &os,
                                           FoldAdaptor adaptor) {
      assert(false && "binary not supported");
    }

    void $cppClass::printInstructionAssembly(llvm::raw_ostream &os,
                                             FoldAdaptor adaptor) {
      os << getOperationName().rsplit('.').second << " "
         << cast<rtg::RegisterAttrInterface>(adaptor.getRd())
              .getRegisterAssembly()
         << ", ";

      if (auto label = dyn_cast<StringAttr>(adaptor.getMem())) {
        os << label.getValue();
        return;
      }
      
      os << cast<rtg::ImmediateAttr>(adaptor.getMem()).getValue();
    }
  }];
}

def RV32I_LUI       : InstFormatUOpBase<"lui",   0b0110111>;
def RV32I_AUIPC     : InstFormatUOpBase<"auipc", 0b0010111>;
def RV32I_JAL       : InstFormatJOpBase<"jal",   0b1101111>;
def RV32I_JALROp    : InstFormatIOpBase<"jalr",  0b1100111, 0b000>;

def RV32I_BEQ       : InstFormatBOpBase<"beq",  0b1100011, 0b000>;
def RV32I_BNE       : InstFormatBOpBase<"bne",  0b1100011, 0b001>;
def RV32I_BLT       : InstFormatBOpBase<"blt",  0b1100011, 0b100>;
def RV32I_BGE       : InstFormatBOpBase<"bge",  0b1100011, 0b101>;
def RV32I_BLTU      : InstFormatBOpBase<"bltu", 0b1100011, 0b110>;
def RV32I_BGEU      : InstFormatBOpBase<"bgeu", 0b1100011, 0b111>;

def RV32I_LBOp      : InstFormatIOpBase<"lb",  0b0000011, 0b000>;
def RV32I_LHOp      : InstFormatIOpBase<"lh",  0b0000011, 0b001>;
def RV32I_LWOp      : InstFormatIOpBase<"lw",  0b0000011, 0b010>;
def RV32I_LBUOp     : InstFormatIOpBase<"lbu", 0b0000011, 0b100>;
def RV32I_LHUOp     : InstFormatIOpBase<"lhu", 0b0000011, 0b101>;

def RV32I_SB        : InstFormatSOpBase<"sb", 0b0100011, 0b000>;
def RV32I_SH        : InstFormatSOpBase<"sh", 0b0100011, 0b001>;
def RV32I_SW        : InstFormatSOpBase<"sw", 0b0100011, 0b010>;

def RV32I_ADDI      : InstFormatIAOpBase<"addi", 0b0010011, 0b000>;
def RV32I_SLTI      : InstFormatIAOpBase<"slti", 0b0010011, 0b010>;
def RV32I_SLTIU     : InstFormatIAOpBase<"sltiu", 0b0010011, 0b011>;
def RV32I_XORI      : InstFormatIAOpBase<"xori", 0b0010011, 0b100>;
def RV32I_ORI       : InstFormatIAOpBase<"ori", 0b0010011, 0b110>;
def RV32I_ANDI      : InstFormatIAOpBase<"andi", 0b0010011, 0b111>;

def RV32I_SLLI      : InstFormatShiftOpBase<"slli", 0b0010011, 0b001, 0b0000000>;
def RV32I_SRLI      : InstFormatShiftOpBase<"srli", 0b0010011, 0b101, 0b0000000>;
def RV32I_SRAI      : InstFormatShiftOpBase<"srai", 0b0010011, 0b101, 0b0100000>;

def RV32I_ADD       : InstFormatROpBase<"add",  0b110011, 0b000, 0b0000000>;
def RV32I_SUB       : InstFormatROpBase<"sub",  0b110011, 0b000, 0b0100000>;
def RV32I_SLL       : InstFormatROpBase<"sll",  0b110011, 0b001, 0b0000000>;
def RV32I_SLT       : InstFormatROpBase<"slt",  0b110011, 0b010, 0b0000000>;
def RV32I_SLTU      : InstFormatROpBase<"sltu", 0b110011, 0b011, 0b0000000>;
def RV32I_XOR       : InstFormatROpBase<"xor",  0b110011, 0b100, 0b0000000>;
def RV32I_SRL       : InstFormatROpBase<"srl",  0b110011, 0b101, 0b0000000>;
def RV32I_SRA       : InstFormatROpBase<"sra",  0b110011, 0b101, 0b0100000>;
def RV32I_OR        : InstFormatROpBase<"or",   0b110011, 0b110, 0b0000000>;
def RV32I_AND       : InstFormatROpBase<"and",  0b110011, 0b111, 0b0000000>;

def RV32I_ECALLOp   : InstFormatIImmOpBase<"ecall", 0b1110011, 0b000000000000>;
def RV32I_EBREAKOp  : InstFormatIImmOpBase<"ebreak", 0b1110011, 0b000000000001>;
