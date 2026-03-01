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
  Op<RTGTestDialect, mnemonic, traits> {
}


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
  let assemblyFormat = [{
    (`implicit_constraint` $implicitConstraint^)? attr-dict
  }];
}

//===- Instruction Formats -------------------------------------------------===//

class InstrFormatOp<string mnemonic, list<Trait> traits = []> :
  RTGTestOp<mnemonic, traits>, ISAInstructionFormat;

class InstFormatIOpBase<string mnemonic, string opcode7, string funct3>
  : InstrFormatOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins Arg<IntegerRegisterType, "", [DestReg]>:$rd,
                       Arg<IntegerRegisterType, "", [SourceReg]>:$rs,
                       ImmediateOfWidth<12>:$imm);

  let assemblyFormat = "$rd `,` $rs `,` $imm attr-dict";
  let isaBinaryFormat = "$imm $rs " # funct3 # " $rd " # opcode7;
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` signed($imm) `(` $rs `)`";
}

class InstFormatIImmOpBase<string mnemonic, string opcode7, string funct12>
  : InstrFormatOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let assemblyFormat = "attr-dict";
  let isaBinaryFormat = funct12 # " 0b0000000000000 " # opcode7;
  let isaAssemblyFormat = "mnemonic";
}

class InstFormatBOpBase<string mnemonic, string opcode7, string funct3>
  : InstrFormatOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins Arg<IntegerRegisterType, "", [SourceReg]>:$rs1,
                       Arg<IntegerRegisterType, "", [SourceReg]>:$rs2,
                       AnyTypeOf<[ImmediateOfWidth<13>, LabelType]>:$imm);

  let assemblyFormat = [{
    $rs1 `,` $rs2 `,` $imm `:` qualified(type($imm)) attr-dict
  }];
  let isaBinaryFormat = "$imm[12] $imm[10:5] $rs2 $rs1 " # funct3 #
                        " $imm[4:1] $imm[11] " # opcode7;
  let isaAssemblyFormat = "mnemonic ` ` $rs1 `, ` $rs2 `, ` unsigned($imm)";
}

class InstFormatROpBase<string mnemonic, string opcode7, string funct3, string funct7>
  : InstrFormatOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins Arg<IntegerRegisterType, "", [DestReg]>:$rd,
                       Arg<IntegerRegisterType, "", [SourceReg]>:$rs1,
                       Arg<IntegerRegisterType, "", [SourceReg]>:$rs2);

  let assemblyFormat = "$rd `,` $rs1 `,` $rs2 attr-dict";
  let isaBinaryFormat = funct7 # " $rs2 $rs1 " # funct3 # " $rd " # opcode7;
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` $rs1 `, ` $rs2";
}

class InstFormatSOpBase<string mnemonic, string opcode7, string funct3>
  : InstrFormatOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins Arg<IntegerRegisterType, "", [SourceReg]>:$rs1,
                       Arg<IntegerRegisterType, "", [SourceReg]>:$rs2,
                       ImmediateOfWidth<12>:$imm);

  let assemblyFormat = "$rs1 `,` $rs2 `,` $imm attr-dict";
  let isaBinaryFormat = "$imm[11:5] $rs2 $rs1 " # funct3 #
                        " $imm[4:0] " # opcode7;
  let isaAssemblyFormat = "mnemonic ` ` $rs1 `, ` signed($imm) `(` $rs2 `)`";
}

class InstFormatUOpBase<string mnemonic, string opcode7>
  : InstrFormatOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins Arg<IntegerRegisterType, "", [DestReg]>:$rd,
                       AnyTypeOf<[ImmediateOfWidth<32>, LabelType]>:$imm);

  let assemblyFormat = "$rd `,` $imm `:` type($imm) attr-dict";
  let isaBinaryFormat = "$imm[31:12] $rd " # opcode7;
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` unsigned($imm)";
}

class InstFormatJOpBase<string mnemonic, string opcode7>
  : InstrFormatOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins Arg<IntegerRegisterType, "", [DestReg]>:$rd,
                       AnyTypeOf<[ImmediateOfWidth<21>, LabelType]>:$imm);

  let assemblyFormat = "$rd `,` $imm `:` type($imm) attr-dict";
  let isaBinaryFormat =
    "$imm[20] $imm[10:1] $imm[11] $imm[19:12] $rd " # opcode7;
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` signed($imm)";
}

class InstFormatIAOpBase<string mnemonic, string opcode7, string funct3>
  : InstrFormatOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins Arg<IntegerRegisterType, "", [DestReg]>:$rd,
                       Arg<IntegerRegisterType, "", [SourceReg]>:$rs,
                       ImmediateOfWidth<12>:$imm);

  let assemblyFormat = "$rd `,` $rs `,` $imm attr-dict";

  // Auto-generated binary and assembly methods
  let isaBinaryFormat = "$imm $rs " # funct3 # " $rd " # opcode7;
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` $rs `, ` signed($imm)";
}

class InstFormatShiftOpBase<string mnemonic, string opcode7,
                            string funct3, string funct7>
  : InstrFormatOp<"rv32i." # mnemonic, [InstructionOpAdaptor]> {

  let arguments = (ins Arg<IntegerRegisterType, "", [DestReg]>:$rd,
                       Arg<IntegerRegisterType, "", [SourceReg]>:$rs,
                       ImmediateOfWidth<5>:$imm);

  let assemblyFormat = "$rd `,` $rs `,` $imm attr-dict";
  let isaBinaryFormat = funct7 # " $imm $rs " # funct3 # " $rd " # opcode7;
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` $rs `, ` unsigned($imm)";
}

//===- Instructions -------------------------------------------------------===//

def RV32I_LA : InstrFormatOp<"rv32i.la", [InstructionOpAdaptor]> {

  let arguments = (ins Arg<IntegerRegisterType, "", [DestReg]>:$rd,
                       ISAMemory:$mem);

  let assemblyFormat = "$rd `,` $mem `:` type($mem) attr-dict";
  let isaAssemblyFormat = "mnemonic ` ` $rd `, ` unsigned($mem)";
}

def RV32I_LUI       : InstFormatUOpBase<"lui",   "0b0110111">;
def RV32I_AUIPC     : InstFormatUOpBase<"auipc", "0b0010111">;
def RV32I_JAL       : InstFormatJOpBase<"jal",   "0b1101111">;
def RV32I_JALROp    : InstFormatIOpBase<"jalr",  "0b1100111", "0b000">;

def RV32I_BEQ       : InstFormatBOpBase<"beq",  "0b1100011", "0b000">;
def RV32I_BNE       : InstFormatBOpBase<"bne",  "0b1100011", "0b001">;
def RV32I_BLT       : InstFormatBOpBase<"blt",  "0b1100011", "0b100">;
def RV32I_BGE       : InstFormatBOpBase<"bge",  "0b1100011", "0b101">;
def RV32I_BLTU      : InstFormatBOpBase<"bltu", "0b1100011", "0b110">;
def RV32I_BGEU      : InstFormatBOpBase<"bgeu", "0b1100011", "0b111">;

def RV32I_LBOp      : InstFormatIOpBase<"lb",  "0b0000011", "0b000">;
def RV32I_LHOp      : InstFormatIOpBase<"lh",  "0b0000011", "0b001">;
def RV32I_LWOp      : InstFormatIOpBase<"lw",  "0b0000011", "0b010">;
def RV32I_LBUOp     : InstFormatIOpBase<"lbu", "0b0000011", "0b100">;
def RV32I_LHUOp     : InstFormatIOpBase<"lhu", "0b0000011", "0b101">;

def RV32I_SB        : InstFormatSOpBase<"sb", "0b0100011", "0b000">;
def RV32I_SH        : InstFormatSOpBase<"sh", "0b0100011", "0b001">;
def RV32I_SW        : InstFormatSOpBase<"sw", "0b0100011", "0b010">;

def RV32I_ADDI      : InstFormatIAOpBase<"addi", "0b0010011", "0b000">;
def RV32I_SLTI      : InstFormatIAOpBase<"slti", "0b0010011", "0b010">;
def RV32I_SLTIU     : InstFormatIAOpBase<"sltiu", "0b0010011", "0b011">;
def RV32I_XORI      : InstFormatIAOpBase<"xori", "0b0010011", "0b100">;
def RV32I_ORI       : InstFormatIAOpBase<"ori", "0b0010011", "0b110">;
def RV32I_ANDI      : InstFormatIAOpBase<"andi", "0b0010011", "0b111">;

def RV32I_SLLI      : InstFormatShiftOpBase<"slli", "0b0010011", "0b001", "0b0000000">;
def RV32I_SRLI      : InstFormatShiftOpBase<"srli", "0b0010011", "0b101", "0b0000000">;
def RV32I_SRAI      : InstFormatShiftOpBase<"srai", "0b0010011", "0b101", "0b0100000">;

def RV32I_ADD       : InstFormatROpBase<"add",  "0b0110011", "0b000", "0b0000000">;
def RV32I_SUB       : InstFormatROpBase<"sub",  "0b0110011", "0b000", "0b0100000">;
def RV32I_SLL       : InstFormatROpBase<"sll",  "0b0110011", "0b001", "0b0000000">;
def RV32I_SLT       : InstFormatROpBase<"slt",  "0b0110011", "0b010", "0b0000000">;
def RV32I_SLTU      : InstFormatROpBase<"sltu", "0b0110011", "0b011", "0b0000000">;
def RV32I_XOR       : InstFormatROpBase<"xor",  "0b0110011", "0b100", "0b0000000">;
def RV32I_SRL       : InstFormatROpBase<"srl",  "0b0110011", "0b101", "0b0000000">;
def RV32I_SRA       : InstFormatROpBase<"sra",  "0b0110011", "0b101", "0b0100000">;
def RV32I_OR        : InstFormatROpBase<"or",   "0b0110011", "0b110", "0b0000000">;
def RV32I_AND       : InstFormatROpBase<"and",  "0b0110011", "0b111", "0b0000000">;

def RV32I_ECALLOp   : InstFormatIImmOpBase<"ecall", "0b1110011", "0b000000000000">;
def RV32I_EBREAKOp  : InstFormatIImmOpBase<"ebreak", "0b1110011", "0b000000000001">;
