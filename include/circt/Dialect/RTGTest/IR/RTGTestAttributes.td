//===- RTGTestAttributes.td - RTGTest attributes -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTGTest attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_TD
#define CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_TD

include "circt/Dialect/RTGTest/IR/RTGTestDialect.td"
include "circt/Dialect/RTG/IR/RTGInterfaces.td"
include "circt/Dialect/RTG/IR/RTGISAAssemblyInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class RTGTestAttrDef<string name, list<Trait> traits = []> 
  : AttrDef<RTGTestDialect, name, traits>;

def CPUAttr : RTGTestAttrDef<"CPU", [ContextResourceAttrInterface]> {
  let summary = "this attribute represents a CPU referred to by the core ID";

  let parameters = (ins "size_t":$id);

  let mnemonic = "cpu";
  let assemblyFormat = "`<` $id `>`";

  let extraClassDeclaration = [{
    // TypedAttrInterface
    Type getType() const;
  }];
}

class ImmediateAttrBase<int width> : RTGTestAttrDef<"Imm" # width, [
  DeclareAttrInterfaceMethods<TypedAttrInterface>,
]> {
  let summary = "represents a " # width # "-bit immediate value";

  let mnemonic = "imm" # width;
  let parameters = (ins "uint32_t":$value);

  let assemblyFormat = "`<` $value `>`";

  let extraClassDeclaration = [{
    llvm::APInt getAPInt() const {
      return llvm::APInt(}] # width # [{, getValue());
    }
  }];

  let extraClassDefinition = [{
    Type $cppClass::getType() const {
      return Imm}] # width # [{Type::get(getContext());
    }

    LogicalResult $cppClass::verify(
      ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
      uint32_t value) {

      if (32 - llvm::countl_zero(value) > }] # width # [{)
        return emitError() << "cannot represent " << value << " with }] #
          width # [{ bits";

      return success();
    }
  }];

  let genVerifyDecl = 1;
}

def Imm5  : ImmediateAttrBase<5>;
def Imm12 : ImmediateAttrBase<12>;
def Imm13 : ImmediateAttrBase<13>;
def Imm21 : ImmediateAttrBase<21>;
def Imm32 : ImmediateAttrBase<32>;

class IntegerRegisterAttrBase<string cppName, string name, int classIndex>
  : RTGTestAttrDef<cppName, [RegisterAttrInterface]> {

  let mnemonic = name;

  let extraClassDeclaration = [{
    unsigned getClassIndex() const {
      return }] # classIndex # [{;
    }

    llvm::StringLiteral getRegisterAssembly() const {
      return "}] # name # [{";
    }

    Type getType() const {
      return IntegerRegisterType::get(getContext());
    }
  }];
}

def RegZeroAttr : IntegerRegisterAttrBase<"RegZero", "zero", 0>;
def RegRaAttr   : IntegerRegisterAttrBase<"RegRa", "ra", 1>;
def RegSpAttr   : IntegerRegisterAttrBase<"RegSp", "sp", 2>;
def RegGpAttr   : IntegerRegisterAttrBase<"RegGp", "gp", 3>;
def RegTpAttr   : IntegerRegisterAttrBase<"RegTp", "tp", 4>;
def RegT0Attr   : IntegerRegisterAttrBase<"RegT0", "t0", 5>;
def RegT1Attr   : IntegerRegisterAttrBase<"RegT1", "t1", 6>;
def RegT2Attr   : IntegerRegisterAttrBase<"RegT2", "t2", 7>;
def RegS0Attr   : IntegerRegisterAttrBase<"RegS0", "s0", 8>;
def RegS1Attr   : IntegerRegisterAttrBase<"RegS1", "s1", 9>;
def RegA0Attr   : IntegerRegisterAttrBase<"RegA0", "a0", 10>;
def RegA1Attr   : IntegerRegisterAttrBase<"RegA1", "a1", 11>;
def RegA2Attr   : IntegerRegisterAttrBase<"RegA2", "a2", 12>;
def RegA3Attr   : IntegerRegisterAttrBase<"RegA3", "a3", 13>;
def RegA4Attr   : IntegerRegisterAttrBase<"RegA4", "a4", 14>;
def RegA5Attr   : IntegerRegisterAttrBase<"RegA5", "a5", 15>;
def RegA6Attr   : IntegerRegisterAttrBase<"RegA6", "a6", 16>;
def RegA7Attr   : IntegerRegisterAttrBase<"RegA7", "a7", 17>;
def RegS2Attr   : IntegerRegisterAttrBase<"RegS2", "s2", 18>;
def RegS3Attr   : IntegerRegisterAttrBase<"RegS3", "s3", 19>;
def RegS4Attr   : IntegerRegisterAttrBase<"RegS4", "s4", 20>;
def RegS5Attr   : IntegerRegisterAttrBase<"RegS5", "s5", 21>;
def RegS6Attr   : IntegerRegisterAttrBase<"RegS6", "s6", 22>;
def RegS7Attr   : IntegerRegisterAttrBase<"RegS7", "s7", 23>;
def RegS8Attr   : IntegerRegisterAttrBase<"RegS8", "s8", 24>;
def RegS9Attr   : IntegerRegisterAttrBase<"RegS9", "s9", 25>;
def RegS10Attr  : IntegerRegisterAttrBase<"RegS10", "s10", 26>;
def RegS11Attr  : IntegerRegisterAttrBase<"RegS11", "s11", 27>;
def RegT3Attr   : IntegerRegisterAttrBase<"RegT3", "t3", 28>;
def RegT4Attr   : IntegerRegisterAttrBase<"RegT4", "t4", 29>;
def RegT5Attr   : IntegerRegisterAttrBase<"RegT5", "t5", 30>;
def RegT6Attr   : IntegerRegisterAttrBase<"RegT6", "t6", 31>;

#endif // CIRCT_DIALECT_RTGTEST_IR_RTGTESTATTRIBUTES_TD
