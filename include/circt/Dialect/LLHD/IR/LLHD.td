//===- LLHD.td - LLHD dialect definition -------------------*- tablegen -*-===//
//
// This is the top level file for the LLHD dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_LLHD
#define CIRCT_DIALECT_LLHD_IR_LLHD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// LLHD dialect definition
//===----------------------------------------------------------------------===//

def LLHD_Dialect : Dialect {
  let name = "llhd";

  let description = [{
    A low-level hardware description dialect in MLIR.
  }];

  let cppNamespace = "llhd";

  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// LLHD type definitions
//===----------------------------------------------------------------------===//

// LLHD Time Type
def LLHD_TimeType : Type<CPred<"$_self.isa<TimeType>()">, "LLHD time type">,
                    BuildableType<"TimeType::get($_builder.getContext())">;

// LLHD sig type
class LLHD_SigType<list<Type> allowedTypes>
  : ContainerType<AnyTypeOf<allowedTypes>, CPred<"$_self.isa<SigType>()">,
    "$_self.cast<SigType>().getUnderlyingType()", "LLHD sig type">;

def LLHD_AnySigUnderlyingType : AnyTypeOf<[AnySignlessInteger, LLHD_TimeType]>;

def LLHD_AnySigType : LLHD_SigType<[LLHD_AnySigUnderlyingType]>;

//===----------------------------------------------------------------------===//
// LLDH attribute definitions
//===----------------------------------------------------------------------===//

// LLHD time attr
def LLHD_TimeAttr
    : Attr<CPred<"$_self.isa<TimeAttr>()">, "LLHD time attribute"> {
  let storageType= [{ TimeAttr }];
  let returnType = [{ llvm::ArrayRef<unsigned> }];
  let valueType = LLHD_TimeType;
}

//===----------------------------------------------------------------------===//
// LLHD op definition
//===----------------------------------------------------------------------===//

// Base class for all LLHD ops.
class LLHD_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<LLHD_Dialect, mnemonic, traits> {

  // For each LLHD op, the following static functions need to be defined in
  // LLHDOps.cpp:
  //
  // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
  //                                               OperationState &state);
  // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let printer = [{ ::print(p, *this); }];
}

class LLHD_ArithmeticOrBitwiseOp<string mnemonic, list<OpTrait> traits = []>
    : Op<LLHD_Dialect, mnemonic,
         !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])> {

  let results = (outs AnySignlessInteger);

  let parser = [{
      return impl::parseOneResultSameOperandTypeOp(parser, result);
  }];

  let printer = [{
      impl::printOneResultOp(this->getOperation(), p);
  }];
}

//===----------------------------------------------------------------------===//
// LLHD trait definitions
//===----------------------------------------------------------------------===//

class SameTypeArbitraryWidth<string desc, string lhs, string rhs>
    : PredOpTrait<desc, CPred<"sameKindArbitraryWidth(" # lhs # ".getType(),"
                                                        # rhs # ".getType())">>;

//===----------------------------------------------------------------------===//
// LLHD Operations
//===----------------------------------------------------------------------===//

include "ValueOps.td"
include "ArithmeticOps.td"
include "BitwiseOps.td"
include "SignalOps.td"
include "ExtractOps.td"
include "InsertOps.td"
include "StructureOps.td"

#endif // CIRCT_DIALECT_LLHD_IR_LLHD
