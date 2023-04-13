//===- FIRRTLDialect.td - FIRRTL dialect definition --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLDIALECT_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLDIALECT_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def HasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface,
                         ["getAsmResultNames"]>;

def FIRRTLDialect : Dialect {
  let name = "firrtl";
  let cppNamespace = "::circt::firrtl";

  let summary = "Types and operations for firrtl dialect";
  let description = [{
    This dialect defines the `firrtl` dialect, which is used to lower from Chisel
    code to Verilog.  For more information, see the [FIRRTL GitHub
    page](https://github.com/freechipsproject/firrtl).
  }];

  let hasCanonicalizer = 1;
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "circt::hw::HWDialect"
  ];

  let extraClassDeclaration = [{
     ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
     void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;

    /// Register all FIRRTL types.
    void registerTypes();
    /// Register all attributes.
    void registerAttributes();
  }];
}

// Base class for the operation in this dialect.
class FIRRTLOp<string mnemonic, list<Trait> traits = []> :
    Op<FIRRTLDialect, mnemonic, traits> {
  
  // Because FIRRTL hardware types can be 'const', folding an op can change
  // it's type from a non-'const' to a 'const' type. 
  // To accomodate this, ops with `hasFolder` automatically generate their
  // `fold` functions to call a `constFold` function. `constFold` handles
  // folding logic as normal. `fold` automatically handles type changes
  // to 'const'.
  //
  // This type change is needed for folding both values and attributes:
  // - MLIR's canonicalizer checks that folded values have types identical 
  //   to the types of the op results they are replacing. 
  //   Updating the op result type allows that check to pass. 
  // - MLIR's canonicalizer calls `materializeConstant` on folded attributes. 
  //   Updating the op result type ensures that `materializeConstant` is 
  //   provided a 'const' type for constant ops and that the canonicalizer's 
  //   internal bookkeeping of cached constants is consistent.
  
  // If the op `hasFolder`, generate a `constFold` prototype 
  // based on the number of results.
  code constFoldDecl = !if(hasFolder, !if(!eq(!size(results), 1), 
    [{
      OpFoldResult constFold(FoldAdaptor adaptor);
    }],
    [{
      LogicalResult constFold(FoldAdaptor adaptor, 
                              SmallVectorImpl<OpFoldResult> &results);
    }]
  ), "");

  // If the op `hasFolder`, generate a `fold` implementation that calls
  // `constFold` and handles type changes.
  code constFoldDef = !if(hasFolder, !if(!eq(!size(results), 1), 
    [{    
      OpFoldResult $cppClass::fold(FoldAdaptor adaptor) {
        auto result = this->constFold(adaptor);
        if (result) {
          if (auto value = result.dyn_cast<Value>()) {
            assert(mixedConstTypes((*this)->getResultTypes()[0].cast<FIRRTLBaseType>(),
                                   value.getType().cast<FIRRTLBaseType>()) 
                  && "result type can only differ by constness");
            (*this)->getResult(0).setType(value.getType());
          } else {
            // TODO: Uncomment once constant ops return const types
            // if (auto resultType =
            // (*this)->getResultTypes()[0].dyn_cast<FIRRTLBaseType>())
            //   (*this)->getResult(0).setType(resultType.getConstType(true));
          }
        }
        return result;
      }
    }],
    [{    
      LogicalResult $cppClass::fold(FoldAdaptor adaptor,
                                    SmallVectorImpl<OpFoldResult> &results) {
        if (failed(this->constFold(adaptor, results)))
          return failure();

        for (size_t i = 0, e = results.size(); i != e; ++i) {
          if (auto value = results[i].dyn_cast<Value>()) {
            assert(mixedConstTypes((*this)->getResultTypes()[i].cast<FIRRTLBaseType>(),
                                   value.getType().cast<FIRRTLBaseType>()) 
                  && "result type can only differ by constness");
            (*this)->getResult(i).setType(value.getType());
          } else if (results[i]) {
            // TODO: Uncomment once constant ops return const types
            // if (auto resultType =
            // (*this)->getResultTypes()[i].dyn_cast<FIRRTLBaseType>();
            //     resultType && result)
            //   (*this)->getResult(i).setType(resultType.getConstType(true));
          }
        }
        return success();
      }
    }]
  ), "");

  // Additional class declarations to emit
  code firrtlExtraClassDeclaration = "";

  let extraClassDeclaration = constFoldDecl # firrtlExtraClassDeclaration;
  let extraClassDefinition = constFoldDef;
}

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLDIALECT_TD
