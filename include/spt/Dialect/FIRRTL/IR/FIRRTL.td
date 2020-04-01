//===- FIRRTL.td - FIRRTL dialect definition ---------------*- tablegen -*-===//
//
// This is the operation definition file for FIRRTL dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef FIRRTL_OPS
#define FIRRTL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"

def FIRRTLDialect : Dialect {
  let name = "firrtl";

  let summary = "Types and operations for firrtl dialect";
  let description = [{
    This dialect defines the `firrtl` dialect, which is used to lower from
    Chisel code to Verilog.  For more information, see the
    [FIRRTL GitHub page](https://github.com/freechipsproject/firrtl).
  }];

  let cppNamespace = "firrtl";
}

//===----------------------------------------------------------------------===//
// FIRRTL op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class FIRRTLOp<string mnemonic, list<OpTrait> traits = []> :
    Op<FIRRTLDialect, mnemonic, traits>;

def CircuitOp : FIRRTLOp<"circuit",
      [IsolatedFromAbove, SingleBlockImplicitTerminator<"DoneOp">]> {
  let summary = "FIRRTL Circuit";
  let description = [{
    The "firrtl.circuit" operation represents an overall Verilog circuit,
    containing a list of modules.
  }];
  let arguments = (ins StrAttr:$name);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringAttr name">
  ];

  let extraClassDeclaration = [{
    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder(&bodyBlock, std::prev(bodyBlock.end()));
    }
  }];
  
  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def FModuleOp : FIRRTLOp<"module",
      [IsolatedFromAbove, FunctionLike, Symbol,
       SingleBlockImplicitTerminator<"DoneOp">]> {
  let summary = "FIRRTL Module";
  let description = [{
    The "firrtl.module" operation represents a Verilog module, including a given
    name, a list of ports, and a body that represents the connections within
    the module.
  }];
  let arguments = (ins);
  let results = (outs);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, StringAttr name, "
              "ArrayRef<std::pair<StringAttr, FIRRTLType>> ports">
  ];

  let extraClassDeclaration = [{
    typedef std::pair<StringAttr, FIRRTLType> PortInfo;
    OpBuilder getBodyBuilder() {
      assert(!body().empty() && "Unexpected empty 'body' region.");
      Block &bodyBlock = body().front();
      return OpBuilder(&bodyBlock, std::prev(bodyBlock.end()));
    }
  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FModuleOp>;

    /// Returns the number of arguments, implementing OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }
    /// Returns the number of results, implementing OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type.  Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    ///  safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  public:
  }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def DoneOp : FIRRTLOp<"done", [Terminator]> {
  let summary = "FIRRTL termination operation";
  let description = [{
    "firrtl.done" marks the end of a region in the FIRRTL dialect.
  }];

  let arguments = (ins);
}

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

def FIRRTLWireOp : FIRRTLOp<"wire", [NoSideEffect]> {
  let summary = "Define a new wire";
  let description = [{
    Declare a new wire:
    ```
      %name = firrt.wire : t1
    ```
    }];

  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
     attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

def FIRRTLConnectOp : FIRRTLOp<"connect", []> {
  let summary = "Connect two signals";
  let description = [{
    Connect Operation:
    ```
      firrt.connect %lhs, %rhs : t1, t2
    ```
    }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs);
  
  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:` type($lhs) `,` type($rhs)
  }];
}

def FIRRTLInvalid : FIRRTLOp<"invalid", [NoSideEffect]> {
  let summary = "Invalid operation";
  let description = [{
    Invalidate Operation:
    ```
       %result = firrt.invalid : i8
    ```

    The invalid operation produces a value of unspecified value.  Drivers may
    be attached to it with `firrtl.connect`.
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  // TODO: Print '$name' more nicely.  This breaks the asmparser tho.
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Primitive Operations
//===----------------------------------------------------------------------===//

// TODO: Should this be split out to addu/adds?
def FIRRTLAddOp : FIRRTLOp<"add", [Commutative, NoSideEffect]> {
  let summary = "Add operations";
  let description = [{
    Add Operation:
    ```
       %result = firrt.add %lhs, %rhs "name" : (t1, t2) -> t3
    ```

    The two options are:
    * add(UInt, UInt) -> UInt
    * add(SInt, SInt) -> SInt
    
    The result width is max(we1, we2)+1
    The add operation result is the sum of e1 and e2 without loss of precision.
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs, OptionalAttr<StrAttr>:$name);
  let results = (outs AnyType:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:`
       `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

#endif // FIRRTL_OPS
