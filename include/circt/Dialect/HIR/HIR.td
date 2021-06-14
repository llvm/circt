#ifndef HIR_OPS
#define HIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "circt/Dialect/HIR/HIRInterfaces.td"

// Dialect
def HIR_Dialect : Dialect {
  let name = "hir";
  let summary = "An MLIR dialect for Hardware acclerator description";
  let description = [{
    This dialect is for description of hardware accelerators with focus on
      FPGAs.
  }];
  let cppNamespace = "hir";
}

// Types
def TimeType : Type<CPred<"$_self.isa<TimeType>()">, "HIR time type">,
    BuildableType<"TimeType::get($_builder.getContext())">;

def MemrefType: Type<CPred<"$_self.isa<MemrefType>()">,
    "HIR memref type ">,
    BuildableType<"MemrefType::get($_builder.getContext())">;

def FuncType: Type<CPred<"$_self.isa<FuncType>()">,
    "HIR interface type ">,
    BuildableType<"FuncType::get($_builder.getContext())">;

// Ops
class HIR_Op<string mnemonic, list<OpTrait> traits = []>
: Op<HIR_Dialect, mnemonic, traits>;

def DefTimeVarOp : HIR_Op<"def_time_var", [NoSideEffect]> {
  let summary = "Op defines a new time var.";
  let description = [{
    This op defines a new time variable. 

      Example:

      ```mlir %t = hir.def_time_var: !hir.time```
  }];

  let arguments = (ins);
  let results = (outs TimeType : $res);

  let assemblyFormat = [{attr-dict `:` type($res)}];
}

def DuplicateTimeVarOp : HIR_Op<"duplicate_time_var", [NoSideEffect]> {
  let summary = "Op duplicates a time var.";
  let description = [{
    This op duplicates a time var. Semantics are the same as using original time 
      var but in the hardware a separate path is used to propagate the new var.

      Example:

      ```mlir % t2 = hir.duplicate_time_var %t1 : !hir.time```
  }];

  let arguments = (ins TimeType : $input);
  let results = (outs TimeType : $res);

  let assemblyFormat = [{$input attr-dict `:` type($res)}];
}

def SyncTimeOp : HIR_Op<"sync_time", [NoSideEffect]> {
  let summary = "synchronize two time domains.";
  let description = [{
    This op synchronizes two time domains. Until the specified instant does not
      occur in both time zones, the time zones will wait.

      Example:

      ```mlir hir.sync_time(% t1, % t2): (!hir.time, !hir.time)```
  }];

  let arguments = (ins TimeType : $inp1, TimeType : $inp2);
  let results = (outs);

  let assemblyFormat = [{`(`$inp1 `,` $inp2 `)` attr-dict `:` `(` type($inp1)
      `,` type($inp2) `)`}];
}

def LoadOp : HIR_Op<"load", [NoSideEffect,AttrSizedOperandSegments]> {
  let summary = "read from a memory.";
  let description = [{
    This Op reads from a memory interface. 
      Indices for banks should be of index type and indices for address should be
      integers of correct width (based on the size of the dimension).

      Example:

      ```mlir hir.load %A[%x,%y] at %t : !hir.memref<...>```
  }];

  let arguments = (ins MemrefType:$mem, 
      Variadic<AnyType>:$indices, TimeType : $tstart,
      Optional<Index>: $offset);
  let results = (outs AnyType: $res);
  let assemblyFormat = [{
    $mem `[`  $indices `]` `at` $tstart (`+` $offset^)? 
      attr-dict
      `:`custom<MemrefAndElementType>
      (type($mem),type($indices),type($res)) 
  }];
  let hasCanonicalizeMethod = true;
  let extraClassDeclaration = [{
    public:
      SmallVector<Value> getBankIdx();
      SmallVector<Value> getAddrIdx();
  }];
}

def StoreOp : HIR_Op<"store", [AttrSizedOperandSegments]> {
  let summary = "Write to a memory.";
  let description = [{
    This Op writes to memory. 
      Indices for banks should be of index types and indices for address should be
      integers of correct width (based on the size of the dimension).

      Example:

      ```mlir hir.mem_write %v to %A[%x,%y] at %t + %1 : !hir.memref<...>```
  }];

  let arguments = (ins AnyType:$value, MemrefType:$mem,
      Variadic<AnyType>:$indices, 
      TimeType :$tstart, Optional<Index>:$offset);
  let results = (outs );
  let assemblyFormat = [{
    $value `to` $mem  `[` $indices `]` `at` $tstart (`+` $offset^ )? attr-dict `:` 
      custom<MemrefAndElementType>
      (type($mem),type($indices), type($value))
  }];

  let hasCanonicalizeMethod = true;
  let extraClassDeclaration = [{
    public:
      SmallVector<Value> getBankIdx();
      SmallVector<Value> getAddrIdx();
  }];
}

def SendOp : HIR_Op<"send", []> {
  let summary = "write to an interface.";
  let description = [{
    This Op writes to a bus. 
    Example:
      ```mlir hir.send %1 to %v[%0] : !hir.bus<...>```
  }];

  let arguments = (ins AnyType:$value, AnyType:$bus,
      Index:$index, 
      TimeType :$tstart, Optional<Index>:$offset);
  let results = (outs );
  let assemblyFormat = [{
    $value `to` $bus  `[` $index `]` `at` $tstart (`+` $offset^ )? attr-dict 
      `:` type($value) `to` type($bus)  
  }];
  let hasCanonicalizeMethod = true;
}

def RecvOp : HIR_Op<"recv", []> {
  let summary = "read from a group or array.";
  let description = [{
    This Op Reads from a bus. 

    ```mlir %v = hir.recv %x[%0] : !hir.bus<...> -> i32```
  }];

  let arguments = (ins AnyType:$bus,
      Index:$index, 
      TimeType :$tstart, Optional<Index>:$offset);
  let results = (outs AnyType:$res);

  let assemblyFormat = [{
    $bus  `[` $index `]` `at` $tstart (`+` $offset^ )? attr-dict 
      `:` type($bus) `->` type($res)  
  }];
  let hasCanonicalizeMethod = true;
}

def ImplicitTerminator
: SingleBlockImplicitTerminator<"TerminatorOp">;

def TerminatorOp :
HIR_Op<"terminator", [NoSideEffect, Terminator]> {
  let summary = "HIR terminator operation";
  let description = [{
    This is the terminator op for loop and if statement regions
      Syntax:
      ```operation ::= `"hir.terminator"() : () -> ()````
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;

  // Fully specified by traits.
  let verifier = ?;
}

def DelayOp: HIR_Op<"delay",[NoSideEffect]>{
  let summary = "Delay operation";
  let description = [{
Example:
    ```mlir %yy = hir.delay %xx by %1 at %tj```
  }];

  let arguments = (ins AnyType : $input, Index:$delay, 
      TimeType:$tstart, Optional<Index>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{$input `by` $delay `at` 
    $tstart (`+` $offset^ )? attr-dict `:` type($input) `->` type($res)}];
}

//def WhileOp : HIR_Op<"while", 
//    [ImplicitTerminator, RecursiveSideEffects]> {
//      let summary = "While loop";
//      let description = [{
//        This op implements while loop.
//          ```mlir
//          hir.while (%cond) at %t {
//            ...
//          }else{
//            ...
//          }
//        ```
//      }];
//
//      let arguments = (ins AnyType: $cond, TimeType:$tstart);
//      let regions = (region SizedRegion<1>:$if_body, SizedRegion<1>:$else_body);
//      let results = (outs );
//
//      //let extraClassDeclaration = [{
//      //  Value getIfBody() { return TODO; }
//      //  Value getElseBody() { return TODO; }
//      //}];
//
//      let parser = [{return parse$cppClass(parser,result);}];
//      let printer = [{return print$cppClass(p,*this);}];
//  }


def IfOp : HIR_Op<"if", [ImplicitTerminator, RecursiveSideEffects]> {
  let summary = "if operation";
  let description = [{
    This op implements an if statement.
      ```mlir
      hir.if(%b) at %t {...}        ```
  }];

  let arguments = (ins AnyType : $cond, TimeType : $tstart);
  let regions = (region SizedRegion<1>:$if_region);
  let results = (outs );

  let extraClassDeclaration = [{
  }];
  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
}

def ForOp : HIR_Op<"for", 
    [ImplicitTerminator, RecursiveSideEffects,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
      let summary = "for operation";
      let description = [{
        This op implements a sequential for loop.
          ```mlir
          hir.for %i = 0 to %N step 1 at iter_time(%ti = %t offset %1) {
            hir.for %j = %lb to %ub step %s at iter_time(%tj = %ti offset %1) 
              iter_args(%xx = %x at %tj){   
                %yy = hir.delay %xx by 1 at %tj
                  hir.yield %yy at %tj offset 1;
              }
            hir.yield at %ti offset 1;
          }
        ```
      }];

      let arguments = (ins AnyType: $lb, AnyType: $ub, AnyType: $step, 
          TimeType:$tstart, Index : $offset);
      let regions = (region SizedRegion<1>:$region);
      let results = (outs TimeType:$tfinish);

      let parser = [{return parse$cppClass(parser,result);}];
      let printer = [{return print$cppClass(p,*this);}];
      let builders = [OpBuilder<(ins "Value":$lb, "Value":$ub, "Value":$step,
          "Value":$tstart, "Value":$offset,"Type":$inductionVarTy),[{
        build($_builder,$_state,hir::TimeType::get($_builder.getContext()),lb,ub,step,tstart,offset);

      }]>];
      let extraClassDeclaration = [{
        Value getInductionVar() { return getBody()->getArgument(0); }
        Value getIterTimeVar() { return getBody()->getArgument(1); }
        Block* addEntryBlock(MLIRContext *context, Type inductionVarTy);
        void beginRegion(mlir::OpBuilder &builder);        
        void endRegion(mlir::OpBuilder &builder);
      }];
      let hasCanonicalizeMethod = true; 
    }

//RecursiveSideEffects,
def UnrollForOp : HIR_Op<"unroll_for", 
    [ImplicitTerminator, 
    DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
      let summary = "for operation";
      let description = [{
        This op implements a sequential for loop.
          Example:
          ```
          mlir %tdone=hir.for %i = 0 to %N step 1 at iter_time(%ti = %t)
          {}
        ```
      }];

      let arguments = (ins I32Attr: $lb, I32Attr: $ub, I32Attr: $step, 
          TimeType:$tstart, Optional<Index>:$offset);
      let regions = (region SizedRegion<1>:$region);
      let results = (outs TimeType:$tlast);

      let extraClassDeclaration = [{
        Value getInductionVar() { return getBody()->getArgument(0); }
        Value getIterTimeVar() { return getBody()->getArgument(1); }
        Block* addEntryBlock(MLIRContext *context);
        void beginRegion(mlir::OpBuilder &builder);        
        void endRegion(mlir::OpBuilder &builder);
      }];
      let parser = [{return parse$cppClass(parser,result);}];
      let printer = [{return print$cppClass(p,*this);}];
      let hasCanonicalizeMethod = true;
    }

def CallOp : HIR_Op<"call",[AttrSizedOperandSegments] > {
  let summary = "Function call";
  let description = [{
    This Op calls a function/module. 
      Example:

      ```mlir 
      %a,%b = hir.call @foo (%x,%A) at %ti : (!hir.val,!hir.memref) -> (!hir.val,!hir.val)
      ```
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee, 
      TypeAttr : $funcTy,
      Optional<AnyType>:$callee_var, 
      Variadic<AnyType>:$operands, 
      TimeType:$tstart, 
      Optional<Index>: $offset
      );
  let results = (outs Variadic<AnyType> : $res);

  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  let hasCanonicalizeMethod = true;
  let extraClassDeclaration=[{
    public:
      ///Get the function arguments (excluding tstart).
      SmallVector<Value,4> getOperands();
  }];
}

def FuncOp : HIR_Op<"func", [
  Symbol, 
  FunctionLike, 
  IsolatedFromAbove, 
  DeclareOpInterfaceMethods<CallableOpInterface>
]> {
  let summary = "Function definition";
  let description = [{
    This Op creates a new function 
      Example:

      ```mlir 
      hir.func @foo (%x, %A) at %t : (!hir.val,!hir.memref) -> (!hir.val,!hir.val)
      ```
  }];

  let arguments = (ins
      TypeAttr:$type,
      StrAttr:$sym_name,
      TypeAttr:$funcTy 
      );

  let regions = (region SizedRegion<1>:$region);
  let results = (outs );
  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  let extraClassDeclaration = [{
    private:
      // This trait needs access to the hooks defined below.
      friend class OpTrait::FunctionLike<FuncOp>;

      /// Returns the number of arguments. Hook for OpTrait::FunctionLike.
      unsigned getNumFuncArguments() { return getType().getNumInputs(); }

      /// Returns the number of results. Hook for OpTrait::FunctionLike.
      unsigned getNumFuncResults() { return getType().getNumResults(); }

      /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
      /// attribute is present and checks if it holds a function type. Ensures
      /// getType, getNumFuncArguments, and getNumFuncResults can be called safely
      LogicalResult verifyType();
      Region& body(){ return getOperation()->getRegion(0);}

      /// Hook for OpTrait::FunctionLike, called after verifying the function
      /// type and the presence of the (potentially empty) function body.
      LogicalResult verifyBody();

    public:
      ///Get the function arguments (excluding tstart).
      SmallVector<Value,4> getOperands();
  }];
  let verifier =[{return verifyFuncOp(*this);}];
}

def YieldOp : HIR_Op<"yield", [AttrSizedOperandSegments] > {
  let summary = "Yield from loops";
  let description = [{
    This Op yields from a loop.
      Example:

      ```mlir hir.yield (%x, %A) at %t offset %3 : (!hir.val,!hir.val)```
  }];

  let arguments = (ins Variadic<AnyType> : $operands, TimeType: $tstart,
      Optional<Index>:$offset);
  let results = (outs );

  let assemblyFormat = [{ attr-dict (`(` $operands^ `)` `:` type($operands))? 
    `at` $tstart (`+` $offset^ )?}];
  let hasCanonicalizeMethod = true;
}

def ConstantOp : HIR_Op<"constant",[NoSideEffect]> {
  let summary = "Get an ssa var with a constant value";
  let description = [{
    This Op calls a function/module. 
      Example:

      ```mlir %0 = hir.const 0 : !hir.const```
      ```mlir %f = hir.const 3.18 : !hir.group<f32>```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType : $res);

  let assemblyFormat = [{  `(` $value `)` attr-dict `:` type($res)}];
}

def CastOp : HIR_Op<"cast">{
  let summary = "Cast a value to another type";
  let description = [{
    This Op casts value of one type to another type
      Example:

      ```mlir %1 = hir.cast %0: i32 -> i4```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType : $res);

  let assemblyFormat = [{ $input  attr-dict `:` type($input) `->` type($res)}];

}

def ReturnOp : HIR_Op<"return", [HasParent<"FuncOp">,NoSideEffect,
    Terminator]> {
      let summary = "Return from function";

      let description = [{
        This instruction must be the last instruction in a block.
          ```mlir hir.return (%x, %y): (i32, !hir.const<i1>)```
      }];

      let arguments = (ins Variadic<AnyType>:$operands);

      let assemblyFormat = "(`(`$operands^`)` `:` `(`type($operands)`)`)? attr-dict";
    }

def AllocaOp :HIR_Op<"alloca",[NoSideEffect]> {
  let summary = "Instantiate new hardware.";

  let description = [{

    ```%w = hir.alloc("bram") : !hir.memref<...>```
  }];

  let arguments = (ins AnyAttr:$moduleAttr);
  let results = (outs Variadic<AnyType>:$res);

  let assemblyFormat = [{`(`$moduleAttr`)`attr-dict 
    `:` type($res)}];
}

def TupleOp:HIR_Op<"tuple",[NoSideEffect]>{
  let summary = "creates a tuple from inputs";
  let description = [{
Examples:
    ```mlir %1 = hir.tuple (%x, %y) : (i1, i32) -> Tuple<i1,i32>```
  }];

  let arguments = (ins Variadic<AnyType>: $args);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ `(` $args`)` attr-dict `:` `(`
      type($args) `)` `->` type($res) }];
}

def SplitOp:HIR_Op<"split",[NoSideEffect]>{
  let summary = "splits a bus into two";
  let description = [{
Examples:
    ```mlir %b2 = hir.split (%b[%i]) : Tensor<!hir.bus<...>> -> !hir.bus<...>```
  }];

  let arguments = (ins AnyType: $bus, Variadic<Index>: $indices);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ $bus (`[` $indices^ `]`)? attr-dict `:` 
    type($bus)  `->` type($res) }];
}
#endif // HIR_OPS
