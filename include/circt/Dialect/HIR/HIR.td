#ifndef HIR_OPS
#define HIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

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

def ConstType: Type<CPred<"$_self.isa<ConstType>()">
, "HIR const type">,
  BuildableType<"ConstType::get($_builder.getContext())">;

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
      Example:

      ```mlir hir.load %A[%x,%y] at %t : !hir.memref<...>[i32,i32] -> i32```
  }];

  let arguments = (ins MemrefType:$mem, 
      Variadic<AnyType>:$addr, TimeType : $tstart,
      Optional<ConstType>: $offset);
  let results = (outs AnyType: $res);
  let assemblyFormat = [{$mem `[`  $addr `]` `at` $tstart (`+` $offset^)? 
    attr-dict `:` type($mem) `[` type($addr) `]` `->` type($res)}];
  let hasCanonicalizeMethod = true;
}

def StoreOp : HIR_Op<"store", [AttrSizedOperandSegments]> {
  let summary = "Write to a memory.";
  let description = [{
    This Op writes to memory. 
      Example:

      ```
      mlir hir.mem_write %v to %A[%x,%y] at %t 
        : (i32, !hir.memref<...>[i32,i32])
      ```
  }];

  let arguments = (ins AnyType:$value, MemrefType:$mem,
      Variadic<AnyType>:$addr, 
      TimeType :$tstart, Optional<ConstType>:$offset);
  let results = (outs );
  let assemblyFormat = [{$value `to` $mem  `[` $addr `]` `at` $tstart 
    (`+` $offset^ )? attr-dict 
      `:` `(`type($value) `,` type($mem)`[`type($addr) `]` `)`}];
  let hasCanonicalizeMethod = true;
}

def SendOp : HIR_Op<"send", [AttrSizedOperandSegments]> {
  let summary = "write to an interface.";
  let description = [{
    This Op writes to group or array. 
      ```
  }];

  let arguments = (ins AnyType:$value, AnyType:$var,
      Variadic<AnyType>:$addr, 
      TimeType :$tstart, Optional<ConstType>:$offset);
  let results = (outs );
  let assemblyFormat = [{$value `to` $var  `[` $addr `]` `at` $tstart 
    (`+` $offset^ )? attr-dict 
      `:` type($value) `to` type($var)`[`type($addr) `]` }];
  let hasCanonicalizeMethod = true;
}

def RecvOp : HIR_Op<"recv", [AttrSizedOperandSegments]> {
  let summary = "read from a group or array.";
  let description = [{
    This Op Reads from group or array. 
      ```
  }];

  let arguments = (ins AnyType:$var,
      Variadic<AnyType>:$addr, 
      TimeType :$tstart, Optional<ConstType>:$offset);
  let results = (outs AnyType:$res);

  let assemblyFormat = [{$var  `[` $addr `]` `at` $tstart 
    (`+` $offset^ )? attr-dict 
      `:` type($var)`[`type($addr) `]` `->` type($res) }];
  let hasCanonicalizeMethod = true;
}

def SelectOp : HIR_Op<"select", [NoSideEffect]> {
  let summary = "Select a select of the interface.";
  let description = [{
    This Op selects from hir.array or hir.group. 
  }];

  let arguments = (ins AnyType:$var, 
      Variadic<AnyType>:$addr);
  let results = (outs AnyType: $res);
  let assemblyFormat = [{$var
  `[` $addr `]` attr-dict `:` type($var) `[` type($addr) `]` `->` type($res) }];

  let verifier = [{return verifySelectOp(*this);}];
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

  let arguments = (ins AnyType : $input, ConstType:$delay, 
      TimeType:$tstart, Optional<ConstType>: $offset);
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
          TimeType:$tstart, ConstType : $offset);
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
          TimeType:$tstart, Optional<ConstType>:$offset);
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
      Optional<ConstType>: $offset
      );
  let results = (outs Variadic<AnyType> : $res);

  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  let hasCanonicalizeMethod = true;
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
      }];
    }

def YieldOp : HIR_Op<"yield", [AttrSizedOperandSegments] > {
  let summary = "Yield from loops";
  let description = [{
    This Op yields from a loop.
      Example:

      ```mlir hir.yield (%x, %A) at %t offset %3 : (!hir.val,!hir.val)```
  }];

  let arguments = (ins Variadic<AnyType> : $operands, TimeType: $tstart,
      Optional<ConstType>:$offset);
  let results = (outs );

  let assemblyFormat = [{ attr-dict (`(` $operands^ `)` `:` type($operands))? 
    `at` $tstart (`+` $offset^ )?}];
  let hasCanonicalizeMethod = true;
}

def ConstantOp : HIR_Op<"constant" > {
  let summary = "Get an ssa var with a constant value";
  let description = [{
    This Op calls a function/module. 
      Example:

      ```mlir %0 = hir.const 0 : !hir.const```
      ```mlir %f = hir.const 3.18 : !hir.group<f32>```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType : $res);

  let assemblyFormat = [{ `(` $value `)` attr-dict `:` type($res)}];
}

def CastOp : HIR_Op<"cast">{
  let summary = "Cast a value to another type";
  let description = [{
    This Op casts one ssa value to another type
      Example:

      ```mlir %1 = hir.cast %0 at %ti : !hir.const -> !hir.val```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType : $res);

  let assemblyFormat = [{ $input  attr-dict
    `:` type($input) `->` type($res)}];

}

def AddOp : HIR_Op<"add",[NoSideEffect]>{
  let summary = "Add two hir.val values";
  let description = [{
Examples:
    ```mlir %1 = hir.add (%x, %y) : (i32, i32) -> (i32)```
    ```mlir %1 = hir.add (%x, %y) : (!hir.const<i32>, !hir.conts<i32>) -> (!hir.const<i32>)```
  }];

  let arguments = (ins AnyType: $left, AnyType:$right);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ `(` $left `,` $right`)` attr-dict `:` `(`
      type($left) `,` type($right) `)` `->` `(` type($res) `)`}];
}

def SubtractOp : HIR_Op<"subtract",[NoSideEffect]>{
  let summary = "Subtract two hir.val values";
  let description = [{
Examples:
    ```mlir %1 = hir.subtract (%x, %y) : (i32, i32) -> (i32)```
    ```mlir %1 = hir.subtract (%x, %y) : (!hir.const<i32>, !hir.conts<i32>) -> (!hir.const<i32>)```
  }];

  let arguments = (ins AnyType: $left, AnyType:$right);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ `(` $left `,` $right`)` attr-dict `:` `(`
      type($left) `,` type($right) `)` `->` `(` type($res) `)`}];
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

def AllocaOp :HIR_Op<"alloca", [NoSideEffect]> {
  let summary = "Instantiate new hardware.";

  let description = [{

    ```%w = hir.alloc("bram") : !hir.memref<...>```
  }];

  let arguments = (ins AnyAttr:$moduleAttr);
  let results = (outs Variadic<AnyType>:$res);

  let assemblyFormat = [{`(`$moduleAttr`)`attr-dict 
                        `:` type($res)}];
}

def EQOp : HIR_Op<"eq",[NoSideEffect]>{
  let summary = "returns true if the values are equal";
  let description = [{
Examples:
    ```mlir %1 = hir.eq (%x, %y) : (i32, i32) -> (i1)```
    ```mlir %1 = hir.eq (%x, %y) :  (!hir.const, !hir.const) -> (!hir.const)```
  }];

  let arguments = (ins AnyType: $left, AnyType:$right);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ `(` $left `,` $right`)` attr-dict `:` `(`
      type($left) `,` type($right) `)` `->` `(` type($res) `)`}];
}

def NEQOp : HIR_Op<"neq",[NoSideEffect]>{
  let summary = "returns true if the values are not equal";
  let description = [{
Examples:
    ```mlir %1 = hir.neq (%x, %y) : (i32, i32) -> (i1)```
    ```mlir %1 = hir.neq (%x, %y) :  (!hir.const, !hir.const) -> (!hir.const)```
  }];

  let arguments = (ins AnyType: $left, AnyType:$right);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ `(` $left `,` $right`)` attr-dict `:` `(`
      type($left) `,` type($right) `)` `->` `(` type($res) `)`}];
}

def GTOp : HIR_Op<"gt",[NoSideEffect]>{
  let summary = "returns an i1 to show if one value is greater than or other";
  let description = [{
Examples:
    ```mlir %1 = hir.gt (%x, %y) : (i32, i32) -> (i1)```
    ```mlir %1 = hir.gt (%x, %y) :  (!hir.const, !hir.const) -> (!hir.const)```
  }];

  let arguments = (ins AnyType: $left, AnyType:$right);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ `(` $left `,` $right`)` attr-dict `:` `(`
      type($left) `,` type($right) `)` `->` `(` type($res) `)`}];
}

def LTOp : HIR_Op<"lt",[NoSideEffect]>{
  let summary = "returns an i1 to show if one value is less than or other";
  let description = [{
Examples:
    ```mlir %1 = hir.gt (%x, %y) : (i32, i32) -> (i1)```
    ```mlir %1 = hir.gt (%x, %y) :  (!hir.const, !hir.const) -> (!hir.const)```
  }];

  let arguments = (ins AnyType: $left, AnyType:$right);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ `(` $left `,` $right`)` attr-dict `:` `(`
      type($left) `,` type($right) `)` `->` `(` type($res) `)`}];
}

def AndOp : HIR_Op<"and",[NoSideEffect]>{
  let summary = "returns and of two i1 inputs";
  let description = [{
Examples:
    ```mlir %1 = hir.gt (%x, %y) : (i1, i1) -> (i1)```
    ```mlir %1 = hir.gt (%x, %y) :  (!hir.const, !hir.const) -> (!hir.const)```
  }];

  let arguments = (ins AnyType: $left, AnyType:$right);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ `(` $left `,` $right`)` attr-dict `:` `(`
      type($left) `,` type($right) `)` `->` `(` type($res) `)`}];
}

def OrOp : HIR_Op<"or",[NoSideEffect]>{
  let summary = "returns 'or' of two i1 inputs";
  let description = [{
Examples:
    ```mlir %1 = hir.gt (%x, %y) : (i1, i1) -> (i1)```
    ```mlir %1 = hir.gt (%x, %y) :  (!hir.const, !hir.const) -> (!hir.const)```
  }];

  let arguments = (ins AnyType: $left, AnyType:$right);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ `(` $left `,` $right`)` attr-dict `:` `(`
      type($left) `,` type($right) `)` `->` `(` type($res) `)`}];
}

#endif // HIR_OPS
