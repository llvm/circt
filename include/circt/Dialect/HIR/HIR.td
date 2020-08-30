#ifndef HIR_OPS
#define HIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// Dialect
def HIRR_Dialect : Dialect {
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

def ValType : Type<CPred<"$_self.isa<ValType>() || $_self.isa<ConstType>()">
, "HIR int type">,
  BuildableType<"ValType::get($_builder.getContext())">;

def MemrefType
: Type<CPred<"$_self.isa<MemrefType>()">,
  "HIR memref type ">,
  BuildableType<"MemrefType::get($_builder.getContext())">;

def WireType
: Type<CPred<"$_self.isa<WireType>()">,
  "HIR wire type ">,
  BuildableType<"WireType::get($_builder.getContext())">;

// Ops
class HIR_Op<string mnemonic, list<OpTrait> traits = []>
: Op<HIRR_Dialect, mnemonic, traits>;

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
    This op duplicates a time var. Semantics are same as using original time var
      but in the hardware a separate path is used to propagate the new time var.

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

def MemReadOp : HIR_Op<"mem_read", [NoSideEffect,AttrSizedOperandSegments]> {
  let summary = "read from a memory interface.";
  let description = [{
    This Op reads from a memory interface. 
      Example:

      ```mlir hir.mem_read %A[%const][%x,%y] at %t 
      : !hir.wire -> !hir.val```
  }];

  let arguments = (ins MemrefType:$mem, 
  Variadic<AnyType>:$addr, TimeType : $tstart,
      Optional<ConstType>: $offset);
  let results = (outs AnyType: $res);
  let assemblyFormat = [{$mem `[`  $addr `]` `at` $tstart (`offset` $offset^)?
    attr-dict `:` type($mem) `[` type($addr) `]` `->` type($res)}];
}

def MemWriteOp : HIR_Op<"mem_write", [AttrSizedOperandSegments]> {
  let summary = "Write to a memory interface.";
  let description = [{
    This Op writes to memory. 
      Example:

      ```mlir hir.mem_write %v to %A[%const][%x,%y] at %t 
      : (!hir.val, !hir.memref)```
  }];

  let arguments = (ins AnyType:$value, MemrefType:$mem,
      Variadic<AnyType>:$addr, 
      TimeType :$tstart, Optional<ConstType>:$offset);
  let results = (outs );
  let assemblyFormat = [{$value `to` $mem  `[` $addr `]` `at` $tstart 
    (`offset` $offset^)? attr-dict `:` `(`type($value) `,` type($mem)`[`
    type($addr) `]` `)`}];
}

def WireReadOp : HIR_Op<"wire_read", [NoSideEffect,AttrSizedOperandSegments]> {
  let summary = "Read from a wire.";
  let description = [{
    This Op reads from wire. 
      Example:

      ```mlir hir.wire_read %A[%x] at %t 
      : !hir.wire -> !hir.val```
  }];

  let arguments = (ins WireType:$wire, Variadic<ConstType>:$addr, 
      TimeType : $tstart, Optional<ConstType>: $offset);

  let results = (outs AnyType: $res);

  let assemblyFormat = [{$wire ( `[`$addr^`]` )? `at` $tstart (`offset` $offset^)?
    attr-dict `:` type($wire) `->` type($res)}];
}

def WireWriteOp : HIR_Op<"wire_write", [AttrSizedOperandSegments]> {
  let summary = "Write to a wire";
  let description = [{
    This Op writes to a wire. 
      Example:

      ```mlir hir.wire_write %v to %A at %t 
      : (!hir.val, !hir.wire)```
  }];

  let arguments = (ins AnyType:$value, WireType:$wire, Variadic<ConstType>:$addr,
      TimeType :$tstart, Optional<ConstType>:$offset);

  let results = (outs );

  let assemblyFormat = [{$value `to` $wire ( `[`$addr^`]` )? `at` $tstart 
    (`offset` $offset^)? attr-dict `:` `(`type($value) `,` type($wire)`)`}];
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
  let description = [{```mlir %yy = hir.delay %xx by %1 at %tj```}];

  let arguments = (ins ValType:$input, ConstType:$delay, 
      TimeType:$tstart, Optional<ConstType>: $offset);
  let results = (outs ValType:$res);
  let assemblyFormat = [{$input `by` $delay `at` $tstart (`offset` $offset^)? 
    attr-dict `:` type($input)}];
}

def ForOp : HIR_Op<"for", 
    [ImplicitTerminator, RecursiveSideEffects,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
      let summary = "for operation";
      let description = [{
        This op implements a sequential for loop.
          ```mlir
          %tdone=hir.for %i = 0 to %N step 1 at iter_time(%ti = %t tstep 2) {
            hir.for %j = %lb to %ub step %s at iter_time(%tj = %ti) 
              iter_args(%xx = %x at %tj){   
                %yy = hir.delay %xx by 1 at %tj
                  hir.yield %yy at %tj+1;
              }
          }
        ```
      }];

      let arguments = (ins ValType: $lb, ValType: $ub, ValType: $step, 
          TimeType:$tstart, Optional<ValType>: $tstep);
      let regions = (region SizedRegion<1>:$region);
      let results = (outs TimeType:$tlast);

      let extraClassDeclaration = [{
        Value getInductionVar() { return getBody()->getArgument(0); }
        Value getIterTimeVar() { return getBody()->getArgument(1); }
      }];
      let parser = [{return parse$cppClass(parser,result);}];
      let printer = [{return print$cppClass(p,*this);}];
    }

def UnrollForOp : HIR_Op<"unroll_for", 
    [ImplicitTerminator, RecursiveSideEffects,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
      let summary = "for operation";
      let description = [{
        This op implements a sequential for loop.
          ```mlir
          %tdone=hir.for %i = 0 to %N step 1 at iter_time(%ti = %t tstep 2) {
          }
      }
      ```
    }];

let arguments = (ins I32Attr: $lb, I32Attr: $ub, I32Attr: $step, 
    TimeType:$tstart, OptionalAttr<I32Attr>: $tstep);
let regions = (region SizedRegion<1>:$region);
let results = (outs TimeType:$tlast);

let extraClassDeclaration = [{
  Value getInductionVar() { return getBody()->getArgument(0); }
  Value getIterTimeVar() { return getBody()->getArgument(1); }
}];
let parser = [{return parse$cppClass(parser,result);}];
let printer = [{return print$cppClass(p,*this);}];
}

def CallOp : HIR_Op<"call",[AttrSizedOperandSegments] > {
  let summary = "Function call";
  let description = [{
    This Op calls a function/module. 
      Example:

      ```mlir %a,%b = hir.call @foo (%x,%A) at %ti
      : (!hir.val,!hir.memref) -> (!hir.val,!hir.val)```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands,
      TimeType:$tstart, Optional<ConstType>: $offset);
  let results = (outs Variadic<AnyType> : $res);

  let assemblyFormat = [{
    $callee `(` $operands `)` `at` $tstart (`offset` $offset^)? attr-dict 
      `:` functional-type($operands, $res)
  }];
}

def DefOp : HIR_Op<"def",
[DeclareOpInterfaceMethods<CallableOpInterface>, FunctionLike,IsolatedFromAbove,Symbol] > {
  let summary = "Function definition";
  let description = [{
    This Op creates a new function 
      Example:

      ```mlir hir.func @foo (%x, %A) at %t 
      : (!hir.val,!hir.memref) -> (!hir.val,!hir.val)```
  }];

  let arguments = (ins
    TypeAttr:$type,
    StrAttr:$sym_name,
    OptionalAttr<I32ArrayAttr> : $time_domains, 
    OptionalAttr<I32ArrayAttr>: $time_offsets);

  let regions = (region SizedRegion<1>:$region);
  let results = (outs );
  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  let extraClassDeclaration = [{
  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<DefOp>;

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

      ```mlir hir.yield (%x, %A) at %t offset %3
      : (!hir.val,!hir.val)```
  }];

  let arguments = (ins Variadic<AnyType> : $operands, TimeType: $tstart,
      Optional<ConstType>:$offset);
  let results = (outs );

  let assemblyFormat = [{ attr-dict (`(` $operands^ `)` `:` type($operands))? 
    `at` $tstart (`offset` $offset^)?}];
}

def ConstantOp : HIR_Op<"constant" > {
  let summary = "Get an ssa var with a constant value";
  let description = [{
    This Op calls a function/module. 
      Example:

      ```mlir %0 = hir.const 0 at %ti : !hir.val```
  }];

  let arguments = (ins I32Attr:$value);
  let results = (outs ConstType : $res);

  let assemblyFormat = [{ $value attr-dict `:` type($res)}];
}

def CastOp : HIR_Op<"cast">{
  let summary = "Cast a value to another type";
  let description = [{
    This Op casts one ssa value to another type
      Example:

      ```mlir %1 = hir.cast %0 at %ti : !hir.const -> !hir.val```
  }];

  let arguments = (ins AnyType:$input, TimeType: $tstart,
      Optional<ConstType>:$offset);
  let results = (outs AnyType : $res);

  let assemblyFormat = [{ $input  `at` $tstart (`offset` $offset^)? attr-dict
    `:` type($input) `->` type($res)}];

}

def AddOp : HIR_Op<"add",[NoSideEffect]>{
  let summary = "Add two hir.val values";
  let description = [{
      Example:

      ```mlir %1 = hir.add (%x,1 : !hir.const -> !hir.val```
  }];

  let arguments = (ins ValType: $left, ValType:$right, TimeType: $tstart,
      Optional<ConstType>:$offset);
  let results = (outs ValType : $res);

  let assemblyFormat = [{ `at` $tstart (`offset` $offset^)? `(` $left `,` $right`)` attr-dict}];

}

def ConstAddOp : HIR_Op<"const_add",[NoSideEffect]>{
  let summary = "Add two hir.const values";
  let description = [{
      Example:

      ```mlir %1 = hir.add (%x,%y) : !hir.const -> !hir.val```
  }];

  let arguments = (ins ConstType: $left, ConstType:$right);
  let results = (outs ConstType : $res);

  let assemblyFormat = [{ `(` $left `,` $right`)` attr-dict}];

}

def ReturnOp : HIR_Op<"return", [HasParent<"DefOp">,NoSideEffect,
                                     Terminator]> {
  let summary = "Return from function";

  let description = [{
    This instruction must be the last instruction in a block.
    ```mlir hir.return (%x, %y): (!hir.val, !hir.val)```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "(`(`$operands^`)` `:` `(`type($operands)`)`)? attr-dict";
}

def NewWireOp :HIR_Op<"wire", [NoSideEffect]> {
  let summary = "Create new wire";

  let description = [{

    ```mlir %w = hir.wire```
  }];

  let arguments = (ins);
  let results = (outs WireType:$res);

  let assemblyFormat = "attr-dict";
}

#endif // HIR_OPS
