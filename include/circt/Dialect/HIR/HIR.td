#ifndef HIR_OPS
#define HIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

def MemoryInterfaceType
    : Type<CPred<"$_self.isa<MemoryInterfaceType>()">,
           "HIR memory interface type ">,
      BuildableType<"MemoryInterfaceType::get($_builder.getContext())">;

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

def MemReadOp : HIR_Op<"mem_read", [NoSideEffect]> {
  let summary = "read from a memory interface.";
  let description = [{
    This Op reads from a memory interface. 
    Example:

        ```mlir hir.mem_read %A[%x] at %t 
        : !hir.mem_interface -> !hir.int```
  }];

  let arguments = (ins MemoryInterfaceType:$mem, I32:$addr, TimeType : $tstart,
  OptionalAttr<I32Attr>: $delay);
  let results = (outs AnyType: $res);
  let assemblyFormat = [{$mem `[` $addr `]` `at` $tstart (`delay` $delay^)?
    attr-dict `:` type($mem) `->` type($res)}];
}

def MemWriteOp : HIR_Op<"mem_write", []> {
  let summary = "read from a memory interface.";
  let description = [{
    This Op reads from a memory interface. 
    Example:

        ```mlir hir.mem_write %v to %A[%x] at %t 
        : (!hir.int, !hir.mem_interface)```
  }];

  let arguments = (ins AnyType:$value, MemoryInterfaceType:$mem, I32:$addr, 
  TimeType :$tstart, OptionalAttr<I32Attr>:$delay);
  let results = (outs );
  let assemblyFormat = [{$value `to` $mem `[` $addr `]` `at` $tstart 
  (`delay` $delay^)? attr-dict `:` `(`type($value) `,` type($mem)`)`}];
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

def ForOp : HIR_Op<"for", 
    [ImplicitTerminator, RecursiveSideEffects,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "for operation";
  let description = [{
    This op implements a sequential for loop.
        ```mlir
     hir.for %i = 0 to %N step 1 at iter_time(%ti = %t tstep 2) {
        hir.for %j = %lb to %ub step %s at iter_time(%tj = %ti) 
          iter_args(%xx = %x at %tj){   
            %yy = hir.delay %xx by 1 at %tj
            hir.yield %yy at %tj+1;
        }
     }
    ```
  }];

  let arguments = (ins AnyType: $lb, AnyType: $ub, AnyType: $step, 
    TimeType:$tstart, Optional<AnyType>: $tstep);
  let regions = (region SizedRegion<1>:$region);

  let extraClassDeclaration = [{
    Value getInductionVar() { return getBody()->getArgument(0); }
    Value getIterTimeVar() { return getBody()->getArgument(1); }
  }];
  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
}

def CallOp : HIR_Op<"call" > {
  let summary = "Function call";
  let description = [{
    This Op calls a function/module. 
    Example:

        ```mlir %a,%b = hir.call @foo (%x,%A) at %ti
        : (!hir.int,!hir.mem_interface) -> (!hir.int,!hir.int)```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands,
    TimeType:$tstart, OptionalAttr<I32Attr>: $delay);
  let results = (outs Variadic<AnyType> : $res);

  let assemblyFormat = [{
    $callee `(` $operands `)` `at` $tstart (`delay` $delay^)? attr-dict `:` functional-type($operands, $res)
  }];

}

def DefOp : HIR_Op<"def", [ImplicitTerminator] > {
  let summary = "Function definition";
  let description = [{
    This Op creates a new function 
    Example:

        ```mlir hir.func @foo (%x at %t, %A) at %t 
        : (!hir.int,!hir.mem_interface) -> (!hir.int,!hir.int)```
  }];

  let arguments = (ins I32ArrayAttr : $time_domains, I32ArrayAttr: $time_offsets);
  let regions = (region SizedRegion<1>:$region);
  let results = (outs );
  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
}
#endif // HIR_OPS
