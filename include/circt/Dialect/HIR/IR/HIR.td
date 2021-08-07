#ifndef HIR_OPS
#define HIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "circt/Dialect/HIR/IR/HIRInterfaces.td"

// Dialect
def HIR_Dialect : Dialect {
  let name = "hir";
  let summary = "An MLIR dialect for Hardware description";
  let description = [{
    This dialect is designed for hardware description. A circuit described in HIR 
      dialect contains the schedule of execution and functional unit binding 
      information. HIR supports high level constructs such as loops and 
      conditionals.
  }];
  let cppNamespace = "hir";
}

// Types
def TimeType : Type<CPred<"$_self.isa<TimeType>()">, "HIR time type">,
    BuildableType<"TimeType::get($_builder.getContext())">;

def MemrefType: Type<CPred<"$_self.isa<MemrefType>()">, "HIR memref type ">,
    BuildableType<"MemrefType::get($_builder.getContext())">;

def FuncType: Type<CPred<"$_self.isa<FuncType>()">, "HIR interface type ">,
    BuildableType<"FuncType::get($_builder.getContext())">;

def BusType: Type<CPred<"$_self.isa<BusType>()">, "HIR bus type ">,
    BuildableType<"BusType::get($_builder.getContext())">;

// Ops
class HIR_Op<string mnemonic, list<OpTrait> traits = []>
: Op<HIR_Dialect, mnemonic, traits>;


//def TerminatorOp : HIR_Op<"terminator", [ParentOneOf<["BlockOp"]>,Terminator] > {
//  let summary = "Terminator op";
//}

def BlockYieldOp : HIR_Op<"block.yield", [ParentOneOf<["BlockOp"]>,Terminator] > {
  let summary = "Yield from hir.block";
  let arguments = (ins Variadic<AnyType> : $operands);

  let assemblyFormat = [{ attr-dict (`(` $operands^ `)` `:` type($operands))? }];
}

def YieldOp : HIR_Op<"yield", [ParentOneOf<["ForOp","UnrollForOp","IfOp"]>,Terminator,AttrSizedOperandSegments] > {
  let summary = "Yield from loops";
  let description = [{
    This Op yields from a loop.
      Example:

      ```mlir hir.yield (%x, %A) at %t offset %3 : (!hir.val,!hir.val)```
  }];

  let arguments = (ins Variadic<AnyType> : $operands, Optional<TimeType>: $tstart,
      OptionalAttr<I64Attr>: $offset);
  let results = (outs );

  let builders = [OpBuilder<(ins),[{
    build($_builder,$_state,SmallVector<Value>(),Value(),IntegerAttr());
  }]>];
  //TODO: Currently yieldOp prints even if there is no timevar or operands.
  // If the custom printer returns without printing anything, then mlir does not print the op.
  // We can use custom printer/parser to ensure that the yieldOp is not printed.
  let assemblyFormat = [{ attr-dict (`(` $operands^ `)` `:` type($operands))? 
    `at` custom<TimeAndOffset>($tstart, $offset)}];

  //let parser = [{return parse$cppClass(parser,result);}];
  //let printer = [{return print$cppClass(p,*this);}];
  let hasCanonicalizeMethod = true;
}

def ReturnOp : HIR_Op<"return", [HasParent<"FuncOp">, Terminator]> {
  let summary = "Return from function";

  let description = [{
    This instruction must be the last instruction in the function.
      If hir.yield is specified for the function then return must be specified.
      ```mlir hir.return (%x, %y): (i32, !hir.const<i1>)```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins),[{
    build($_builder,$_state,SmallVector<Value>());
  }]>];

  let assemblyFormat = "(`(`$operands^`)` `:` `(`type($operands)`)`)? attr-dict";
}

def FuncOp : HIR_Op<"func", [Symbol, SingleBlockImplicitTerminator<"ReturnOp">, FunctionLike, IsolatedFromAbove, DeclareOpInterfaceMethods<CallableOpInterface>]> {
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
      Region& getFuncBody(){ return body();}
      ///Get the function arguments (excluding tstart).
      SmallVector<Value,4> getOperands();
      Value getTimeVar(){
        return getFuncBody().getArguments().back();
      }
      void updateArguments(ArrayRef<DictionaryAttr>);
      hir::FuncType getFuncType(){return funcTy().dyn_cast<hir::FuncType>();}
  }];
  let verifier =[{return verify$cppClass(*this);}];
}

def BlockOp : HIR_Op<"block", [SingleBlock, RecursiveSideEffects]> {
  let summary = "code block";
  let description = [{
    This op is used for arranging code in a program. It helps the codegen pass to generate human readable Verilog.```
  }];

  let arguments = (ins );
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{(`->` `(` type($results)^`)`)? attr-dict-with-keyword regions  }];
}


def IfOp : HIR_Op<"if", [SingleBlockImplicitTerminator<"YieldOp">, RecursiveSideEffects]> {
  let summary = "if operation";
  let description = [{
    This op implements an if statement.
      Example:
      ```mlir
      hir.if(%b) at %t {...}        ```
  }];

  let arguments = (ins AnyType : $cond, Optional<TimeType> : $tstart, OptionalAttr<I64Attr>: $offset);
  let regions = (region SizedRegion<1>:$if_region);
  let results = (outs );

  let extraClassDeclaration = [{
  }];
  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
}

def ForOp : HIR_Op<"for", [SingleBlockImplicitTerminator<"YieldOp">, RecursiveSideEffects, DeclareOpInterfaceMethods<LoopLikeOpInterface>,AttrSizedOperandSegments]> {
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

  let arguments = (ins AnyType: $lb, AnyType: $ub, AnyType: $step, Variadic<AnyType>: $captures,
      Optional<TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let regions = (region SizedRegion<1>:$region);
  let results = (outs TimeType:$res);

  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  let builders = [OpBuilder<(ins "Value":$lb, "Value":$ub, "Value":$step, "ValueRange":$captures,
      "Value":$tstart, "IntegerAttr":$offset,"Type":$inductionVarTy),[{
    build($_builder,$_state,hir::TimeType::get($_builder.getContext()),lb,ub,step,captures,tstart,offset);

  }]>];
  let extraClassDeclaration = [{
    Value getInductionVar() { return getBody()->getArgument(0); }
    Value setInductionVar(Type ty) { 
      Value newInductionVar = getBody()->insertArgument(
        (unsigned)0, ty);
      getBody()->getArgument(1).replaceAllUsesWith(newInductionVar);
      getBody()->eraseArgument(1);
      return newInductionVar;
    }
    ArrayRef<BlockArgument> getLatchedInputs(){return getBody()->getArguments().slice(1,captures().size());}
    Value setLatchedInput(unsigned i, Type ty) { 
      Value newArg = getBody()->insertArgument(1+i,ty);
      getBody()->getArgument(2+i).replaceAllUsesWith(newArg);
      getBody()->eraseArgument(2+i);
      return newArg;
    }
    Value getIterTimeVar() { return getBody()->getArgument(captures().size()+1); }
    Block* addEntryBlock(MLIRContext *context, Type inductionVarTy);
    void beginRegion(mlir::OpBuilder &builder);        
    void endRegion(mlir::OpBuilder &builder);
  }];
  let hasCanonicalizeMethod = true; 
  let verifier =[{return verify$cppClass(*this);}];
}

def UnrollForOp : HIR_Op<"unroll_for", [SingleBlockImplicitTerminator<"YieldOp">, RecursiveSideEffects, DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
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
      TimeType:$tstart, OptionalAttr<I64Attr>: $offset);
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

def AllocaOp :HIR_Op<"alloca",[NoSideEffect]> {
  let summary = "Instantiate new memory.";

  let description = [{

    ```%w = hir.alloc("bram") : !hir.memref<...>```
  }];

  let arguments = (ins StrAttr:$mem_type, ArrayAttr: $ports);
  let results = (outs MemrefType:$res);

  let assemblyFormat = [{`(`$mem_type`)` attr-dict 
    `:` type($res) `ports` $ports}];
}

def BusInstantiateOp :HIR_Op<"bus.instantiate",[NoSideEffect]> {
  let summary = "Instantiate new bus.";

  let description = [{ Instantiate a new !hir.bus or tensor<...x!hir.bus>
    Tensor of bus can not contain flipped members.
      Example: 
      ```%w = hir.bus.instantiate : !hir.bus<...>```
  }];

  let arguments = (ins);
  let results = (outs AnyType:$res);

  let assemblyFormat = [{attr-dict 
    `:` type($res)
  }];
}

def LoadOp : HIR_Op<"load", [NoSideEffect, AttrSizedOperandSegments]> {
  let summary = "read from a memory.";
  let description = [{
    This Op reads from a memory interface. 
      Indices for banks should be of index type and indices for address should be
      integers of correct width (based on the size of the dimension).

      Example:

      ```mlir hir.load %A[%x,%y] at %t : !hir.memref<...>```
  }];

  let arguments = (ins MemrefType:$mem, 
      Variadic<AnyType>:$indices, OptionalAttr<I64Attr>:$port, OptionalAttr<I64Attr>: $delay, Optional<TimeType> : $tstart,
      OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType: $res);
  let assemblyFormat = [{
    $mem (`[` `port` $port^ `]`)? `[`  $indices `]` `at` custom<TimeAndOffset>($tstart ,$offset) 
      attr-dict
      `:`custom<MemrefAndElementType>
      (type($mem),type($indices),type($res)) (`delay` $delay^)?
  }];

  let hasCanonicalizeMethod = true;
  let verifier =[{return verify$cppClass(*this);}];
  let extraClassDeclaration = [{
    public:
      SmallVector<Value> filterIndices(hir::DimKind);
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
      Variadic<AnyType>:$indices, OptionalAttr<I64Attr>:$port, 
      OptionalAttr<I64Attr>:$delay, Optional<TimeType> :$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs );
  let assemblyFormat = [{
    $value `to` $mem  (`[` `port` $port^ `]`)? `[` $indices `]` `at` custom<TimeAndOffset>($tstart, $offset) attr-dict `:` 
      custom<MemrefAndElementType>
      (type($mem),type($indices), type($value)) (`delay` $delay^)?
  }];

  let hasCanonicalizeMethod = true;
  let verifier =[{return verify$cppClass(*this);}];
  let extraClassDeclaration = [{
    public:
      SmallVector<Value> filterIndices(hir::DimKind);
  }];
}

def SendOp : HIR_Op<"send", []> {
  let summary = "write to an interface.";
  let description = [{
    This Op writes to a bus. 
      Example:
      ```mlir hir.send %1 to %v[%0] : !hir.bus<...>```
  }];

  let arguments = (ins AnyType:$value, BusType:$bus, I64Attr:$index,
      Optional<TimeType> :$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs );
  let assemblyFormat = [{
    $value `to` $bus `[` $index`]`
      `at` custom<TimeAndOffset> ($tstart, $offset) attr-dict  `:` type($value) `to` type($bus)  
  }];
  let hasCanonicalizeMethod = true;
}

def RecvOp : HIR_Op<"recv", [NoSideEffect]> {
  let summary = "read from a group or array.";
  let description = [{
    This Op Reads from a bus. 

      ```mlir %v = hir.recv %x[%0] : !hir.bus<...> -> i32```
  }];

  let arguments = (ins BusType:$bus,I64Attr:$index, 
      Optional<TimeType> :$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType:$res);

  let assemblyFormat = [{
    $bus `[` $index `]`  
      `at` custom<TimeAndOffset>($tstart, $offset) attr-dict  `:` type($bus) `->` type($res)  
  }];
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

  let arguments = (ins FlatSymbolRefAttr:$callee, 
      TypeAttr : $funcTy,
      Variadic<AnyType>:$operands, 
      Optional<TimeType>:$tstart,
      OptionalAttr<I64Attr>: $offset
      );
  let results = (outs Variadic<AnyType> : $res);

  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  let hasCanonicalizeMethod = true;
  let verifier =[{return verify$cppClass(*this);}];
  let extraClassDeclaration=[{
    public:
      ///Get the function arguments (excluding tstart).
      SmallVector<Value,4> getOperands();
      hir::FuncType getFuncType(){return funcTy().dyn_cast<hir::FuncType>();}
  }];
}

def CallInstanceOp : HIR_Op<"call.instance",[AttrSizedOperandSegments] > {
  let summary = "Call a previously instantiated function.";
  let description = [{
    This Op calls a previously instantiated function. It is used to fuse multiple instances of same function. 
      Example:

      ```mlir 
      %a,%b = hir.call.instance %foo (%x,%A) at %ti : hir.func<(!hir.val,!hir.memref) -> (!hir.val,!hir.val)>
      ```
  }];

  let arguments = (ins FuncType:$callee, 
      Variadic<AnyType>:$operands, 
      Optional<TimeType>:$tstart, 
      OptionalAttr<I64Attr>: $offset
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
def DelayOp: HIR_Op<"delay",[NoSideEffect]>{
  let summary = "Delay operation";
  let description = [{
    This op delays the input value by a fixed amount of time.
      Example:
      ```mlir %yy = hir.delay %xx by %1 at %tj```
  }];

  let arguments = (ins AnyType : $input, I64Attr:$delay, 
      Optional<TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{$input `by` $delay `at` custom<TimeAndOffset> 
    ($tstart, $offset) attr-dict `:` type($res) custom<CopyType>(type($input),ref(type($res)))}];
  let verifier =[{return verify$cppClass(*this);}];
}

def TimeOp: HIR_Op<"time",[NoSideEffect]>{
  let summary = "Define a new time var";
  let description = [{
    This op defines a new time var that represents a fixed delay from another time var.
      Example:
      ```mlir %t = hir.time %t+3 :!hir.time``
  }];

  let arguments = (ins TimeType : $timevar, I64Attr:$delay);
  let results = (outs TimeType : $res);
  let assemblyFormat = [{custom<TimeAndOffset>($timevar, $delay) attr-dict `:` type($res)}];
  let hasFolder = true;
}

def LatchOp: HIR_Op<"latch",[NoSideEffect]>{
  let summary = "Latch a value for certain time.";
  let description = [{
      This op latches the input value until the next input value is latched at a later instant. 
      It is undefined behaviour if the latched value is accessed after it is overwritten by a new value.
      This instruction can be used to move values between time domains.

      Example:
      ```mlir %yy = hir.latch %xx at %tx+%1 : i32```
  }];

  let arguments = (ins AnyType : $input,
      Optional<TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{$input `at` custom<TimeAndOffset> ($tstart, $offset)  
      attr-dict `:` type($res) custom<CopyType>(type($input),ref(type($res)))
  }];
  let hasFolder = true;
}

def AssignOp: HIR_Op<"assign",[]>{
  let summary = "Assign a constant value to a bus.";
  let description = [{
    This op delays the input value by a fixed amount of time.
      Example:
      ```mlir %yy = hir.delay %xx by %1 at %tj```
  }];

  let arguments = (ins AnyAttr:$value, BusType : $bus);
  let results = (outs);
  let assemblyFormat = [{$value `to` $bus attr-dict `:` type($bus)}];
}

def AddIOp: HIR_Op<"addi",[NoSideEffect]>{
  let summary = "Integer add operation";
  let description = [{
    This op adds two inputs of same type (integer or index).
      Example:
      ```mlir %yy = hir.addi (%a, %b) at %t: i32 delay 1```
  }];

  let arguments = (ins AnyType : $op1, AnyType: $op2,  OptionalAttr<I64Attr>:$delay, 
      Optional<TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{`(`$op1 `,` $op2 `)` `at` custom<TimeAndOffset> 
    ($tstart, $offset) attr-dict `:` custom<BinOpOperandsAndResultType>(type($res), type($op1),type($op2)) (`delay` $delay^)?
  }];
  let hasCanonicalizeMethod = true;
}

def SubIOp: HIR_Op<"subi",[NoSideEffect]>{
  let summary = "Integer subtract operation";
  let description = [{
    This op subtracts second operand from first operand. Operands must be of same type (integer or index).
      Example:
      ```mlir %yy = hir.subi (%a, %b) at %t: i32 delay 1```
  }];

  let arguments = (ins AnyType : $op1, AnyType: $op2,  OptionalAttr<I64Attr>:$delay, 
      Optional<TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{`(`$op1 `,` $op2 `)` `at` custom<TimeAndOffset> 
    ($tstart, $offset) attr-dict `:` custom<BinOpOperandsAndResultType>(type($res), type($op1),type($op2)) (`delay` $delay^)?
  }];
  let hasCanonicalizeMethod = true;
}

def MulIOp: HIR_Op<"muli",[NoSideEffect]>{
  let summary = "Integer Multiply operation";
  let description = [{
    This op multipliers to integer operands. Operands must be of same type (integer or index).
      Example:
      ```mlir %yy = hir.muli (%a, %b ) at %t: i32 delay 1```
  }];

  let arguments = (ins AnyType : $op1, AnyType: $op2,  OptionalAttr<I64Attr>:$delay, 
      Optional<TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{`(`$op1 `,` $op2`)` `at` custom<TimeAndOffset> 
    ($tstart, $offset) attr-dict `:` custom<BinOpOperandsAndResultType>(type($res), type($op1),type($op2)) (`delay` $delay^)?
  }];
  let hasCanonicalizeMethod = true;
}

def AddFOp: HIR_Op<"addf",[NoSideEffect]>{
  let summary = "floating point add operation";
  let description = [{
    This op adds two floatin point inputs of same type.
      Example:
      ```mlir %yy = hir.addf (%a, %b) at %t: f32 delay 4```
  }];

  let arguments = (ins AnyType : $op1, AnyType: $op2,  OptionalAttr<I64Attr>:$delay, 
      Optional<TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{`(`$op1 `,` $op2`)` `at` custom<TimeAndOffset> 
    ($tstart, $offset) attr-dict `:` custom<BinOpOperandsAndResultType>(type($res), type($op1),type($op2)) (`delay` $delay^)?
  }];
  let hasCanonicalizeMethod = true;
}

def SubFOp: HIR_Op<"subf",[NoSideEffect]>{
  let summary = "Floating point subtract operation";
  let description = [{
    This op subtracts second floating point operand from first. Operands must be of same type.
      Example:
      ```mlir %yy = hir.subf (%a, %b) at %t: f32 delay 4```
  }];

  let arguments = (ins AnyType : $op1, AnyType: $op2,  OptionalAttr<I64Attr>:$delay, 
      Optional<TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{`(`$op1 `,` $op2 `)` `at` custom<TimeAndOffset> 
    ($tstart, $offset) attr-dict `:` custom<BinOpOperandsAndResultType>(type($res), type($op1),type($op2)) (`delay` $delay^)?
  }];
  let hasCanonicalizeMethod = true;
}

def MulFOp: HIR_Op<"mulf",[NoSideEffect]>{
  let summary = "Floating point Multiply operation.";
  let description = [{
    This op multipliers two floating point operands. Operands must be of same type.
      Example:
      ```mlir %yy = hir.mulf (%a, %b) at %t: f32 delay 1```
  }];

  let arguments = (ins AnyType : $op1, AnyType: $op2,  OptionalAttr<I64Attr>:$delay, 
      Optional<TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{`(`$op1 `,` $op2`)` `at` custom<TimeAndOffset> 
    ($tstart, $offset) attr-dict `:` custom<BinOpOperandsAndResultType>(type($res), type($op1),type($op2)) (`delay` $delay^)?
  }];
  let hasCanonicalizeMethod = true;
}

def TensorExtract :HIR_Op<"tensor.extract",[NoSideEffect]>{
  let summary = "Extract a bus from a tensor";
  let description = [{
    This op extracts a single bus from a tensor of busses.
      Examples:
      ```mlir %1 = hir.bus.extract %x[%0] : tensor<4x!hir.bus<...>>```
  }];

  let arguments = (ins AnyType: $tensor, Variadic<Index>: $indices, ArrayAttr: $ports);
  let results = (outs BusType: $res);

  let assemblyFormat = [{ $tensor `[`$indices`]` attr-dict `:`
    type($tensor) `->` type($res) `ports` $ports}];
}

def CreateTupleOp:HIR_Op<"create_tuple",[NoSideEffect]>{
  let summary = "creates a tuple from inputs";
  let description = [{
    This op creates a tuple of mlir builting types.
      Examples:
      ```mlir %1 = hir.create_tuple (%x, %y) : (i1, i32) -> Tuple<i1,i32>```
  }];

  let arguments = (ins Variadic<AnyType>: $args);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ `(` $args`)` attr-dict `:` 
    `(` type($args) `)` `->` type($res) }];
}

def UnpackTupleOp:HIR_Op<"unpack_tuple",[NoSideEffect]>{
  let summary = "creates a tuple from inputs";
  let description = [{
    This op unpacks a tuple of values of mlir builtin types.
      Examples:
      ```mlir %1, %2 = hir.unpack %x : Tuple<i1,i32>```
  }];

  let arguments = (ins AnyType: $tuple);
  let results = (outs Variadic<AnyType>: $res);

  let assemblyFormat = [{ `(` $tuple`)` attr-dict `:` 
    type($tuple) `->` `(` type($res) `)`}];
}

#endif // HIR_OPS
