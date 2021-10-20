#ifndef HIR_OPS
#define HIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

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
  let cppNamespace = "::circt::hir";
  let hasConstantMaterializer = 1;
}
//Op Interfaces
include "HIROpInterfaces.td"

// Types
def HIR_TimeType : DialectType<HIR_Dialect,CPred<"$_self.isa<TimeType>()">, "HIR time type">,
    BuildableType<"TimeType::get($_builder.getContext())">;

def HIR_MemrefType: DialectType<HIR_Dialect,CPred<"$_self.isa<MemrefType>()">, "HIR memref type ">,
    BuildableType<"MemrefType::get($_builder.getContext())">;

def HIR_FuncType: DialectType<HIR_Dialect,CPred<"$_self.isa<FuncType>()">, "HIR interface type ">,
    BuildableType<"FuncType::get($_builder.getContext())">;

def HIR_BusType: DialectType<HIR_Dialect, CPred<"$_self.isa<BusType>()">, "HIR bus type ">,
    BuildableType<"BusType::get($_builder.getContext())">;

// Ops
class HIR_Op<string mnemonic, list<OpTrait> traits = []>
: Op<HIR_Dialect, mnemonic, traits>;


def NextIterOp : HIR_Op<"next_iter", [AttrSizedOperandSegments, ParentOneOf<["ForOp","WhileOp"]>,Terminator] > {
  let summary = "specifies the time at which the next iteration of the loop starts.";
  let description = [{
    This op specifies when the next iteration of the loop starts.
    If the current iteration is the last iteration then it specifies the time 
    returned by the op.
    If a condition is provided then the loop breaks when the condition is false;

    This instruction must be the last instruction in for loop body.

      Example:

      ```mlir hir.next_iter condition %continue at %t```
      ```mlir hir.next_iter at %t```
  }];

  let arguments = (ins Optional<I1> :$condition, Optional<HIR_TimeType> : $tstart, 
    OptionalAttr<I64Attr>: $offset);

  let results = (outs );
  let hasCanonicalizeMethod = true;
  let assemblyFormat = [{ (`condition` $condition^)? `at` custom<TimeAndOffset>($tstart ,$offset) attr-dict }];
  let verifier =[{return verify$cppClass(*this);}];
}

def IsFirstIterOp :HIR_Op<"is_first_iter",[ParentOneOf<["WhileOp","ForOp"]>,NoSideEffect]>{
  let summary = "Returns true if this is the first iteration.";
  let results = (outs I1:$res);

  let assemblyFormat = [{ custom<WithSSANames>(attr-dict) `:` type($res)}];
}

def YieldOp : HIR_Op<"yield", [ParentOneOf<["IfOp"]>,Terminator] > {
  let summary = "Yield from loops";
  let description = [{
    This Op yields values out of an hir.if.
    This instruction must be the last instruction in if/else bodies.
      Example:

      ```mlir hir.yield (%x, %A) : (f32, !hir.time)```
  }];

  let arguments = (ins Variadic<AnyType> : $operands);
  let results = (outs );
  
  let builders = [OpBuilder<(ins),[{
    build($_builder,$_state,SmallVector<Value>());
  }]>];
  let assemblyFormat = [{ attr-dict (`(` $operands^ `)` `:` `(` type($operands) `)`)? }];
  let verifier =[{return verify$cppClass(*this);}];
}

def ReturnOp : HIR_Op<"return", [ParentOneOf<["FuncOp","FuncExternOp"]>, Terminator]> {
  let summary = "Return from function";

  let description = [{
    This op returns the output of the function.
    This instruction must be the last instruction in the function.
      ```mlir hir.return (%x, %y): (i32, !hir.const<i1>)```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins),[{
    build($_builder,$_state,SmallVector<Value>());
  }]>];
  let assemblyFormat = [{ attr-dict (`(` $operands^ `)` `:` `(` type($operands) `)`)? }];
}

def FuncOp : HIR_Op<"func", 
  [Symbol, SingleBlockImplicitTerminator<"ReturnOp">, FunctionLike, FuncLike,
  IsolatedFromAbove, CallableOpInterface, HasParent<"mlir::ModuleOp">]> {

  let summary = "Function definition";
  let description = [{
    This Op creates a new function 
      Example:

      ```mlir 
      hir.func @foo (%x, %A) at %t : (i4,!hir.memref<...>) -> (f32,!hir.time)
      ```
  }];

  let arguments = (ins
      TypeAttr:$type,
      StrAttr:$sym_name,
      TypeAttr:$funcTy 
      );

  let regions = (region SizedRegion<1>:$body);
  let results = (outs );
  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  let extraClassDeclaration = [{
    private:
      // FunctionLike interface.
      friend class OpTrait::FunctionLike<FuncOp>;

      unsigned getNumFuncArguments() { return getType().getNumInputs(); }
      unsigned getNumFuncResults() { return getType().getNumResults(); }
      LogicalResult verifyType();
      LogicalResult verifyBody();
      Block *getBodyBlock() { return &body().front(); }
      public:
      //CallableOpInterface
      Region* getCallableRegion(){return &getOperation()->getRegion(0);}
      ArrayRef<Type> getCallableResults(){return getFuncType().getResultTypes();}

    public:
      Region& getFuncBody(){ return body();}
      ///Get the function arguments (excluding tstart).
      SmallVector<Value,4> getOperands();
      Value getRegionTimeVar(){
        return getFuncBody().getArguments().back();
      }
      void updateArguments(ArrayRef<DictionaryAttr>);
      hir::FuncType getFuncType(){return funcTy().dyn_cast<hir::FuncType>();}
      llvm::Optional<Block*> getOptionalFuncBodyBlock(){ return &getFuncBody().front();}
      ArrayAttr getInputNames(){return
      (*this)->getAttr("argNames").dyn_cast_or_null<ArrayAttr>();}
      ArrayAttr getResultNames(){return
      (*this)->getAttr("resultNames").dyn_cast_or_null<ArrayAttr>();}
  }];
  let verifier =[{return verify$cppClass(*this);}];
}

def FuncExternOp : HIR_Op<"func.extern", 
[Symbol, SingleBlockImplicitTerminator<"ReturnOp">, FuncLike, HasParent<"mlir::ModuleOp">]> {
  let summary = "Function declaration";

  let description = [{
    This Op declares a new external function.
      Example:
      ```mlir 
      hir.func.extern @foo (%x, %A) at %t : (i4,!hir.memref<...>) -> (f32,!hir.time)
      ```
  }];

  let arguments = (ins
      StrAttr:$sym_name,
      TypeAttr:$funcTy 
      );

  let regions = (region SizedRegion<1>:$body);
  let results = (outs );
  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  let extraClassDeclaration = [{
    public:
      // FunctionLike interface.
      friend class OpTrait::FunctionLike<FuncOp>;

      unsigned getNumFuncArguments() { return getFuncType().getInputTypes().size(); }
      unsigned getNumFuncResults() { return getFuncType().getResultTypes().size(); }
      public:
      LogicalResult verifyType();

    public:
    //FuncLike op interface.
      hir::FuncType getFuncType(){return funcTy().dyn_cast<hir::FuncType>();}
      Region& getFuncBody(){ return body();}
      void updateArguments(ArrayRef<DictionaryAttr>);
      ArrayAttr getInputNames(){return
      (*this)->getAttr("argNames").dyn_cast_or_null<ArrayAttr>();}
      ArrayAttr getResultNames(){return
      (*this)->getAttr("resultNames").dyn_cast_or_null<ArrayAttr>();}
  }];
  //let verifier =[{return verify$cppClass(*this);}];
}

def CommentOp : HIR_Op<"comment", []> {
  let summary = "comment";
  let description = [{
    This op is used to add comments. This comments are output by the verilog-generator.```
  }];

  let arguments = (ins StrAttr:$comment );
  let results = (outs );

  let assemblyFormat = [{$comment attr-dict}];
}

def IfOp : HIR_Op<"if", [SingleBlock, RecursiveSideEffects]> {
  let summary = "if operation";
  let description = [{
    This op implements an if statement.
      Example:
      ```mlir
      hir.if(%b) at %t {...}        ```
  }];

  let arguments = (ins AnyType : $condition, Optional<HIR_TimeType> : $tstart, OptionalAttr<I64Attr>: $offset, OptionalAttr<ArrayAttr>:$result_attrs);
  let regions = (region SizedRegion<1>:$if_region, SizedRegion<1>:$else_region);
  let results = (outs Variadic<AnyType>:$results);

  let extraClassDeclaration = [{
      Value getRegionTimeVar(){
        return if_region().getArguments().back();
      }
  }];
  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  let verifier =[{return verify$cppClass(*this);}];
  let hasCanonicalizeMethod =true;
}

def ForOp : HIR_Op<"for", [SingleBlock, RecursiveSideEffects, DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
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
      Optional<HIR_TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let regions = (region SizedRegion<1>:$body);
  let results = (outs HIR_TimeType:$res);

  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  
  let extraClassDeclaration = [{
    Value getInductionVar() { return getBody()->getArgument(0); }
    Value setInductionVar(Type ty) { 
      Value newInductionVar = getBody()->insertArgument(
        (unsigned)0, ty);
      getBody()->getArgument(1).replaceAllUsesWith(newInductionVar);
      getBody()->eraseArgument(1);
      return newInductionVar;
    }
    
    Value getIterTimeVar() { return getBody()->getArguments().back(); }
    Block* addEntryBlock(MLIRContext *context, Type inductionVarTy);
    void beginRegion(mlir::OpBuilder &builder);        
    void endRegion(mlir::OpBuilder &builder);
    SmallVector<Value> getCapturedValues();
    StringRef getInductionVarName(){return (*this)->getAttr("argNames").dyn_cast<ArrayAttr>()[0].dyn_cast<StringAttr>().getValue();}
    StringRef getIterTimeVarName(){return (*this)->getAttr("argNames").dyn_cast<ArrayAttr>()[1].dyn_cast<StringAttr>().getValue();}
  }];
  let hasCanonicalizeMethod = true; 
  let verifier =[{return verify$cppClass(*this);}];
}

def WhileOp : HIR_Op<"while", [SingleBlock, RecursiveSideEffects]> {
  let summary = "for operation";
  let arguments = (ins I1:$condition, Optional<HIR_TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs HIR_TimeType:$res);
  let regions = (region SizedRegion<1>:$body);
  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];

  let extraClassDeclaration = [{
    Value getIterTimeVar() { return body().getArguments().back(); }
    SmallVector<Value> getCapturedValues();
    Block* addEntryBlock();
  }];

  let builders = [OpBuilder<(ins "Value":$condition, "Value":$tstart, "IntegerAttr":$offset),[{
    build($_builder,$_state,hir::TimeType::get($_builder.getContext()), condition, tstart, offset);
  }]>];
}

def AllocaOp :HIR_Op<"alloca",[]> {
  let summary = "Instantiate new memory.";

  let description = [{

    ```%w = hir.alloc("bram") : !hir.memref<...>```
  }];

  let arguments = (ins StrAttr:$mem_type, ArrayAttr: $ports);
  let results = (outs HIR_MemrefType:$res);

  let assemblyFormat = [{`(`$mem_type`)` custom<WithSSANames>(attr-dict) 
    `:` type($res) `ports` $ports}];
  let verifier =[{return verify$cppClass(*this);}];
}

def BusOp :HIR_Op<"bus",[]> {
  let summary = "Instantiate new bus.";

  let description = [{ Instantiate a new !hir.bus or tensor<...x!hir.bus>
    Tensor of bus can not contain flipped members.
      Example: 
      ```%w = hir.bus : !hir.bus<...>```
  }];

  let arguments = (ins);
  let results = (outs AnyType:$res);

  let assemblyFormat = [{custom<WithSSANames>(attr-dict) `:` type($res) }];
}

def LoadOp : HIR_Op<"load", [AttrSizedOperandSegments]> {
  let summary = "read from a memory.";
  let description = [{
    This Op reads from a memory interface. 
      Indices for banks should be of index type and indices for address should be
      integers of correct width (based on the size of the dimension).

      Example:

      ```mlir hir.load %A[%x,%y] at %t : !hir.memref<...>```
  }];

  let arguments = (ins HIR_MemrefType:$mem, 
      Variadic<AnyType>:$indices, OptionalAttr<I64Attr>:$port, OptionalAttr<I64Attr>: $delay, Optional<HIR_TimeType> : $tstart,
      OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType: $res);
  let assemblyFormat = [{
    $mem (`[` `port` $port^ `]`)? `[`  $indices `]` `at` custom<TimeAndOffset>($tstart ,$offset) 
      custom<WithSSANames>(attr-dict)
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

  let arguments = (ins AnyType:$value, HIR_MemrefType:$mem,
      Variadic<AnyType>:$indices, OptionalAttr<I64Attr>:$port, 
      OptionalAttr<I64Attr>:$delay, Optional<HIR_TimeType> :$tstart, OptionalAttr<I64Attr>: $offset);
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

  let arguments = (ins AnyType:$value, HIR_BusType:$bus, I64Attr:$index,
      Optional<HIR_TimeType> :$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs );
  let assemblyFormat = [{
    $value `to` $bus `[` $index`]`
      `at` custom<TimeAndOffset> ($tstart, $offset) attr-dict  `:` type($value) `to` type($bus)  
  }];
  let hasCanonicalizeMethod = true;
}

def RecvOp : HIR_Op<"recv", []> {
  let summary = "read from a group or array.";
  let description = [{
    This Op Reads from a bus. 

      ```mlir %v = hir.recv %x[%0] : !hir.bus<...> -> i32```
  }];

  let arguments = (ins HIR_BusType:$bus,I64Attr:$index, 
      Optional<HIR_TimeType> :$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType:$res);

  let assemblyFormat = [{
    $bus `[` $index `]`  
      `at` custom<TimeAndOffset>($tstart, $offset) custom<WithSSANames>(attr-dict)  `:` type($bus) `->` type($res)  
  }];
  let hasCanonicalizeMethod = true;
}

def CallOp : HIR_Op<"call", [CallOpInterface,AttrSizedOperandSegments] > {
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
      Optional<HIR_TimeType>:$tstart,
      OptionalAttr<I64Attr>: $offset
      );
  let results = (outs Variadic<AnyType> : $results);

  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
  let hasCanonicalizeMethod = true;
  let verifier =[{return verify$cppClass(*this);}];
  let extraClassDeclaration=[{
    public:
      ///Get the function arguments (excluding tstart).
      SmallVector<Value,4> getOperands();
      hir::FuncType getFuncType(){return funcTy().dyn_cast<hir::FuncType>();}

    //CallOpInterface

    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType() {return getFuncType().getFunctionType();}

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Lookup the FuncOp/ExternalFuncOp for the callee. This returns null on invalid IR.
    Operation* getCalleeDecl() ;
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

  let arguments = (ins HIR_FuncType:$callee, 
      Variadic<AnyType>:$operands, 
      Optional<HIR_TimeType>:$tstart, 
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
      Optional<HIR_TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{$input `by` $delay `at` custom<TimeAndOffset> 
    ($tstart, $offset) custom<WithSSANames>(attr-dict) `:` type($res) custom<CopyType>(type($input),ref(type($res)))}];
  let verifier =[{return verify$cppClass(*this);}];
}

def TimeOp: HIR_Op<"time",[NoSideEffect]>{
  let summary = "Define a new time var";
  let description = [{
    This op defines a new time var that represents a fixed delay from another time var.
      Example:
      ```mlir %t = hir.time %t+3 :!hir.time``
  }];

  let arguments = (ins HIR_TimeType : $timevar, I64Attr:$delay);
  let results = (outs HIR_TimeType : $res);
  let assemblyFormat = [{custom<TimeAndOffset>($timevar, $delay) attr-dict `:` type($res)}];
  let hasFolder = true;
}

def TimeMaxOp: HIR_Op<"time.max",[NoSideEffect]>{
  let summary = "Define a new time var that is the largest of the input time vars.";
  let description = [{
    This op defines a new time var that is the larger of the input time vars.
      Example:
      ```mlir %t = hir.time (%t1, %t2, %t3) :!hir.time``
  }];

  let arguments = (ins Variadic<HIR_TimeType> : $input_timevars);
  let results = (outs HIR_TimeType : $res);
  let assemblyFormat = [{`(` $input_timevars `)` custom<WithSSANames>(attr-dict) `:` type($res)}];
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
      Optional<HIR_TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
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

  let arguments = (ins AnyAttr:$value, HIR_BusType : $bus);
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
      Optional<HIR_TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{`(`$op1 `,` $op2 `)` `at` custom<TimeAndOffset> 
    ($tstart, $offset) custom<WithSSANames>(attr-dict) `:` custom<BinOpOperandsAndResultType>(type($res), type($op1),type($op2)) (`delay` $delay^)?
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
      Optional<HIR_TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{`(`$op1 `,` $op2 `)` `at` custom<TimeAndOffset> 
    ($tstart, $offset) custom<WithSSANames>(attr-dict) `:` custom<BinOpOperandsAndResultType>(type($res), type($op1),type($op2)) (`delay` $delay^)?
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
      Optional<HIR_TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{`(`$op1 `,` $op2`)` `at` custom<TimeAndOffset> 
    ($tstart, $offset) custom<WithSSANames>(attr-dict) `:` custom<BinOpOperandsAndResultType>(type($res), type($op1),type($op2)) (`delay` $delay^)?
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
      Optional<HIR_TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
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
      Optional<HIR_TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
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
      Optional<HIR_TimeType>:$tstart, OptionalAttr<I64Attr>: $offset);
  let results = (outs AnyType : $res);
  let assemblyFormat = [{`(`$op1 `,` $op2`)` `at` custom<TimeAndOffset> 
    ($tstart, $offset) attr-dict `:` custom<BinOpOperandsAndResultType>(type($res), type($op1),type($op2)) (`delay` $delay^)?
  }];
  let hasCanonicalizeMethod = true;
}

def TensorExtractOp :HIR_Op<"tensor.extract",[NoSideEffect]>{
  let summary = "Extract a bus from a tensor";
  let description = [{
    This op extracts a single bus from a tensor of busses.
      Examples:
      ```mlir %1 = hir.tensor.extract %x[%0] : tensor<4x!hir.bus<...>>```
  }];

  let arguments = (ins AnyType: $tensor, Variadic<Index>: $indices, ArrayAttr: $ports);
  let results = (outs HIR_BusType: $res);

  let assemblyFormat = [{ $tensor `[`$indices`]` custom<WithSSANames>(attr-dict) `:`
    type($tensor) `->` type($res) `ports` $ports}];
}

def MemrefExtractOp :HIR_Op<"memref.extract",[NoSideEffect]>{
  let summary = "Extract a port from a memref";
  let description = [{
    This op extracts a port from a memref.
      Examples:
      ```mlir %1 = hir.memref.extract %x[port 1] : hir.memref<1xf32> -> hir.memref<1xf32> ports [#wr]```
  }];

  let arguments = (ins HIR_MemrefType: $mem, OptionalAttr<I64Attr>: $port, ArrayAttr: $ports);
  let results = (outs HIR_MemrefType: $res);

  let assemblyFormat = [{$mem `[` `port` $port `]`  custom<WithSSANames>(attr-dict) `:`
    type($res) `ports` $ports  custom<CopyType>(type($mem),ref(type($res)))}];
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

  let assemblyFormat = [{ `(` $args`)` custom<WithSSANames>(attr-dict) `:` 
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

  let assemblyFormat = [{ `(` $tuple`)` custom<WithSSANames>(attr-dict) `:` 
    type($tuple) `->` `(` type($res) `)`}];
}

def BusSelectOp: HIR_Op<"bus.select",[NoSideEffect]>{
  let summary = "Create a new bus from two buses.";
  let description = [{
    This op creates a new bus from two buses. At any instant, the value is
    chosen from the input buses based on a third select bus.
      Examples:
      ```mlir %merged_bus 
        = hir.bus.select %select_bus, %input1, %input2 : !hir.bus<i32>```
  }];

  let arguments = (ins AnyType: $select_bus, AnyType:$true_bus,
  AnyType:$false_bus);
  let results = (outs AnyType: $res);

  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
}

def TensorInsertOp: HIR_Op<"tensor.insert",[NoSideEffect]>{
  let summary = "Create a new tensor by replacing an element from the input tensor.";
  let description = [{
    This op creates a new tensor from an old tensor by replacing one element of
    the old tensor with a new element.
      Examples:
      ```mlir %new_tensor 
        = hir.tensor.insert %element into %tensor[%c0,%c1] : type(%res)```
  }];

  let arguments = (ins AnyType: $element, AnyType:$input_tensor,
  Variadic<Index>: $indices);
  let results = (outs AnyType: $res);

  let parser = [{return parse$cppClass(parser,result);}];
  let printer = [{return print$cppClass(p,*this);}];
}

def BusAssignOp: HIR_Op<"bus.assign",[NoSideEffect,SameOperandsAndResultType]>{
  let summary = "Assign one bus/tensor-of-buses to another.";
  let description = [{
    This op assigns one bus to another.
      Examples:
      ```mlir %merged_bus 
        = hir.bus.assign %dest, %src: !hir.bus<i32>```
  }];

  let arguments = (ins AnyType: $dest, AnyType:$src);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ $dest `,` $src custom<WithSSANames>(attr-dict) `:` type($res) }];
}

def BusOrOp: HIR_Op<"bus.or",[NoSideEffect,SameOperandsAndResultType]>{
  let summary = "or of two buses.";
  let description = [{
    This op creates a new bus which is the or of the two input buses.
      Examples:
      ```mlir %merged_bus 
        = hir.bus.or %left, %right: !hir.bus<i1>```
  }];

  let arguments = (ins AnyType: $left, AnyType:$right);
  let results = (outs AnyType: $res);

  let assemblyFormat = [{ $left `,` $right custom<WithSSANames>(attr-dict) `:` type($res) }];
}

#endif // HIR_OPS
