#HIR
HIR is an mlir dialect for hardware description.

#Enhancement proposals 

##instantiateOp
* Syntax: `hir.instantiate hardware-block : interface-type`
* `hardware-block` can be one of the following - Function symbol,
    uram/bram/lutram/reg/latch or nothing.
* `hir.instantiate nothing:...` only creates the interface but does not connect 
    it to anything.

## TimeType
* Allow casting between vref<i1> and TimeType.

## unroll_for loop
* Allow multiple induction variables. 
* Sytax: `hir.unroll_for (i,j) = (0,0) to (10,100) step (1,10) ...`
* The type of the induction var is always mlir index type.
* Add support for iter_args.

## commentOp.
* The comment op is a nop and is used to optionally inserts comments in the 
    generated verilog.
* Syntax: `hir.comment string-attr` 

## Interface types
The container types provide uniform interfaces to underlying hardware elements.
The vref and memref container types can be cast to each other.

### vref type
* vref types lower to a set of wires.
* The writes to vrefs are volatile, i.e. reads must occur in the same cycle.
* Operations: instantiate, cast, zip, slice, assign, read, write, call
* Attributes: read, write, call
* Syntax: `hir.vref< element-type optional-attr-dict>`
* element-type - IntType, FloatType, ComplexType, Tuple, Tensor
* Tuple is enhanced to support optional attr-dict
  - Syntax : hir.vref<tuple<i1 #rd, i1 #wr, f32 #wr>>
  - Syntax : hir.vref<tuple<i8 , i8> #rw>

* Usecases:
  - Multi-dimensional array of wires: `hir.vref<tensor<3x4xi32> attr-dict>`
  - Array of interfaces:`hir.vref<tensor<4xtuple<...>>>`
* slice op indices must be either constant or mlir index type.
  - Syntax: `hir.slice %a ([1,i,j], [0] attr-dict)`, i and j are of index type.

### memref type
* memref is an interface to memory
* The underlying memory element can be a dram, uram, bram, lutram, reg or latch.
* Operations: instantiate, cast, load, store
* Attributes: load, store
* Syntax: `hir.memref<MxNxelement-type, attr-dict>`
* The `attr-dict` specifies the banked dims and load/store delays.
* `element-type` can be IntType,FloatType,ComplexType and TupleType.
* Indices for banked dims in load/store op must be of type index or constants.

## Lowering
* Convert a fully packed memref to a `vref<tuple<...>>`
* For banked memrefs use `vref<tensor<tuple<...>>>`
* remove the casts between memrefs and interfaces.
* Lower load/store to read and write ops.
* Convert read/write/call/assign op of more complex interfaces to a slice op 
    followed by read/write/call/assign of a simple vref (i.e. element type is 
    either IntType or FloatType).
* Simplify slice of multiple elements to slice + zip.
* If the slice is nested inside unroll_for loop then
  - Check if the slice op accesses all elements.
  - Create a new slice op at the outermost level where the interface is defined.
  - Use the output interface of the slice instead of the original interface
      inside unroll loop nest.
  - If the step size in any of the unroll_for loops is greater than 1 then add a
      new induction var with step size 1 and use it to index the new interface 
      in the slice op inside the unroll loop.
* Lower iter_time in unroll_for op into iter_args by casting between time var 
    and vref<i1>.
* Lower iter_args of unroll_for loop to vref<tensor<...>> + read/write.
* Lower hir.instantiate to hir.call.

