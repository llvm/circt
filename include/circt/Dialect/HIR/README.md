#HIR
HIR is an mlir dialect for hardware description.

#Enhancement proposals 

## Unroll for loop
* Allow multiple induction variables. 
* Sytax: `hir.unroll_for (i,j) = (0,0) to (10,100) step (1,10)`
* The type of the induction var is always mlir index type.

## Add a comment op.
* The comment op is a nop and is used to optionally inserts comments in the 
    generated verilog.
* Syntax: `hir.comment string-attr` 

## Container types
The container types provide uniform interfaces to underlying hardware elements.
The interface and memref container types can be cast to each other.

### interface 
* interface types lower to a set of wires.
* Operations: alloc, cast, zip, slice, assign, read, write, call
* Attributes: read, write, call
* Syntax: `hir.interface< element-type optional-attr-dict, ... >`
* element-type : Following mlir primitive types are supported - IntType,
    FloatType, ComplexType, TupleType, TensorType
* Usecases:
  - Multi-dimensional array of wires: `hir.interface<tensor<3x4xi32> attr-dict>`
  - Array of interfaces:`hir.interface<tensor<4xhir.interface<...>> attr-dict>`
  - Tuple of interfaces:`hir.interface<tuple<hir.interface<...>,...> attr-dict>`
* A tuple can either have only interfaces or only primitive types as its members.
  - Enhancement: For brevity allow attr inside tensor and tuple within a 
      interface so that we don't need to nest interfaces.
* slice op syntax: `hir.slice %a ([1,i,j], [0] attr-dict)`
* slice op indices must be either constant or mlir index type.
* The writes to interfaces are volatile, i.e. reads must occur in the same 
    cycle.

###memref
* memref is an interface to memory
* The underlying memory element can be a dram, uram, bram, lutram, reg or latch.
* Operations: alloc, cast, load, store
* Attributes: load, store
* Syntax: `hir.memref<MxNxelement-type, attr-dict>`
* The `attr-dict` specifies the banked dims and load/store delays.
* `element-type` can be IntType,FloatType,ComplexType and TupleType.
* Indices for banked dims in load/store op must be of type index or constants.

###Lowering
* Convert a fully packed memref to a `interface<tuple<...>>`
* For banked memrefs use `interface<tensor<interface<tuple<...>>`
* remove the casts between memrefs and interfaces.
* Lower load/store to read and write ops.
* Simplify interface of tuple to interface.
* Convert read/write/call/assign op of more complex interfaces to a slice op 
    followed by read/write of a simple interface (i.e. the interface element 
    type must be i32 or f32).
* Simplify slice of multiple elements to slice + zip.
* If the slice is nested inside unroll_for loop then
  - Create a new slice op at the outermost level where the interface is defined.
  - Use the output interface of the slice instead of the original interface
      inside unroll loop nest.
  - If the step size in any of the unroll_for loops is greater than 1 then add a
      new induction var with step size 1 and use it to index the new interface 
      in the slice op inside the unroll loop.
