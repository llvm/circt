#HIR
HIR is an mlir dialect for hardware description.

#TODO

* Match data type between function call and function def.
* Support external def.

#Observations

* Optimizations enabled by multiple drivers for bus type.
    - Loop fusion between producer and consumer.
    - if two instances of a bus are written-to at time %t1 and %t2 then we know
        that %t1 != %t2 (can be used to fuse hardware blocks).


#Enhancement proposals : Ops

##AllocaOp
* Syntax: `hir.alloca ("hardware-block") : type`
* `hardware-block` can be one of the following - Function symbol,
    uram/bram/lutram/reg/latch or nothing.
* `hir.alloca nothing:...` only creates the interface but does not connect 
    it to anything.

## unroll_for loop
* Allow multiple induction variables. 
* Sytax: `hir.unroll_for (i,j) = (0,0) to (10,100) step (1,10) ...`
* The type of the induction var is always mlir index type.
* Add support for iter_args.

## commentOp.
* The comment op is a nop and is used to optionally inserts comments in the 
    generated verilog.
* Syntax: `hir.comment string-attr` 

## CastOp
* Syntax: `hir.cast %v : #type1 -> #type2`

#Enhancement proposals : Types

## TimeType
* Allow casting between group<i1> and TimeType.

##FuncType
* Syntax: `!hir.func<()->()>`
* Operations: hir.call, hir.cast

## Protocols
* Syntax: `!hir.protocol<
    bus=("valid" : out i1, "data" : out), 
    split="m_valid_split">`
* If type is undefined then the corresponding element's width will be passed as
    a verilog parameter to the split function.

##BusType
* Syntax: `!hir.bus<in/out BuiltinType, ..., proto #protocol>`
* Operations: hir.send, hir.recv.
* Allow nesting between Tensor, Tuple and BusType but all buses in the tensor 
    must have the same direction.

#Lowerings

## Codegen restrictions before lowering.
* Only allow select into simple types, i.e. hir.select output can not be array
    or group.

## MemrefType lowering
* Convert `!hir.memref` to 
    `Tuple<
    Tensor<!hir.bus<out i1, out i8>, proto valid>, 
    Tensor< !hir.bus<in i32>>>`
* For each use of the memref, use an hir.split function.
* Convert LoadOp/StoreOp to SelectOp + SendOp + RecvOp.

## BusType lowering
* perform loop-unrolling.
* Find number of uses per bus.
* For each bus that is used multiple times, use the split function to create an
    array and use the array elements.

##CallOp lowering
* Convert CallOp-on-symbol to AllocaOp + CallOp-on-var.

## FuncType lowering
* Convert `!hir.func` to `!hir.group`.
* Convert CallOp-on-var to multiple RecvOp/SendOp.
* Convert SelectOp to multiple SelectOps plus one GroupOp.

## ArrayType lowering
* Convert array-of-groups to group-of-arrays.
* Rearrange SelectOp indices accordingly (SelectOp should be to only simple
    types).

## GroupType lowering
* [Not urgent] Flatten nested groups into a single group. 
  * Update SelectOp with flattened index to select item.
* Break group of multiple items to separate items.
  * Replace the group-name+first-index in SelectOp with the correct item.
* Replace group-of-single-array to just the array. Update SelectOp.

## Codegen restrictions after lowering.
* Only supports groups of BuiltinTypes (int/float).
* Only support array of BuiltinTypes.
* No MemrefType and CallOp.


