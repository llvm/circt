#HIR
HIR is an mlir dialect for hardware description.

#Enhancement proposals : Ops

##instantiateOp
* Syntax: `hir.instantiate hardware-block : interface-type`
* `hardware-block` can be one of the following - Function symbol,
    uram/bram/lutram/reg/latch or nothing.
* `hir.instantiate nothing:...` only creates the interface but does not connect 
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

##GroupType
* Syntax : `!hir.group<("in" i1, i32, "out" v1)>`
* Operations: hir.send, hir.recv, hir.select, hir.cast, hir.group
* Indices must be constants.

##ModuleType
* Syntax: `!hir.func<()->()>`
* Operations: hir.call, hir.cast

##ArrayType
* Syntax: `!hir.array<"in" [2x4xf32]>`
* Operations: hir.select, hir.array, hir.assign.
* Indices can be IndexType or Constants.

#Lowerings

## Codegen restrictions before lowering.
* Only allow select into simple types, i.e. hir.select output can not be array
    or group.

## MemrefType lowering
* Convert `!hir.memref` to `!hir.func` or `!hir.array<[...xfunc(...)->(...)]>`
* Convert LoadOp/StoreOp to SelectOp and CallOp.

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
