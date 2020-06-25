#ifndef CIRCT_DIALECT_LLHD_IR_LLHDOPS
#define CIRCT_DIALECT_LLHD_IR_LLHDOPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "circt/Dialect/LLHD/IR/LLHDBase.td"

//===----------------------------------------------------------------------===//
// Value Manipulation Operations
//===----------------------------------------------------------------------===//

def LLHD_ConstOp : LLHD_Op<"const", [ConstantLike, NoSideEffect]> {
    let summary = "Introduce a new constant.";

    let description = [{
        The `llhd.const` instruction introduces a new constant value as an
        SSA-operator.
        Legal types are integers and time. Note: Signals
        are not legal to define using `llhd.const`, use the `llhd.sig`
        instruction for that.

        **Custom syntax:**

        ```
        const-op ::= ssa-id `=` `llhd.const` attribute-value attr-dict `:` result-type
        ```

        **Examples:**

        ```
        %0 = llhd.const 1 : i64
        %1 = llhd.const #llhd.time<1ns, 2d, 3d> : !llhd.time
        ```
    }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyTypeOf<[AnySignlessInteger, LLHD_TimeType]>:$out);

    let hasFolder = 1;
}

def LLHD_VecOp : LLHD_Op<"vec", [NoSideEffect, TypesMatchWith<
        "types and number of 'values' have to match the length and type of the 'result' vector",
        "result", "values",
        "std::vector<Type>($_self.cast<ShapedType>().getNumElements(), $_self.cast<ShapedType>().getElementType())"
    >]> {

    let summary = "Create a vector from a list of values.";

    let description = [{
        The `llhd.vec` operation allows to create a vector from a list of
        SSA-values. This allows for more flexibility compared to only using
        `std.constant` and the vector dialect operations.

        **Examples:**

        ```
        %c1 = llhd.const 1 : i32
        %c2 = llhd.const 2 : i32
        %c3 = llhd.const 3 : i32
        %vec = llhd.vec %c1, %c2, %c3 : vector<3xi32>
        %elem = vector.extractelement %vec[%c1 : i32] : vector<3xi32>
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs AnyVector:$result);

    let assemblyFormat = "$values attr-dict `:` type($result)";
}

def LLHD_TupleOp : LLHD_Op<"tuple", [NoSideEffect, TypesMatchWith<
        "types of 'values' have to match the type of the corresponding 'result' tuple element",
        "result", "values",
        "$_self.cast<TupleType>().getTypes()">
    ]> {

    let summary = "Create a tuple from a list of values.";

    let description = [{
        The `llhd.tuple` operation creates a tuple from a list of SSA-values.

        **Examples:**

        ```
        %c1 = llhd.const 1 : i32
        %c2 = llhd.const 2 : i2
        %sig = llhd.sig "sig_name" %c1 : i32
        %vec = constant dense<[1, 2]> : vector<2xi32>
        %tuple = llhd.tuple %c1, %c2, %vec, %sig : tuple<i32, i2, vector<2xi32>, !llhd.sig<i32>>
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs AnyTuple:$result);

    let assemblyFormat = "$values attr-dict `:` type($result)";
}

def LLHD_ExtsOp : LLHD_Op<"exts", [NoSideEffect,
        PredOpTrait<"'start' + size of the slice have to be smaller or equal to the 'target' size",
            CPred<"$start.cast<IntegerAttr>().getInt() + this->getSliceSize() <= this->getTargetSize()">>,
        SameTypeArbitraryWidth<"'target' and 'result' have to be both either signless integers, signals or vectors with the same element type",
            "$target", "$result">
        ]> {
    let summary = "Extract a slice of consecutive elements.";

    let description = [{
        The `llhd.exts` operation allows access to a slice of the `$target`
        operand. The `$start` attribute defines the index of the first element.
        The return type is the same as `$target` but with the width of the
        specified result type.
        If `%target` is a signal, a new subsignal aliasing the slice will be
        returned.

        **Examples:**
        ```
        %0 = llhd.const 123 : i32
        %1 = llhd.exts %0, 0 : i32 -> i2

        %2 = llhd.sig %0 : i32
        %3 = llhd.exts %2, 0 : !llhd.sig<i32> -> !llhd.sig<i5>
        ```
    }];

    let arguments = (ins AnyTypeOf<[AnySignlessInteger, LLHD_AnySigType, AnyVector]>: $target,
                         IndexAttr: $start);
    let results = (outs AnyTypeOf<[AnySignlessInteger, LLHD_AnySigType, AnyVector]>: $result);

    let assemblyFormat = [{
        $target `,` $start attr-dict `:` type($target) `->` type($result)
    }];

    let extraClassDeclaration = [{
        unsigned getSliceSize() {
            Type sliceType = result().getType();
            if (auto vec = sliceType.dyn_cast<VectorType>())
                return vec.getNumElements();
            if (auto sig = sliceType.dyn_cast<llhd::SigType>()) {
                Type ty = sig.getUnderlyingType();
                if (auto vec = ty.dyn_cast<VectorType>())
                    return vec.getNumElements();
                if (auto tup = ty.dyn_cast<TupleType>())
                    return tup.size();
                return ty.getIntOrFloatBitWidth();
            }
            return sliceType.getIntOrFloatBitWidth();
        }

        unsigned getTargetSize() {
            Type targetType = target().getType();
            if (auto vec = targetType.dyn_cast<VectorType>())
                return vec.getNumElements();
            if (auto sig = targetType.dyn_cast<llhd::SigType>()) {
                Type ty = sig.getUnderlyingType();
                if (auto vec = ty.dyn_cast<VectorType>())
                    return vec.getNumElements();
                if (auto tup = ty.dyn_cast<TupleType>())
                    return tup.size();
                return ty.getIntOrFloatBitWidth();
            }
            return targetType.getIntOrFloatBitWidth();
        }
    }];
}

def LLHD_DextsOp : LLHD_Op<"dexts", [NoSideEffect,
         SameTypeArbitraryWidth<"'target' and 'result' types have to match apart from their width",
            "$target", "$result">,
         PredOpTrait<"the result width cannot be larger than the target operand width",
            CPred<"this->getTargetWidth() >= this->getSliceWidth()">>
        ]> {
    let summary = "Dynamically extract a slice of consecutive elements";

    let description = [{
        The `llhd.dexts` operation allows to dynamically access a slice of the
        `$target` operand, starting at the index given by the `$start` operand.
        The resulting slice length is defined by the result type.
        The `$target` operand kind has to match the result kind.
        If `$target` is a vector, only the number of elements can change, while
        the element type has to remain the same.

        **Examples:**
        ```
        %0 = llhd.const 0x0f0 : i12
        %1 = llhd.const 4 : i3

        %3 = llhd.dexts %0, %1 : (i12, i3) -> i4    // %3: 0xf
        ```
    }];

    let arguments = (ins AnyTypeOf<[AnySignlessInteger, AnyVector, LLHD_AnySigType]>: $target,
                         AnySignlessInteger: $start);
    let results = (outs AnyTypeOf<[AnySignlessInteger, AnyVector, LLHD_AnySigType]>: $result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

    let extraClassDeclaration = [{
        unsigned getSliceWidth();
        unsigned getTargetWidth();
    }];
}

def LLHD_InssOp : LLHD_Op<"inss",
        [NoSideEffect, AllTypesMatch<["target", "result"]>,
        SameTypeArbitraryWidth<"'target' and 'slice' have to be both either signless integers or vectors with the same element type",
            "$target", "$slice">,
         PredOpTrait<"'start' + size of the 'slice' have to be smaller or equal to the 'target' size",
            CPred<"$start.cast<IntegerAttr>().getInt() + this->getSliceSize() <= this->getTargetSize()">>
        ]> {

    let summary = "Insert a slice of consecutive elements.";

    let description = [{
        The `llhd.inss` operation allows insertion of a slice represented by the
        `$slice` operand into the `$target` operand. The `$start` attribute
        defines the index of the first element. The return type is the same as
        `$target`. Note that the `$target` is not changed, but a new value with
        the slice inserted is returned.

        **Examples:**

        ```mlir
        %itarget = llhd.const 123 : i32
        %islice = llhd.const 2 : i2
        %0 = llhd.inss %itarget, %islice, 0 : i32, i2

        %vtarget = constant dense<[1,2,3]> : vector<3xi32>
        %vslice = constant dense<[4,5]> : vector<2xi32>
        %1 = llhd.inss %vtarget, %vslice, 0 : vector<3xi32>, vector<2xi32>
        ```
    }];

    let arguments = (ins AnyTypeOf<[AnySignlessInteger, AnyVector]>: $target,
                         AnyTypeOf<[AnySignlessInteger, AnyVector]>: $slice,
                         IndexAttr: $start);

    let results = (outs AnyTypeOf<[AnySignlessInteger, AnyVector]>: $result);

    let assemblyFormat = [{
        $target `,` $slice `,` $start attr-dict `:`
        type($target) `,` type($slice)
    }];

    let extraClassDeclaration = [{
        unsigned getSliceSize() {
            Type sliceType = slice().getType();
            if (auto vec = sliceType.dyn_cast<VectorType>())
                return vec.getNumElements();
            return sliceType.getIntOrFloatBitWidth();
        }

        unsigned getTargetSize() {
            Type targetType = target().getType();
            if (auto vec = targetType.dyn_cast<VectorType>())
                return vec.getNumElements();
            return targetType.getIntOrFloatBitWidth();
        }
    }];
}

def LLHD_InsfOp : LLHD_Op<"insf",
        [NoSideEffect, AllTypesMatch<["target", "result"]>,
         PredOpTrait<"'index' has to be smaller than the 'target' size",
            CPred<"$index.cast<IntegerAttr>().getInt() < this->getTargetSize()">>,
         TypesMatchWith<"'element' type has to match type at 'index' of 'target'",
            "target", "element", "this->getElementTypeAtIndex($index.cast<IntegerAttr>().getInt())">
        ]> {

    let summary = "Insert an element into a vector or tuple.";

    let description = [{
        The `llhd.insf` operation allows insertion of an element represented by
        the `$element` operand into the `$target` operand. The `$index`
        attribute defines the index where to insert the element. The return type
        is the same as `$target`. Note that the `$target` is not changed, but a
        new value with the element inserted is returned.

        **Examples:**

        ```mlir
        %target = constant dense<[1,2,3]> : vector<3xi8>
        %element = llhd.const 2 : i8
        %0 = llhd.insf %target, %element, 0 : vector<3xi8>, i8

        %tuptarget = llhd.tuple %element, %target : tuple<i8, vector<3xi8>
        %newelement = llhd.const 4 : i8
        %1 = llhd.insf %tuptarget, %newelement, 0 : tuple<i8, vector<3xi8>>, i8
        ```
    }];

    let arguments = (ins AnyTypeOf<[AnyVector, AnyTuple]>: $target,
                         AnyType: $element,
                         IndexAttr: $index);

    let results = (outs AnyTypeOf<[AnyVector, AnyTuple]>: $result);

    let assemblyFormat = [{
        $target `,` $element `,` $index attr-dict `:`
        type($target) `,` type($element)
    }];

    let extraClassDeclaration = [{
        int64_t getTargetSize() {
            if (auto vec = target().getType().dyn_cast<VectorType>())
                return vec.getNumElements();
            return target().getType().dyn_cast<TupleType>().size();
        }

        Type getElementTypeAtIndex(int64_t index) {
            if (auto vec = target().getType().dyn_cast<VectorType>())
                return vec.getElementType();
            return target().getType().dyn_cast<TupleType>().getTypes()[index];
        }
    }];
}

//===----------------------------------------------------------------------===//
//=== Signal Operations
//===----------------------------------------------------------------------===//

def LLHD_SigOp : LLHD_Op<"sig",
        [HasParent<"EntityOp">,
         TypesMatchWith<"type of 'init' and underlying type of 'signal' have to match.",
                        "init", "result",
                        "SigType::get($_self)">]> {
    let summary = "Create a signal.";

    let description = [{
       The `llhd.sig` instruction introduces a new signal in the IR. The input
       operand determines the initial value carried by the signal, while the
       result type will always be a signal carrying the type of the init operand.
       A signal defines a unique name within the entity it resides in. Signals
       can only be allocated within entities.

       **Custom syntax:**

       ```
       sig-op ::= ssa-id `=` `llhd.sig` sig-name ssa-init attr-dict `:` init-type
       ```

       **Examples:**

       ```
       %init_i64 = llhd.const 123 : i64
       %sig_i64 = llhd.sig "foo" %init_64 : i64

       %init_i1 = llhd.const 1 : i1
       %sig_i1 = llhd.sig "bar" %init_i1 : i1
       ```
       The first `llhd.sig` instruction creates a new signal named "foo", carrying an `i64`
       type with initial value of 123, while the second one creates a new signal
       named "bar", carrying an `i1` type with initial value of 1.
    }];

    let arguments = (ins StrAttr:$name, AnySignlessInteger:$init);
    let results = (outs LLHD_AnySigType:$result);

    let assemblyFormat = "$name $init attr-dict `:` type($init)";
}

def LLHD_PrbOp : LLHD_Op<"prb",
        [NoSideEffect,
         TypesMatchWith<"type of 'result' and underlying type of 'signal' have to match.",
                        "signal", "result",
                        "$_self.cast<SigType>().getUnderlyingType()">]> {
    let summary = "Probe a signal.";

    let description = [{
       The `llhd.prb` instruction probes a signal and returns the value it
       currently carries as a new SSA operand. The result type is always
       the type carried by the signal.

        **Custom syntax:**

        ```
        prb-op ::= ssa-id `=` `llhd.prb` ssa-sig attr-dict `:` !llhd.sig<type>
        ```

        **Examples:***

        ```
        %const_i1 = llhd.const 1 : i1
        %sig_i1 = llhd.sig %const_i1 : i1
        %prbd = llhd.prb %sig_i1 : !llhd.sig<i1>
        ```
    }];

    let arguments = (ins LLHD_AnySigType:$signal);
    let results = (outs AnySignlessInteger:$result);

    let assemblyFormat = "$signal attr-dict `:` type($signal)";
}

def LLHD_DrvOp : LLHD_Op<"drv",
        [TypesMatchWith<"type of 'value' and underlying type of 'signal' have to match.",
                        "signal", "value",
                        "$_self.cast<SigType>().getUnderlyingType()">]> {
    let summary = "Drive a value into a signal.";

    let description = [{
        The `llhd.drv` operation drives a new value onto a signal. A time
        operand also has to be passed, which specifies the frequency at which
        the drive will be performed. An optional enable value can be passed as
        last argument. In this case the drive will only be performed if the
        value is 1. In case no enable signal is passed the drive will always be
        performed. This operation does not define any new SSA operands.

        **Custom Syntax:**

        ```
        drv-op ::= `llhd.drv` ssa-signal `,` ssa-const `after` ssa-time (`if` ssa-enable)? `:` !llhd.sig<const-type>
        ```

        **Examples:**

        ```
        %init = llhd.const 1 : i1
        %en = llhd.const 0 : i1
        %time = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
        %sig = llhd.sig %init : i1
        %new = llhd.not %init : i1

        llhd.drv %sig, %new after %time : !llhd.sig<i1>
        llhd.drv %sig, %new after %time if %en : !llhd.sig<i1>
        ```
    }];

    let arguments = (ins LLHD_AnySigType:$signal,
                         AnySignlessInteger:$value,
                         LLHD_TimeType:$time,
                         Optional<I1>:$enable);

    let assemblyFormat = [{
        $signal `,` $value `after` $time ( `if` $enable^ )? attr-dict `:` type($signal)
    }];
}

//===----------------------------------------------------------------------===//
//=== Control Flow Operations
//===----------------------------------------------------------------------===//

def LLHD_TerminatorOp : LLHD_Op<"terminator", [Terminator, HasParent<"EntityOp">]> {
    let summary = "Dummy terminator";
    let description = [{
        The `"llhd.terminator"` op is a dummy terminator for an `EntityOp` unit.
        It provides no further meaning other than ensuring correct termination
        of an entitiy's region. This operation provides no custom syntax and
        should never explicitly appear in LLHD's custom syntax.
    }];

    let parser = ?;
    let printer = ?;
}

def LLHD_WaitOp : LLHD_Op<"wait",
        [Terminator, AttrSizedOperandSegments, HasParent<"ProcOp">,
         DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let summary = "Suspends execution of a process.";
    let description = [{
        The `wait` instruction suspends execution of a process until any of the
        observed signals change or a fixed time interval has passed. Execution
        resumes at the specified basic block with the passed arguments.
        * This is a terminator instruction.
        * This instruction is only allowed in processes (`llhd.proc`).

        **Syntax:**
        ```
        wait-op ::= `llhd.wait` ( `for` ssa-time `,` )? ( `(` ssa-list-obs `:` type-list-obs `)` `,` )? successor-dest ( `(` ssa-list-dest-arguments `:` type-list-dest-arguments `)` )? attr-dict
        ```
        Notes:
        * `ssa-list-obs`, `ssa-list-dest-arguments`, `type-list-dest-arguments` and `type-list-obs` are comma-separated lists of 0 or more elements.

        **Examples:**
        ```
        llhd.wait ^bb1
        llhd.wait for %time, ^bb1(%time : !llhd.time)
        llhd.wait (%0, %1 : !llhd.sig<i64>, !llhd.sig<i1>), ^bb1(%1 : !llhd.sig<i1>)
        llhd.wait for %time, (%0, %1 : !llhd.sig<i64>, !llhd.sig<i1>), ^bb1(%1, %0 : !llhd.sig<i1>, !llhd.sig<i64>)
        ```
    }];

    let arguments = (ins Variadic<LLHD_AnySigType>:$obs,
                         Optional<LLHD_TimeType>:$time,
                         Variadic<AnyType>:$destOps);

    let successors = (successor AnySuccessor:$dest);

    let assemblyFormat = [{
        (`for` $time^ `,`)? (`(`$obs^ `:` type($obs)`)` `,`)?
        $dest (`(` $destOps^ `:` type($destOps) `)`)? attr-dict
    }];
}

def LLHD_HaltOp : LLHD_Op<"halt", [Terminator, HasParent<"ProcOp">]> {
    let summary = "Terminates execution of a process.";
    let description = [{
        The `halt` instruction terminates execution of a process. All processes
        must halt eventually or consist of an infinite loop.
        * This is a terminator instruction
        * This instruction is only allowed in processes (`llhd.proc`).

        **Syntax:**
        ```
        halt-op ::= `llhd.halt`
        ```

        **Examples:**
        ```
        llhd.halt
        ```
    }];

    let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
//=== Bitwise Operations
//===----------------------------------------------------------------------===//

def LLHD_NotOp : LLHD_ArithmeticOrBitwiseOp<"not", []> {
    let summary = "Bitwise NOT";
    let description = [{
        Takes an integer of any width or a nine-valued-logic (IEEE 1164) value
        of any width as input. Flips each bit of a value. The result always has
        the exact same type.

        **Syntax:**
        ```
        not-op ::= ssa-id `=` `llhd.not` ssa-value attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 0 : i32
        %1 = llhd.not %0 : i32
        ```

        **Truth Table for `iN`:**

        | `not` |  0  |  1  |
        |:-----:|:---:|:---:|
        |       |  1  |  0  |

        **Truth Table for `lN`:**

        | `not` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |       |  U  |  X  |  1  |  0  |  X  |  X  |  1  |  0  |  X  |
    }];

    let arguments = (ins AnySignlessInteger:$value);
    let hasFolder = 1;
}

def LLHD_AndOp : LLHD_ArithmeticOrBitwiseOp<"and", [Commutative]> {
    let summary = "Bitwise AND";
    let description = [{
        Takes two integers of the same width or two nine-valued-logic (IEEE 1164)
        values of the same width as input. Calculates the bitwise AND. The
        result is always of the exact same type as the two inputs.

        **Syntax:**
        ```
        and-op ::= ssa-id `=` `llhd.and` ssa-lhs `,` ssa-rhs attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 0 : i32
        %1 = llhd.and %0, %0 : i32
        ```

        **Truth Table for `iN`:**

        | `and` |  0  |  1  |
        |:-----:|:---:|:---:|
        |   0   |  0  |  0  |
        |   1   |  0  |  1  |

        **Truth Table for `lN`:**

        | `and` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |   U   |  U  |  U  |  0  |  U  |  U  |  U  |  0  |  U  |  U  |
        |   X   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
        |   0   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
        |   1   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   Z   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
        |   W   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
        |   L   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
        |   H   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   -   |  U  |  X  |  0  |  X  |  X  |  X  |  0  |  X  |  X  |
    }];

    let arguments = (ins AnySignlessInteger:$lhs,
                         AnySignlessInteger:$rhs);
    let hasFolder = 1;
}

def LLHD_OrOp : LLHD_ArithmeticOrBitwiseOp<"or", [Commutative]> {
    let summary = "Bitwise OR";
    let description = [{
        Takes two integers of the same width or two nine-valued-logic (IEEE 1164)
        values of the same width as input. Calculates the bitwise OR. The
        result is always of the exact same type as the two inputs.

        **Syntax:**
        ```
        or-op ::= ssa-id `=` `llhd.or` ssa-lhs `,` ssa-rhs attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 0 : i32
        %1 = llhd.or %0, %0 : i32
        ```

        **Truth Table for `iN`:**

        | `or`  |  0  |  1  |
        |:-----:|:---:|:---:|
        |   0   |  0  |  1  |
        |   1   |  1  |  1  |

        **Truth Table for `lN`:**

        | `or`  |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |   U   |  U  |  U  |  U  |  1  |  U  |  U  |  U  |  1  |  U  |
        |   X   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
        |   0   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   1   |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
        |   Z   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
        |   W   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
        |   L   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   H   |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
        |   -   |  U  |  X  |  X  |  1  |  X  |  X  |  X  |  1  |  X  |
    }];

    let arguments = (ins AnySignlessInteger:$lhs,
                         AnySignlessInteger:$rhs);
    let hasFolder = 1;
}

def LLHD_XorOp : LLHD_ArithmeticOrBitwiseOp<"xor", [Commutative]> {
    let summary = "Bitwise XOR";
    let description = [{
        Takes two integers of the same width or two nine-valued-logic (IEEE 1164)
        values of the same width as input. Calculates the bitwise XOR. The
        result is always of the exact same type as the two inputs.

        **Syntax:**
        ```
        xor-op ::= ssa-id `=` `llhd.xor` ssa-lhs `,` ssa-rhs attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 0 : i32
        %1 = llhd.xor %0, %0 : i32
        ```

        **Truth Table for `iN`:**

        | `xor` |  0  |  1  |
        |:-----:|:---:|:---:|
        |   0   |  0  |  1  |
        |   1   |  1  |  0  |

        **Truth Table for `lN`:**

        | `xor` |  U  |  X  |  0  |  1  |  Z  |  W  |  L  |  H  |  -  |
        |:-----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        |   U   |  U  |  U  |  U  |  U  |  U  |  U  |  U  |  U  |  U  |
        |   X   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
        |   0   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   1   |  U  |  X  |  1  |  0  |  X  |  X  |  1  |  0  |  X  |
        |   Z   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
        |   W   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
        |   L   |  U  |  X  |  0  |  1  |  X  |  X  |  0  |  1  |  X  |
        |   H   |  U  |  X  |  1  |  0  |  X  |  X  |  1  |  0  |  X  |
        |   -   |  U  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |  X  |
    }];

    let arguments = (ins AnySignlessInteger:$lhs,
                         AnySignlessInteger:$rhs);
    let hasFolder = 1;
}

def LLHD_ShlOp : LLHD_Op<"shl", [NoSideEffect]> {
    let summary = "Shifts a value to the left by a given amount.";
    let description = [{
        The type of the base value and the hidden value must be the same, but
        may differ in the number of bits or elements. The result always has the
        same type (including width) of the base value.
        The instruction is transparent to signals and pointers. For example,
        passing a signal as argument will shift the underlying value and return
        a signal to the shifted value.
        Allowed (underlying) types are signless integers, nine-valued-logic values
        and arrays. The shift amount has to be a signless integer. A shift amount
        bigger than the number of bits or elements of the hidden value is undefined.
        The hidden value is uncovered by non-zero shift amounts. E.g. consider
        the four bit values `base = 0xf`, `hidden = 0xc` shifted by an amount of
        three result in `0xe`.

        **Syntax:**
        ```
        shl-op ::= ssa-id `=` `llhd.shl` ssa-base `,` ssa-hidden `,` ssa-amount attr-dict `:` `(` base-type `,` hidden-type `,` amount-type `)` `->` result-type
        ```

        **Examples:**
        ```
        %0 = llhd.shl %base, %hidden, %amount : (i4, i2, i2) -> i4
        ```
    }];

    // TODO: adjust type T and Th to include arrays and pointers
    let arguments = (ins AnyTypeOf<[AnySignlessInteger, LLHD_AnySigType]>:$base,
                         AnyTypeOf<[AnySignlessInteger, LLHD_AnySigType]>:$hidden,
                         AnySignlessInteger:$amount);
    let results = (outs AnyTypeOf<[AnySignlessInteger, LLHD_AnySigType]>:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

    let verifier = [{ return ::verify(*this); }];

    let hasFolder = 1;
}

def LLHD_ShrOp : LLHD_Op<"shr", [NoSideEffect]> {
    let summary = "Shifts a value to the right by a given amount.";
    let description = [{
        The type of the base value and the hidden value must be the same, but
        may differ in the number of bits or elements. The result always has the
        same type (including width) of the base value.
        The instruction is transparent to signals and pointers. For example,
        passing a signal as argument will shift the underlying value and return
        a signal to the shifted value.
        Allowed (underlying) types are signless integers, nine-valued-logic values
        and arrays. The shift amount has to be a signless integer. A shift amount
        bigger than the number of bits or elements of the hidden value is undefined.
        The hidden value is uncovered by non-zero shift amounts. E.g. consider
        the four bit values `base = 0xf`, `hidden = 0xc` shifted by an amount of
        three result in `0x9`.

        **Syntax:**
        ```
        shr-op ::= ssa-id `=` `llhd.shr` ssa-base `,` ssa-hidden `,` ssa-amount attr-dict `:` `(` base-type `,` hidden-type `,` amount-type `)` `->` result-type
        ```

        **Examples:**
        ```
        %0 = llhd.shr %base, %hidden, %amount : (i4, i2, i2) -> i4
        ```
    }];

    // TODO: adjust type T and Th to include arrays and pointers
    let arguments = (ins AnyTypeOf<[AnySignlessInteger, LLHD_AnySigType]>:$base,
                         AnyTypeOf<[AnySignlessInteger, LLHD_AnySigType]>:$hidden,
                         AnySignlessInteger:$amount);
    let results = (outs AnyTypeOf<[AnySignlessInteger, LLHD_AnySigType]>:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

    let verifier = [{ return ::verify(*this); }];

    let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
//=== Arithmetic Operations
//===----------------------------------------------------------------------===//

def LLHD_NegOp : LLHD_ArithmeticOrBitwiseOp<"neg", []> {
    let summary = "Negate a value.";
    let description = [{
        The operand and result always have the same type. The type has to be a
        signless integer of any width. Although, only signless integers are
        allowed, this instruction applies two's complement negation of the
        integer, basically treating it as a signed integer.

        **Syntax:**
        ```
        neg-op ::= ssa-id `=` `llhd.neg` ssa-value attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 42 : i32
        %1 = llhd.neg %0 : i32
        ```
    }];

    let arguments = (ins AnySignlessInteger:$value);
    let hasFolder = 1;
}

def LLHD_SModOp : LLHD_ArithmeticOrBitwiseOp<"smod", []> {
    let summary = "Signed modulo.";
    let description = [{
        This instruction computes the signed modulo of two signless integers of
        any width, treating the leading bit as sign. The operand and result
        types always have to be the same.
        To calculate the signed remainder of two integers, use `remi_signed`
        from the standard dialect.

        **Syntax:**
        ```
        smod-op ::= ssa-id `=` `llhd.smod` ssa-lhs `,` ssa-rhs attr-dict `:` type
        ```

        **Examples:**
        ```
        %0 = llhd.const 9 : i4
        %1 = llhd.const 4 : i4
        %2 = llhd.smod %0, %1 : i4
        ```
    }];

    let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs);
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
//=== Units
//===----------------------------------------------------------------------===//

def LLHD_EntityOp : LLHD_Op<"entity", [Symbol, FunctionLike, IsolatedFromAbove, SingleBlockImplicitTerminator<"TerminatorOp">, DeclareOpInterfaceMethods<CallableOpInterface>]> {
    let summary = "Create an entity.";
    let description = [{
        The `llhd.entity` operation defines a new entity unit. An entity
        represents the data-flow description of how a circuit's output values
        change in reaction to changing input values.
        An entity contains one region with a single block and an implicit
        `TerminatorOp` terminator. Both the block name and terminator are
        omitted in the custom syntax. No further blocks and control-flow are
        legal inside an entity.

        **Custom syntax:**
        ```
        entity-op ::= `llhd.entity` entity-symbol `(` arg-list `)` `->` `(` out-list `)` attr-dict entity-region
        ```

        **Examples:**
        ```
        llhd.entity @Foo () -> () {
            %0 = llhd.const 0 : i1
            %toggle = llhd.sig %0 : i1 -> !llhd.sig<i1>
            %1 = llhd.prb %toggle : !llhd.sig<i1> -> i1
            %2 = llhd.not %1 : i1
            %dt = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
            llhd.drv %toggle, %2, %dt : !llhd.sig<i1>, i1, !llhd.time
        }
        ```
    }];

    let arguments = (ins I64Attr:$ins);
    let regions = (region SizedRegion<1>:$body);

    let verifier = [{ return ::verify(*this); }];

    let extraClassDeclaration = [{
        friend class OpTrait::FunctionLike<EntityOp>;

        // use FunctionLike traits's getBody method
        using OpTrait::FunctionLike<EntityOp>::getBody;

        /// Hooks for the input/output type enumeration in FunctionLike.
        unsigned getNumFuncArguments() { return getType().getNumInputs(); }
        unsigned getNumFuncResults() { return getType().getNumResults(); }

        /// Hook for FunctionLike verifier.
        LogicalResult verifyType();

        /// Verifies the body of the function.
        LogicalResult verifyBody();
    }];
}

def LLHD_ProcOp : LLHD_Op<"proc", [Symbol, FunctionLike, IsolatedFromAbove, DeclareOpInterfaceMethods<CallableOpInterface>]> {
    let summary = "Create a process";
    let description = [{
        A `llhd.proc` represents control-flow in a timed fashion. It allows a
        procedural description of how a circuit's output signals change in
        reaction to changing input signals. It has a region with arbitrarily
        many basic blocks. The first block is the entry block and cannot be
        targeted by the terminators. It uses `llhd.wait` as a terminator to add
        timed control-flow. Immediate control-flow with `br` or `cond_br` is
        also possible. Every process must either contain an infinite loop or
        terminate with the `llhd.halt` terminator.

        How does a process compare to functions and entities?

        | Unit     | Paradigm     | Timing    | Models                         |
        |----------|--------------|-----------|--------------------------------|
        | Function | control-flow | immediate | Computation in zero time       |
        | Process  | control-flow | timed     | Behavioral circuit description |
        | Entity   | data-flow    | timed     | Structural circuit description |

        Syntax:
        ```
        proc-op ::= `llhd.proc` proc-symbol `(` ssa-input-list `)` `->` `(` ssa-output-list `)` attr-dict `{` proc-region `}`
        ```

        Examples:
        ```
        llhd.proc @example(%in0 : !llhd.sig<i64>, %in1 : !llhd.sig<i1>) -> (%out2 : !llhd.sig<i1>) {
            br ^bb1
        ^bb1:
            llhd.halt
        }
        ```
    }];

    let arguments = (ins I64Attr:$ins);
    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
        friend class OpTrait::FunctionLike<ProcOp>;

        /// Hooks for the input/output type enumeration in FunctionLike.
        unsigned getNumFuncArguments() { return getType().getNumInputs(); }
        unsigned getNumFuncResults() { return getType().getNumResults(); }

        /// Hook for FunctionLike verifier.
        LogicalResult verifyType();

        /// Verifies the body of the function.
        LogicalResult verifyBody();
    }];

    let verifier = [{ return ::verify(*this); }];
}

def LLHD_InstOp : LLHD_Op<"inst", [CallOpInterface, HasParent<"EntityOp">,
        AttrSizedOperandSegments]> {
    let summary = "Instantiates a process or entity.";
    let description = [{
        Instantiates a process or entity and thus allows to build hierarchies.
        Can only be used within an entity. An instance defines a unique name
        within the entity it resides in.

        Syntax:
        ```
        inst-op ::= `llhd.inst` inst-name symbol-name `(` ssa-input-list `)` `->` `(` ssa-output-list `)` attr-dict `:` functional-type(ssa-input-list, ssa-output-list)
        ```

        Examples:
        ```
        llhd.inst "foo" @empty() -> () : () -> ()
        llhd.inst "bar" @proc_symbol() -> (%out0) : () -> !llhd.sig<i32>
        llhd.inst "baz" @entity_symbol(%in0, %in1) -> (%out0, %out1) : (!llhd.sig<i32>, !llhd.sig<i16>) -> (!llhd.sig<i8>, !llhd.sig<i4>)
        ```
    }];

    let arguments = (ins StrAttr:$name,
                         FlatSymbolRefAttr:$callee,
                         Variadic<LLHD_AnySigType>:$inputs,
                         Variadic<LLHD_AnySigType>:$outputs);

    let assemblyFormat = [{
        $name $callee `(` $inputs `)` `->` `(` $outputs `)` attr-dict `:`
        functional-type($inputs, $outputs)
    }];

    let extraClassDeclaration = [{
        StringRef getCallee() { return callee(); }
        FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return getAttrOfType<SymbolRefAttr>("callee");
        }
    }];

    let verifier = [{ return ::verify(*this); }];
}

def REG_MODE_LOW  : I64EnumAttrCase<"low", 0>;
def REG_MODE_HIGH : I64EnumAttrCase<"high", 1>;
def REG_MODE_RISE : I64EnumAttrCase<"rise", 2>;
def REG_MODE_FALL : I64EnumAttrCase<"fall", 3>;
def REG_MODE_BOTH : I64EnumAttrCase<"both", 4>;

def LLHD_RegModeAttr : I64EnumAttr<"RegMode", "",
        [REG_MODE_LOW, REG_MODE_HIGH, REG_MODE_RISE, REG_MODE_FALL,
         REG_MODE_BOTH]> {
    let cppNamespace = "::mlir::llhd";
}

def LLHD_RegModeArrayAttr : TypedArrayAttrBase<LLHD_RegModeAttr,
                                               "reg mode array attribute"> {}

def LLHD_RegOp : LLHD_Op<"reg", [HasParent<"EntityOp">, AttrSizedOperandSegments]> {
    let summary = "Represents a storage element";
    let description = [{
        This instruction represents a storage element. It drives its output onto
        the 'signal' value. An arbitrary amount of triggers can be added to the
        storage element. However, at least one is required. They are quadruples
        consisting of the new value to be stored if the trigger applies, the
        mode and trigger value which specify when this trigger has to be applied
        as well as a delay. Optionally, each triple may also have a gate
        condition, in this case the trigger only applies if the gate is one. If
        multiple triggers apply the left-most in the list takes precedence.

        There are five modes available:

        | Mode   | Meaning                                                     |
        |--------|-------------------------------------------------------------|
        | "low"  | Storage element stores `value` while the `trigger` is low. Models active-low resets and low-transparent latches. |
        | "high" | Storage element stores `value` while the `trigger` is high. Models active-high resets and high-transparent latches. |
        | "rise" | Storage element stores `value` upon the rising edge of the `trigger`. Models rising-edge flip-flops.
        | "fall" | Storage element stores `value` upon the falling edge of the `trigger`. Models falling-edge flip-flops.
        | "both" | Storage element stores `value` upon the a rising or a falling edge of the `trigger`. Models dual-edge flip-flops.

        This instruction may only be used in an LLHD entity.

        Syntax:

        ```
        reg-op ::= `llhd.reg` signal-ssa-value ( `,` `(` value-ssa-value `,` mode-string trigger-ssa-value `after` delay-ssa-value ( `if` gate-ssa-value )? `:` value-type )+ attr-dict `:` signal-type
        ```

        Examples:

        A rising, falling, and dual-edge triggered flip-flop:

        ```
        llhd.reg %Q, (%D, "rise" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
        llhd.reg %Q, (%D, "fall" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
        llhd.reg %Q, (%D, "both" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
        ```

        A rising-edge triggered flip-flop with active-low reset:

        ```
        llhd.reg %Q, (%init, "low" %RSTB after %T : !llhd.sig<i8>), (%D, "rise" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
        ```

        A rising-edge triggered enable flip-flop with active-low reset:

        ```
        llhd.reg %Q, (%init, "low" %RSTB after %T : !llhd.sig<i8>), (%D, "rise" %CLK after %T if %EN : !llhd.sig<i8>) : !llhd.sig<i8>
        ```

        A transparent-low and transparent-high latch:

        ```
        llhd.reg %Q, (%D, "low" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
        llhd.reg %Q, (%D, "high" %CLK after %T : !llhd.sig<i8>) : !llhd.sig<i8>
        ```

        An SR latch:

        ```
        %0 = llhd.const 0 : i1
        %1 = llhd.const 1 : i1
        llhd.reg %Q, (%0, "high" %R after %T : !llhd.sig<i1>), (%1, "high" %S after %T : !llhd.sig<i1>) : !llhd.sig<i1>
        ```
    }];

    let arguments = (ins LLHD_AnySigType:$signal,
                         LLHD_RegModeArrayAttr:$modes,
                         Variadic<AnyTypeOf<[LLHD_AnySigUnderlyingType,
                                             LLHD_AnySigType]>>:$values,
                         Variadic<I1>:$triggers,
                         Variadic<LLHD_TimeType>:$delays,
                         Variadic<I1>:$gates,
                         I64ArrayAttr:$gateMask);

    let extraClassDeclaration = [{
        static StringRef getModeAttrName() { return "modes"; }
        static RegMode getRegModeByName(StringRef name) {
            llvm::Optional<RegMode> optional =  symbolizeRegMode(name);
            assert(optional && "Invalid RegMode string.");
            return optional.getValue();
        }

        bool hasGate(unsigned index) {
            assert(index < gateMask().getValue().size() && "Index out of range.");
            return gateMask().getValue()[index].cast<IntegerAttr>().getInt() != 0;
        }

        Value getGateAt(unsigned index) {
            assert(index < gateMask().getValue().size() && "Index out of range.");
            if (!hasGate(index)) return Value();
            return gates()[gateMask().getValue()[index].cast<IntegerAttr>().getInt()-1];
        }

        RegMode getRegModeAt(unsigned index) {
            assert(index < modes().getValue().size() && "Index out of range.");
            return (RegMode)modes().getValue()[index].cast<IntegerAttr>().getInt();
        }
    }];

    let verifier = [{ return ::verify(*this); }];
}

#endif // CIRCT_DIALECT_LLHD_IR_LLHDOPS
