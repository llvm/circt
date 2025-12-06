; Test the --enable-hw-flatten-modules flag and related options

; RUN: firtool %s --ir-hw | FileCheck %s --check-prefix=DISABLED
; RUN: firtool %s --ir-hw --enable-hw-flatten-modules | FileCheck %s --check-prefix=ENABLED-DEFAULT
; RUN: firtool %s --ir-hw --enable-hw-flatten-modules --hw-inline-all | FileCheck %s --check-prefix=INLINE-ALL
; RUN: firtool %s --ir-hw --enable-hw-flatten-modules --hw-inline-single-use | FileCheck %s --check-prefix=INLINE-SINGLE-USE
; RUN: firtool %s --ir-hw --enable-hw-flatten-modules --hw-inline-small --hw-small-threshold=3 | FileCheck %s --check-prefix=INLINE-SMALL
; RUN: firtool %s --ir-hw --enable-hw-flatten-modules --hw-inline-empty=false | FileCheck %s --check-prefix=NO-INLINE-EMPTY

FIRRTL version 4.0.0

circuit Top :
  ; Private module that is empty (only has output, passes through input)
  module EmptyChild :
    input x : UInt<4>
    output y : UInt<4>
    connect y, x

  ; Private module with single use
  module SingleUseChild :
    input a : UInt<4>
    output b : UInt<4>
    node temp = add(a, a)
    connect b, temp

  ; Private module with multiple uses
  module MultiUseChild :
    input p : UInt<4>
    output q : UInt<4>
    node result = mul(p, p)
    connect q, result

  public module Top :
    input in : UInt<4>
    output out : UInt<4>

    inst empty of EmptyChild
    connect empty.x, in

    inst single of SingleUseChild
    connect single.a, empty.y

    inst multi1 of MultiUseChild
    connect multi1.p, single.b

    inst multi2 of MultiUseChild
    connect multi2.p, single.b

    node sum = add(multi1.q, multi2.q)
    connect out, sum

; When disabled, all modules should be present
; DISABLED-LABEL: hw.module private @EmptyChild
; DISABLED-LABEL: hw.module private @SingleUseChild
; DISABLED-LABEL: hw.module private @MultiUseChild
; DISABLED-LABEL: hw.module @Top
; DISABLED: hw.instance "empty" @EmptyChild
; DISABLED: hw.instance "single" @SingleUseChild
; DISABLED: hw.instance "multi1" @MultiUseChild
; DISABLED: hw.instance "multi2" @MultiUseChild

; When enabled with default options (hwInlineEmpty=true by default), EmptyChild should be inlined
; ENABLED-DEFAULT-NOT: hw.module private @EmptyChild
; ENABLED-DEFAULT-LABEL: hw.module private @SingleUseChild
; ENABLED-DEFAULT-LABEL: hw.module private @MultiUseChild
; ENABLED-DEFAULT-LABEL: hw.module @Top
; ENABLED-DEFAULT-NOT: hw.instance "empty"
; ENABLED-DEFAULT: hw.instance "single" @SingleUseChild
; ENABLED-DEFAULT: hw.instance "multi1" @MultiUseChild
; ENABLED-DEFAULT: hw.instance "multi2" @MultiUseChild

; When --hw-inline-all is set, all private modules should be inlined
; INLINE-ALL-NOT: hw.module private @EmptyChild
; INLINE-ALL-NOT: hw.module private @SingleUseChild
; INLINE-ALL-NOT: hw.module private @MultiUseChild
; INLINE-ALL-LABEL: hw.module @Top
; INLINE-ALL-NOT: hw.instance "empty"
; INLINE-ALL-NOT: hw.instance "single"
; INLINE-ALL-NOT: hw.instance "multi1"
; INLINE-ALL-NOT: hw.instance "multi2"

; When --hw-inline-single-use is set, SingleUseChild and EmptyChild should be inlined
; (EmptyChild is also single-use and hwInlineEmpty=true by default)
; INLINE-SINGLE-USE-NOT: hw.module private @EmptyChild
; INLINE-SINGLE-USE-NOT: hw.module private @SingleUseChild
; INLINE-SINGLE-USE-LABEL: hw.module private @MultiUseChild
; INLINE-SINGLE-USE-LABEL: hw.module @Top
; INLINE-SINGLE-USE-NOT: hw.instance "empty"
; INLINE-SINGLE-USE-NOT: hw.instance "single"
; INLINE-SINGLE-USE: hw.instance "multi1" @MultiUseChild
; INLINE-SINGLE-USE: hw.instance "multi2" @MultiUseChild

; When --hw-inline-small is set with threshold=3, MultiUseChild should be inlined (has 2 ops)
; but SingleUseChild should not (has 4 ops: constant, extract, concat, output)
; EmptyChild should still be inlined (hwInlineEmpty=true by default)
; INLINE-SMALL-NOT: hw.module private @EmptyChild
; INLINE-SMALL-LABEL: hw.module private @SingleUseChild
; INLINE-SMALL-NOT: hw.module private @MultiUseChild
; INLINE-SMALL-LABEL: hw.module @Top
; INLINE-SMALL-NOT: hw.instance "empty"
; INLINE-SMALL: hw.instance "single" @SingleUseChild
; INLINE-SMALL-NOT: hw.instance "multi1"
; INLINE-SMALL-NOT: hw.instance "multi2"

; When --hw-inline-empty=false is explicitly set, EmptyChild should NOT be inlined
; NO-INLINE-EMPTY-LABEL: hw.module private @EmptyChild
; NO-INLINE-EMPTY-LABEL: hw.module private @SingleUseChild
; NO-INLINE-EMPTY-LABEL: hw.module private @MultiUseChild
; NO-INLINE-EMPTY-LABEL: hw.module @Top
; NO-INLINE-EMPTY: hw.instance "empty" @EmptyChild
; NO-INLINE-EMPTY: hw.instance "single" @SingleUseChild
; NO-INLINE-EMPTY: hw.instance "multi1" @MultiUseChild
; NO-INLINE-EMPTY: hw.instance "multi2" @MultiUseChild

