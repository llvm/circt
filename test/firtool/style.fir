; RUN: firtool %s --ir-fir| FileCheck %s --check-prefix=DEFAULT
; RUN: not firtool --ir-fir --lowering-options=bad-option %s 2>&1 | FileCheck %s --check-prefix=BADOPTION
; RUN: firtool --ir-fir --lowering-options=noAlwaysComb %s | FileCheck %s --check-prefix=OPTIONS

FIRRTL version 4.0.0
circuit test :
  public module test :

; DEFAULT: module {
; BADOPTION: lowering-options option: unknown style option 'bad-option'
; OPTIONS: module attributes {circt.loweringOptions = "noAlwaysComb"} {
