; RUN: firtool %s | FileCheck %s --check-prefix=DEFAULT
; RUN: not firtool --circt-lowering-options=bad-option %s 2>&1 | FileCheck %s --check-prefix=BADOPTION
; RUN: firtool --circt-lowering-options=noAlwaysFF %s | FileCheck %s --check-prefix=NOALWAYSFF

circuit test :
  module test :

; DEFAULT: module {
; BADOPTION: circt-lowering-options option: unknown style option 'bad-option'
; NOALWAYSFF: module attributes {circt.loweringOptions = "noAlwaysFF"} {