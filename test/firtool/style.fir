; RUN: firtool %s | FileCheck %s --check-prefix=DEFAULT
; RUN: not firtool --lowering-options=bad-option %s 2>&1 | FileCheck %s --check-prefix=BADOPTION
; RUN: firtool --lowering-options=alwaysFF %s | FileCheck %s --check-prefix=ALWAYSFF

circuit test :
  module test :

; DEFAULT: module {
; BADOPTION: lowering-options option: unknown style option 'bad-option'
; ALWAYSFF: module attributes {circt.loweringOptions = "alwaysFF"} {
