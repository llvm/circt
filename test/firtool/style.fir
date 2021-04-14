; RUN: firtool %s | FileCheck %s --check-prefix=DEFAULT
; RUN: not firtool --lowering-options=bad-option %s 2>&1 | FileCheck %s --check-prefix=BADOPTION
; RUN: firtool --lowering-options=noAlwaysFF %s | FileCheck %s --check-prefix=NOALWAYSFF

circuit test :
  module test :

; DEFAULT: module {
; BADOPTION: lowering-options option: unknown style option 'bad-option'
; NOALWAYSFF: module attributes {circt.loweringOptions = "noAlwaysFF"} {