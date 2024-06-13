; Test interaction between include directory and annotation files.

; Setup:
; RUN: rm -rf %t && mkdir -p %t
; RUN: echo '[{ "class": "circt.test", "target": "~Test" }]' > %t/test.anno.json

; Confirm annotation file not found without include directory provided:
; RUN: not firtool %s --parse-only --annotation-file test.anno.json
; Confirm annotation file is found.
; RUN: firtool %s --parse-only --annotation-file test.anno.json -I %t | grep circt.test

FIRRTL version 4.0.0
circuit Test:
  public module Test:
    output o : UInt<1>
    o <= UInt<1>(0)
    
