; This test checks the interaction between firtool's -o <file> option and
; the assignment/resolution of output directories. Namely, does firtool actually
; put verilog files in the right place.

; RUN: firtool -split-verilog %s -o %t/design

; RUN: FileCheck %s -input-file=%t/design/Foo.sv --check-prefix=FOO
; RUN: FileCheck %s -input-file=%t/design/subdir/Bar.sv --check-prefix=BAR
; RUN: FileCheck %s -input-file=%t/Baz.sv --check-prefix=BAZ

; RUN: FileCheck %s -input-file=%t/design/XFoo.sv --check-prefix=XFOO
; RUN: FileCheck %s -input-file=%t/design/subdir/XBar.sv --check-prefix=XBAR
; RUN: FileCheck %s -input-file=%t/XBaz.sv --check-prefix=XBAZ

; This test checks the interaction between firtool's -o <file> option and
; the assignment/resolution of output directories.

FIRRTL version 4.0.0

circuit Foo: %[[
  {"class": "circt.OutputDirAnnotation", "target": "~Foo|Bar", "dirname": "subdir"},
  {"class": "circt.OutputDirAnnotation", "target": "~Foo|Baz", "dirname": ".."},
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Foo|XFoo>w"},
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Foo|XBar>w"},
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Foo|XBaz>w"}
]]
  module XFoo:
    wire w : UInt<8>
    w is invalid
  module XBar:
    wire w : UInt<8>
    w is invalid
  module XBaz:
    wire w : UInt<8>
    w is invalid
  public module Foo:
    inst x of XFoo
  public module Bar:
    inst x of XBar
  public module Baz:
    inst x of XBaz

; FOO: module Foo();
; BAR: module Bar();
; BAZ: module Baz();
; XFOO: module XFoo();
; XBAR: module XBar();
; XBAZ: module XBaz();
