; RUN: firtool --split-input-file %s | FileCheck %s
FIRRTL version 4.0.0
; Two modules with *different* output directories should not dedup.
circuit Top: %[[
  {
    "class": "circt.OutputDirAnnotation",
    "dirname": "XXX",
    "target": "~Top|A"
  },
  {
    "class": "circt.OutputDirAnnotation",
    "dirname": "YYY",
    "target": "~Top|B"
  },
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Top|A>w"},
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Top|B>w"}
]]
  ; CHECK: FILE "XXX{{/|\\}}A.sv"
  ; CHECK: module A
  public module A:
    wire w : UInt<1>
    w is invalid

  ; CHECK: FILE "YYY{{/|\\}}B.sv"
  ; CHECK: module B
  public module B:
    wire w : UInt<1>
    w is invalid

  public module Top:
    inst a of A
    inst b of B

; // -----
FIRRTL version 4.0.0

; A module with an output directory anno should *not* dedup with a module that
; doesn't have one.
circuit Top: %[[
  {
    "class": "circt.OutputDirAnnotation",
    "dirname": "YYY",
    "target": "~Top|B"
  },
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Top|A>w"},
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Top|B>w"}
]]
  ; CHECK-NOT: FILE "YYY{{/|\\}}B.sv"
  ; CHECK: module A
  public module A:
    wire w : UInt<1>
    w is invalid

  ; CHECK: FILE "YYY{{/|\\}}B.sv"
  ; CHECK: module B
  public module B:
    wire w : UInt<1>
    w is invalid

  public module Top:
    inst a of A
    inst b of B

; // -----
FIRRTL version 4.0.0

; CA (child of A) and CB (child of B) should dedup, and be placed in the
; output directory lca(dir(A), dir(B)).
circuit Top: %[[
  {
    "class": "circt.OutputDirAnnotation",
    "dirname": "ZZZ/XXX",
    "target": "~Top|A"
  },
  {
    "class": "circt.OutputDirAnnotation",
    "dirname": "ZZZ/YYY",
    "target": "~Top|B"
  },
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Top|CA>w"},
  {"class": "firrtl.transforms.DontTouchAnnotation", "target": "~Top|CB>w"}

]]
  ; CHECK: FILE "ZZZ{{/|\\}}CA.sv"
  ; CHECK: module CA
  module CA:
    wire w : UInt<1>
    w is invalid

  ; CHECK-NOT: module CB
  module CB:
    wire w : UInt<1>
    w is invalid
  
  ; CHECK: FILE "ZZZ{{/|\\}}XXX{{/|\\}}A.sv"
  ; CHECK: module A
  public module A:
    ; CHECK: CA c
    inst c of CA

  ; CHECK: FILE "ZZZ{{/|\\}}YYY{{/|\\}}B.sv"
  ; CHECK: module B
  public module B:
    ; CHECK: CA c
    inst c of CB

  public module Top:
    inst a of A
    inst b of B
