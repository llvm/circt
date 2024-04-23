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
  }
]]
  ; CHECK: FILE "XXX/A.sv"
  ; CHECK: module A
  module A:
    input i : UInt<8>
    output o : UInt<8>
    connect o, i

  ; CHECK: FILE "YYY/B.sv"
  ; CHECK: module B
  module B:
    input i : UInt<8>
    output o : UInt<8>
    connect o, i

  public module Top:
    input a_i : UInt<8>
    input b_i : UInt<8>
    output a_o : UInt<8>
    output b_o : UInt<8>
    inst a of A
    inst b of B
    connect a.i, a_i
    connect a_o, a.o
    connect b.i, b_i
    connect b_o, b.o

; // -----
FIRRTL version 4.0.0

; Two modules with the same output directory *should* dedup.
circuit Top: %[[
  {
    "class": "circt.OutputDirAnnotation",
    "dirname": "XXX",
    "target": "~Top|A"
  },
  {
    "class": "circt.OutputDirAnnotation",
    "dirname": "XXX",
    "target": "~Top|B"
  }
]]
  ; CHECK: FILE "XXX/A.sv"
  ; CHECK: module A
  module A:
    input i : UInt<8>
    output o : UInt<8>
    connect o, i

  ; CHECK-NOT: FILE "YYY/B.sv"
  ; CHECK-NOT: module B
  module B:
    input i : UInt<8>
    output o : UInt<8>
    connect o, i

  public module Top:
    input a_i : UInt<8>
    input b_i : UInt<8>
    output a_o : UInt<8>
    output b_o : UInt<8>
    inst a of A
    inst b of B
    connect a.i, a_i
    connect a_o, a.o
    connect b.i, b_i
    connect b_o, b.o

; // -----
FIRRTL version 4.0.0

; A module with an output directory anno should *not* dedup with a module that
; doesn't have one.
circuit Top: %[[
  {
    "class": "circt.OutputDirAnnotation",
    "dirname": "YYY",
    "target": "~Top|B"
  }
]]
  ; CHECK-NOT: FILE "YYY/B.sv"
  ; CHECK: module A
  module A:
    input i : UInt<8>
    output o : UInt<8>
    connect o, i

  ; CHECK: FILE "YYY/B.sv"
  ; CHECK: module B
  module B:
    input i : UInt<8>
    output o : UInt<8>
    connect o, i

  public module Top:
    input a_i : UInt<8>
    input b_i : UInt<8>
    output a_o : UInt<8>
    output b_o : UInt<8>
    inst a of A
    inst b of B
    connect a.i, a_i
    connect a_o, a.o
    connect b.i, b_i
    connect b_o, b.o

; // -----
FIRRTL version 4.0.0

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
  {
    "class": "circt.OutputDirPrecedenceAnnotation",
    "name": "XXX",
    "parent": "ZZZ"
  },
  {
    "class": "circt.OutputDirPrecedenceAnnotation",
    "name": "YYY",
    "parent": "ZZZ"
  },
  {
    "class": "circt.OutputDirPrecedenceAnnotation",
    "name": "ZZZ",
    "parent": ""
  }
]]
  ; CHECK: FILE "ZZZ/CA.sv"
  ; CHECK: module CA
  module CA:
    input i : UInt<8>
    output o : UInt<8>
    connect o, i

  ; CHECK-NOT: module CB
  module CB:
    input i : UInt<8>
    output o : UInt<8>
    connect o, i
  
  ; CHECK: FILE "XXX/A.sv"
  ; CHECK: module A
  module A:
    input i : UInt<8>
    output o : UInt<8>
    ; CHECK: CA c
    inst c of CA
    connect c.i, i
    connect o, c.o

  ; CHECK: FILE "YYY/B.sv"
  ; CHECK: module B
  module B:
    input i : UInt<8>
    output o : UInt<8>
    ; CHECK: CA c
    inst c of CB
    connect c.i, i
    connect o, c.o

  public module Top:
    input a_i : UInt<8>
    input b_i : UInt<8>
    output a_o : UInt<8>
    output b_o : UInt<8>
    inst a of A
    inst b of B
    connect a.i, a_i
    connect a_o, a.o
    connect b.i, b_i
    connect b_o, b.o
