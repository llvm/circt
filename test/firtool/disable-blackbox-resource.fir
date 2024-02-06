; RUN: firtool %s | FileCheck --check-prefix=RESOURCE_FILE %s
; RUN: firtool --disable-blackbox-resource-file %s | FileCheck --check-prefix=NO_RESOURCE_FILE %s
FIRRTL version 3.3.0
circuit BlackBar :%[[
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~BlackBar|BlackFoo",
    "group":"BlackFoo"
  },
  {
    "class":"firrtl.transforms.DedupGroupAnnotation",
    "target":"~BlackBar|BlackBar",
    "group":"BlackBar"
  },
  {
    "class":"firrtl.transforms.BlackBoxInlineAnno",
    "target":"BlackBar.BlackFoo",
    "name":"BlackFoo.v",
    "text":"module BlackFoo (\n    input a,\n    output b\n  );\n\n  assign b = ~a;\n\nendmodule\n"
  }
]]
  extmodule BlackFoo :
    input a : UInt<1>
    output b : UInt<1>
    defname = BlackFoo

  module BlackBar :
    input clock : Clock
    input reset : UInt<1>
    output io : { flip a : UInt<1>, b : UInt<1>}

    inst foo of BlackFoo
    connect foo.a, io.a
    connect io.b, foo.b

; RESOURCE_FILE: BlackFoo.v
; RESOURCE_FILE: firrtl_black_box_resource_files.f
; RESOURCE_FILE-EMPTY:
; RESOURCE_FILE-NEXT: BlackFoo.v

; NO_RESOURCE_FILE: BlackFoo.v
; NO_RESOURCE_FILE-NOT: firrtl_black_box_resource_files.f