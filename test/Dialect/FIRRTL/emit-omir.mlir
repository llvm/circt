// RUN: circt-opt --pass-pipeline='firrtl.circuit(firrtl-emit-omir{file=omir.json})' %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Absence of any OMIR
//===----------------------------------------------------------------------===//

firrtl.circuit "NoOMIR" {
  firrtl.module @NoOMIR() {
  }
}
// CHECK-LABEL: firrtl.circuit "NoOMIR" {
// CHECK-NEXT:    firrtl.module @NoOMIR() {
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  sv.verbatim "[]"
// CHECK-SAME:  #hw.output_file<"omir.json", excludeFromFileList>

//===----------------------------------------------------------------------===//
// Empty OMIR data
//===----------------------------------------------------------------------===//

firrtl.circuit "NoNodes" attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRAnnotation", nodes = []}]}  {
  firrtl.module @NoNodes() {
  }
}
// CHECK-LABEL: firrtl.circuit "NoNodes" {
// CHECK-NEXT:    firrtl.module @NoNodes() {
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  sv.verbatim "[]"

//===----------------------------------------------------------------------===//
// Empty node
//===----------------------------------------------------------------------===//

#loc = loc(unknown)
firrtl.circuit "EmptyNode" attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRAnnotation", nodes = [{fields = {}, id = "OMID:0", info = #loc}]}]}  {
  firrtl.module @EmptyNode() {
  }
}
// CHECK-LABEL: firrtl.circuit "EmptyNode" {
// CHECK-NEXT:    firrtl.module @EmptyNode() {
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  sv.verbatim
// CHECK-SAME:  \22info\22: \22UnlocatableSourceInfo\22
// CHECK-SAME:  \22id\22: \22OMID:0\22
// CHECK-SAME:  \22fields\22: []

//===----------------------------------------------------------------------===//
// Source locator serialization
//===----------------------------------------------------------------------===//

#loc0 = loc("B":2:3)
#loc1 = loc(fused["C":4:5, "D":6:7])
#loc2 = loc("A":0:1)
firrtl.circuit "SourceLocators" attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRAnnotation", nodes = [{fields = {x = {index = 1 : i64, info = #loc0, value = "OMReference:0"}, y = {index = 0 : i64, info = #loc1, value = "OMReference:0"}}, id = "OMID:0", info = #loc2}]}]}  {
  firrtl.module @SourceLocators() {
  }
}
// CHECK-LABEL: firrtl.circuit "SourceLocators" {
// CHECK-NEXT:    firrtl.module @SourceLocators() {
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  sv.verbatim
// CHECK-SAME:  \22info\22: \22@[A 0:1]\22
// CHECK-SAME:  \22id\22: \22OMID:0\22
// CHECK-SAME:  \22fields\22: [
// CHECK-SAME:    {
// CHECK-SAME:      \22info\22: \22@[C 4:5 D 6:7]\22
// CHECK-SAME:      \22name\22: \22y\22
// CHECK-SAME:      \22value\22: \22OMReference:0\22
// CHECK-SAME:    }
// CHECK-SAME:    {
// CHECK-SAME:      \22info\22: \22@[B 2:3]\22
// CHECK-SAME:      \22name\22: \22x\22
// CHECK-SAME:      \22value\22: \22OMReference:0\22
// CHECK-SAME:    }
// CHECK-SAME:  ]

//===----------------------------------------------------------------------===//
// Check that all the OMIR types support serialization
//===----------------------------------------------------------------------===//

firrtl.circuit "AllTypesSupported" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    OMBoolean = {info = #loc, index = 1, value = true},
    OMInt1 = {info = #loc, index = 2, value = 9001 : i32},
    OMInt2 = {info = #loc, index = 3, value = -42 : i32},
    OMDouble = {info = #loc, index = 4, value = 3.14 : f32},
    OMID = {info = #loc, index = 5, value = "OMID:1337"},
    OMReference = {info = #loc, index = 6, value = "OMReference:0"},
    OMBigInt = {info = #loc, index = 7, value = "OMBigInt:42"},
    OMLong = {info = #loc, index = 8, value = "OMLong:ff"},
    OMString = {info = #loc, index = 9, value = "OMString:hello"},
    OMBigDecimal = {info = #loc, index = 10, value = "OMBigDecimal:10.5"},
    OMDeleted = {info = #loc, index = 11, value = "OMDeleted"},
    OMConstant = {info = #loc, index = 12, value = "OMConstant:UInt<2>(\"h1\")"},
    OMArray = {info = #loc, index = 13, value = [true, 9001, "OMString:bar"]},
    OMMap = {info = #loc, index = 14, value = {foo = true, bar = 9001}}
  }}]
}]} {
  firrtl.module @AllTypesSupported() {
  }
}
// CHECK-LABEL: firrtl.circuit "AllTypesSupported" {
// CHECK-NEXT:    firrtl.module @AllTypesSupported() {
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  sv.verbatim
// CHECK-SAME:  \22name\22: \22OMBoolean\22
// CHECK-SAME:  \22value\22: true
// CHECK-SAME:  \22name\22: \22OMInt1\22
// CHECK-SAME:  \22value\22: 9001
// CHECK-SAME:  \22name\22: \22OMInt2\22
// CHECK-SAME:  \22value\22: -42
// CHECK-SAME:  \22name\22: \22OMDouble\22
// CHECK-SAME:  \22value\22: 3.14
// CHECK-SAME:  \22name\22: \22OMID\22
// CHECK-SAME:  \22value\22: \22OMID:1337\22
// CHECK-SAME:  \22name\22: \22OMReference\22
// CHECK-SAME:  \22value\22: \22OMReference:0\22
// CHECK-SAME:  \22name\22: \22OMBigInt\22
// CHECK-SAME:  \22value\22: \22OMBigInt:42\22
// CHECK-SAME:  \22name\22: \22OMLong\22
// CHECK-SAME:  \22value\22: \22OMLong:ff\22
// CHECK-SAME:  \22name\22: \22OMString\22
// CHECK-SAME:  \22value\22: \22OMString:hello\22
// CHECK-SAME:  \22name\22: \22OMBigDecimal\22
// CHECK-SAME:  \22value\22: \22OMBigDecimal:10.5\22
// CHECK-SAME:  \22name\22: \22OMDeleted\22
// CHECK-SAME:  \22value\22: \22OMDeleted\22
// CHECK-SAME:  \22name\22: \22OMConstant\22
// CHECK-SAME:  \22value\22: \22OMConstant:UInt<2>(\\\22h1\\\22)\22
// CHECK-SAME:  \22name\22: \22OMArray\22
// CHECK-SAME:  \22value\22: [
// CHECK-SAME:    true
// CHECK-SAME:    9001
// CHECK-SAME:    \22OMString:bar\22
// CHECK-SAME:  ]
// CHECK-SAME:  \22name\22: \22OMMap\22
// CHECK-SAME:  \22value\22: {
// CHECK-SAME:    \22bar\22: 9001
// CHECK-SAME:    \22foo\22: true
// CHECK-SAME:  }

//===----------------------------------------------------------------------===//
// Trackers as Local Annotations
//===----------------------------------------------------------------------===//

firrtl.circuit "LocalTrackers" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    OMReferenceTarget1 = {info = #loc, index = 1, value = {omir.tracker, id = 0, type = "OMReferenceTarget"}},
    OMReferenceTarget2 = {info = #loc, index = 2, value = {omir.tracker, id = 1, type = "OMReferenceTarget"}},
    OMReferenceTarget3 = {info = #loc, index = 3, value = {omir.tracker, id = 2, type = "OMReferenceTarget"}},
    OMReferenceTarget4 = {info = #loc, index = 4, value = {omir.tracker, id = 3, type = "OMReferenceTarget"}}
  }}]
}]} {
  firrtl.module @A() attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} {
    %c = firrtl.wire {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1}]} : !firrtl.uint<42>
  }
  firrtl.module @LocalTrackers() {
    firrtl.instance a {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 3}]} @A()
    %b = firrtl.wire {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 2}]} : !firrtl.uint<42>
  }
}
// CHECK-LABEL: firrtl.circuit "LocalTrackers" {
// CHECK-NEXT:    firrtl.module @A() {
// CHECK-NEXT:      %c = firrtl.wire sym [[SYMC:@[a-zA-Z0-9_]+]] : !firrtl.uint<42>
// CHECK-NEXT:    }
// CHECK-NEXT:    firrtl.module @LocalTrackers() {
// CHECK-NEXT:      firrtl.instance a sym [[SYMA:@[a-zA-Z0-9_]+]] @A()
// CHECK-NEXT:      %b = firrtl.wire sym [[SYMB:@[a-zA-Z0-9_]+]] : !firrtl.uint<42>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  sv.verbatim
// CHECK-SAME:  \22name\22: \22OMReferenceTarget1\22
// CHECK-SAME:  \22value\22: \22OMReferenceTarget:~LocalTrackers|{{[{][{]0[}][}]}}\22
// CHECK-SAME:  \22name\22: \22OMReferenceTarget2\22
// CHECK-SAME:  \22value\22: \22OMReferenceTarget:~LocalTrackers|{{[{][{]0[}][}]}}>{{[{][{]1[}][}]}}\22
// CHECK-SAME:  \22name\22: \22OMReferenceTarget3\22
// CHECK-SAME:  \22value\22: \22OMReferenceTarget:~LocalTrackers|{{[{][{]2[}][}]}}>{{[{][{]3[}][}]}}\22
// CHECK-SAME:  \22name\22: \22OMReferenceTarget4\22
// CHECK-SAME:  \22value\22: \22OMReferenceTarget:~LocalTrackers|{{[{][{]2[}][}]}}>{{[{][{]4[}][}]}}\22
// CHECK-SAME:  symbols = [
// CHECK-SAME:    @A,
// CHECK-SAME:    #hw.innerNameRef<@A::[[SYMC:@[a-zA-Z0-9_]+]]>,
// CHECK-SAME:    @LocalTrackers,
// CHECK-SAME:    #hw.innerNameRef<@LocalTrackers::[[SYMB:@[a-zA-Z0-9_]+]]>,
// CHECK-SAME:    #hw.innerNameRef<@LocalTrackers::[[SYMA:@[a-zA-Z0-9_]+]]>
// CHECK-SAME:  ]

//===----------------------------------------------------------------------===//
// Trackers as Non-Local Annotations
//===----------------------------------------------------------------------===//

firrtl.circuit "NonLocalTrackers" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    OMReferenceTarget1 = {info = #loc, index = 1, id = "OMID:1", value = {omir.tracker, id = 0, type = "OMReferenceTarget"}}
  }}]
}]} {
  firrtl.nla @nla_0 [#hw.innerNameRef<@NonLocalTrackers::@b>, #hw.innerNameRef<@B::@a>, @A]
  firrtl.module @A() attributes {annotations = [{circt.nonlocal = @nla_0, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} {}
  firrtl.module @B() {
    firrtl.instance a sym @a {annotations = [{circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @A()
  }
  firrtl.module @NonLocalTrackers() {
    firrtl.instance b sym @b {annotations = [{circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @B()
  }
}
// CHECK-LABEL: firrtl.circuit "NonLocalTrackers"
// CHECK:       firrtl.instance a sym [[SYMA:@[a-zA-Z0-9_]+]]
// CHECK:       firrtl.instance b sym [[SYMB:@[a-zA-Z0-9_]+]]
// CHECK:       sv.verbatim
// CHECK-SAME:  \22name\22: \22OMReferenceTarget1\22
// CHECK-SAME:  \22value\22: \22OMReferenceTarget:~NonLocalTrackers|{{[{][{]0[}][}]}}/{{[{][{]1[}][}]}}:{{[{][{]2[}][}]}}/{{[{][{]3[}][}]}}:{{[{][{]4[}][}]}}\22
// CHECK-SAME:  symbols = [
// CHECK-SAME:    @NonLocalTrackers,
// CHECK-SAME:    #hw.innerNameRef<@NonLocalTrackers::[[SYMB:@[a-zA-Z0-9_]+]]>,
// CHECK-SAME:    @B,
// CHECK-SAME:    #hw.innerNameRef<@B::[[SYMA:@[a-zA-Z0-9_]+]]>,
// CHECK-SAME:    @A
// CHECK-SAME:  ]

//===----------------------------------------------------------------------===//
// Targets that are allowed to lose their tracker
//===----------------------------------------------------------------------===//

firrtl.circuit "DeletedTargets" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    a = {info = #loc, index = 1, value = {omir.tracker, id = 0, type = "OMReferenceTarget"}},
    b = {info = #loc, index = 2, value = {omir.tracker, id = 1, type = "OMMemberReferenceTarget"}},
    c = {info = #loc, index = 3, value = {omir.tracker, id = 2, type = "OMMemberInstanceTarget"}}
  }}]
}]} {
  firrtl.module @DeletedTargets() {}
}
// CHECK-LABEL: firrtl.circuit "DeletedTargets"
// CHECK:       sv.verbatim
// CHECK-SAME:  \22name\22: \22a\22
// CHECK-SAME:  \22value\22: \22OMDeleted:\22
// CHECK-SAME:  \22name\22: \22b\22
// CHECK-SAME:  \22value\22: \22OMDeleted:\22
// CHECK-SAME:  \22name\22: \22c\22
// CHECK-SAME:  \22value\22: \22OMDeleted:\22

//===----------------------------------------------------------------------===//
// Make SRAM Paths Absolute (`SetOMIRSRAMPaths`)
//===----------------------------------------------------------------------===//

firrtl.circuit "SRAMPaths" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [
    {
      info = #loc,
      id = "OMID:0",
      fields = {
        omType = {info = #loc, index = 0, value = ["OMString:OMLazyModule", "OMString:OMSRAM"]},
        // We purposefully pick the wrong `OMMemberTarget` here, to check that
        // it actually gets emitted as a `OMMemberInstanceTarget`. These can
        // change back and forth as the FIRRTL passes work on the IR, and the
        // OMIR output should reflect the final target.
        finalPath = {info = #loc, index = 1, value = {omir.tracker, id = 0, type = "OMMemberReferenceTarget"}}
      }
    },
    {
      info = #loc,
      id = "OMID:1",
      fields = {
        omType = {info = #loc, index = 0, value = ["OMString:OMLazyModule", "OMString:OMSRAM"]},
        finalPath = {info = #loc, index = 1, value = {omir.tracker, id = 1, type = "OMMemberReferenceTarget"}}
      }
    }
  ]
}]} {
  firrtl.extmodule @MySRAM()
  firrtl.module @Submodule() {
    firrtl.instance mem1 {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} @MySRAM()
    %mem2_port = firrtl.mem Undefined {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1}], depth = 8, name = "mem2", portNames = ["port"], readLatency = 0 : i32, writeLatency = 1 : i32 } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<42>>
  }
  firrtl.module @SRAMPaths() {
    firrtl.instance sub @Submodule()
  }
}
// CHECK-LABEL: firrtl.circuit "SRAMPaths" {
// CHECK-NEXT:    firrtl.extmodule @MySRAM()
// CHECK-NEXT:    firrtl.module @Submodule() {
// CHECK-NEXT:      firrtl.instance mem1 sym [[SYMMEM1:@[a-zA-Z0-9_]+]]
// CHECK-SAME:        @MySRAM()
// CHECK-NEXT:      firrtl.mem sym [[SYMMEM2:@[a-zA-Z0-9_]+]]
// CHECK-SAME:        name = "mem2"
// CHECK-SAME:        : !firrtl.bundle
// CHECK-NEXT:    }
// CHECK-NEXT:    firrtl.module @SRAMPaths() {
// CHECK-NEXT:      firrtl.instance sub sym [[SYMSUB:@[a-zA-Z0-9_]+]]
// CHECK-NOT:         circt.nonlocal
// CHECK-SAME:        @Submodule()
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  sv.verbatim

// CHECK-SAME:  \22id\22: \22OMID:0\22
// CHECK-SAME:    \22name\22: \22omType\22
// CHECK-SAME:    \22value\22: [
// CHECK-SAME:      \22OMString:OMLazyModule\22
// CHECK-SAME:      \22OMString:OMSRAM\22
// CHECK-SAME:    ]
// CHECK-SAME:    \22name\22: \22finalPath\22
// CHECK-SAME:    \22value\22: \22OMMemberInstanceTarget:~SRAMPaths|{{[{][{]0[}][}]}}/{{[{][{]1[}][}]}}:{{[{][{]2[}][}]}}/{{[{][{]3[}][}]}}:{{[{][{]4[}][}]}}\22

// CHECK-SAME:  \22id\22: \22OMID:1\22
// CHECK-SAME:    \22name\22: \22omType\22
// CHECK-SAME:    \22value\22: [
// CHECK-SAME:      \22OMString:OMLazyModule\22
// CHECK-SAME:      \22OMString:OMSRAM\22
// CHECK-SAME:    ]
// CHECK-SAME:    \22name\22: \22finalPath\22
// CHECK-SAME:    \22value\22: \22OMMemberInstanceTarget:~SRAMPaths|{{[{][{]0[}][}]}}/{{[{][{]1[}][}]}}:{{[{][{]2[}][}]}}/{{[{][{]5[}][}]}}:{{[^\\]+}}\22

// CHECK-SAME:  symbols = [
// CHECK-SAME:    @SRAMPaths,
// CHECK-SAME:    #hw.innerNameRef<@SRAMPaths::[[SYMSUB:@[a-zA-Z0-9_]+]]>,
// CHECK-SAME:    @Submodule,
// CHECK-SAME:    #hw.innerNameRef<@Submodule::[[SYMMEM1:@[a-zA-Z0-9_]+]]>,
// CHECK-SAME:    @MySRAM,
// CHECK-SAME:    #hw.innerNameRef<@Submodule::[[SYMMEM2:@[a-zA-Z0-9_]+]]>
// CHECK-SAME:  ]

//===----------------------------------------------------------------------===//
// Add module port information to the OMIR (`SetOMIRPorts`)
//===----------------------------------------------------------------------===//

firrtl.circuit "AddPorts" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [
    {
      info = #loc,
      id = "OMID:0",
      fields = {
        containingModule = {
          info = #loc,
          index = 0,
          value = {
            omir.tracker,
            id = 0,
            path = "~AddPorts|AddPorts",
            type = "OMInstanceTarget"
          }
        }
      }
    },
    {
      info = #loc,
      id = "OMID:1",
      fields = {
        containingModule = {
          info = #loc,
          index = 0,
          value = {
            omir.tracker,
            id = 1,
            path = "~AddPorts|AddPorts>w",
            type = "OMReferenceTarget"
          }
        }
      }
    }
  ]
}]} {
  firrtl.module @AddPorts(in %x: !firrtl.uint<29>, out %y: !firrtl.uint<31>) attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} {
    %w = firrtl.wire {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 1}]} : !firrtl.uint<29>
    firrtl.connect %y, %x : !firrtl.uint<31>, !firrtl.uint<29>
  }
}
// CHECK-LABEL: firrtl.circuit "AddPorts"
// CHECK:       firrtl.module @AddPorts
// CHECK-SAME:    in %x: !firrtl.uint<29> sym [[SYMX:@[a-zA-Z0-9_]+]]
// CHECK-SAME:    out %y: !firrtl.uint<31> sym [[SYMY:@[a-zA-Z0-9_]+]]
// CHECK:       %w = firrtl.wire sym [[SYMW:@[a-zA-Z0-9_]+]]
// CHECK:       sv.verbatim

// CHECK-SAME:  \22id\22: \22OMID:0\22
// CHECK-SAME:    \22name\22: \22containingModule\22
// CHECK-SAME:    \22value\22: \22OMInstanceTarget:~AddPorts|{{[{][{]0[}][}]}}\22
// CHECK-SAME:    \22name\22: \22ports\22
// CHECK-SAME:    \22value\22: [
// CHECK-SAME:      {
// CHECK-SAME:        \22ref\22: \22OMDontTouchedReferenceTarget:~AddPorts|{{[{][{]0[}][}]}}>{{[{][{]1[}][}]}}\22
// CHECK-SAME:        \22direction\22: \22OMString:Input\22
// CHECK-SAME:        \22width\22: \22OMBigInt:1d\22
// CHECK-SAME:      }
// CHECK-SAME:      {
// CHECK-SAME:        \22ref\22: \22OMDontTouchedReferenceTarget:~AddPorts|{{[{][{]0[}][}]}}>{{[{][{]2[}][}]}}\22
// CHECK-SAME:        \22direction\22: \22OMString:Output\22
// CHECK-SAME:        \22width\22: \22OMBigInt:1f\22
// CHECK-SAME:      }
// CHECK-SAME:    ]

// CHECK-SAME:  \22id\22: \22OMID:1\22
// CHECK-SAME:    \22name\22: \22containingModule\22
// CHECK-SAME:    \22value\22: \22OMReferenceTarget:~AddPorts|{{[{][{]0[}][}]}}>{{[{][{]3[}][}]}}\22
// CHECK-SAME:    \22name\22: \22ports\22
// CHECK-SAME:    \22value\22: [
// CHECK-SAME:      {
// CHECK-SAME:        \22ref\22: \22OMDontTouchedReferenceTarget:~AddPorts|{{[{][{]0[}][}]}}>{{[{][{]1[}][}]}}\22
// CHECK-SAME:        \22direction\22: \22OMString:Input\22
// CHECK-SAME:        \22width\22: \22OMBigInt:1d\22
// CHECK-SAME:      }
// CHECK-SAME:      {
// CHECK-SAME:        \22ref\22: \22OMDontTouchedReferenceTarget:~AddPorts|{{[{][{]0[}][}]}}>{{[{][{]2[}][}]}}\22
// CHECK-SAME:        \22direction\22: \22OMString:Output\22
// CHECK-SAME:        \22width\22: \22OMBigInt:1f\22
// CHECK-SAME:      }
// CHECK-SAME:    ]

// CHECK-SAME:  symbols = [
// CHECK-SAME:    @AddPorts,
// CHECK-SAME:    #hw.innerNameRef<@AddPorts::[[SYMX]]>,
// CHECK-SAME:    #hw.innerNameRef<@AddPorts::[[SYMY]]>,
// CHECK-SAME:    #hw.innerNameRef<@AddPorts::[[SYMW]]>
// CHECK-SAME:  ]
