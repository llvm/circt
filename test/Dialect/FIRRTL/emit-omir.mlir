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
// CHECK-SAME:  #hw.output_file<"omir.json", excludeFromFileList>

//===----------------------------------------------------------------------===//
// Empty node
//===----------------------------------------------------------------------===//

#loc = loc(unknown)
firrtl.circuit "EmptyNode" attributes {annotations = [{class = "freechips.rocketchip.objectmodel.OMIRAnnotation", nodes = [{fields = {}, id = "OMID:0", info = #loc}]}]}  {
  firrtl.module @EmptyNode() {
  }
}

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
// CHECK-SAME:  \22name\22: \22OMBoolean\22,\0A \22value\22: true
// CHECK-SAME:  \22name\22: \22OMInt1\22,\0A \22value\22: 9001
// CHECK-SAME:  \22name\22: \22OMInt2\22,\0A \22value\22: -42
// CHECK-SAME:  \22name\22: \22OMDouble\22,\0A \22value\22: 3.14
// CHECK-SAME:  \22name\22: \22OMID\22,\0A \22value\22: \22OMID:1337\22
// CHECK-SAME:  \22name\22: \22OMReference\22,\0A \22value\22: \22OMReference:0\22
// CHECK-SAME:  \22name\22: \22OMBigInt\22,\0A \22value\22: \22OMBigInt:42\22
// CHECK-SAME:  \22name\22: \22OMLong\22,\0A \22value\22: \22OMLong:ff\22
// CHECK-SAME:  \22name\22: \22OMString\22,\0A \22value\22: \22OMString:hello\22
// CHECK-SAME:  \22name\22: \22OMBigDecimal\22,\0A \22value\22: \22OMBigDecimal:10.5\22
// CHECK-SAME:  \22name\22: \22OMDeleted\22,\0A \22value\22: \22OMDeleted\22
// CHECK-SAME:  \22name\22: \22OMConstant\22,\0A \22value\22: \22OMConstant:UInt<2>(\\\22h1\\\22)\22
// CHECK-SAME:  \22name\22: \22OMArray\22,\0A \22value\22: [\0A true,\0A 9001,\0A \22OMString:bar\22\0A ]
// CHECK-SAME:  \22name\22: \22OMMap\22,\0A \22value\22: {\0A \22bar\22: 9001,\0A \22foo\22: true\0A }
// CHECK-SAME:  #hw.output_file<"omir.json", excludeFromFileList>

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
// CHECK-NEXT:      %c = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<42>
// CHECK-NEXT:    }
// CHECK-NEXT:    firrtl.module @LocalTrackers() {
// CHECK-NEXT:      firrtl.instance a {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} @A()
// CHECK-NEXT:      %b = firrtl.wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<42>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  sv.verbatim
// CHECK-SAME:  \22name\22: \22OMReferenceTarget1\22,\0A \22value\22: \22OMReferenceTarget:~LocalTrackers|{{[{][{]0[}][}]}}\22
// CHECK-SAME:  \22name\22: \22OMReferenceTarget2\22,\0A \22value\22: \22OMReferenceTarget:~LocalTrackers|{{[{][{]0[}][}]}}>c\22
// CHECK-SAME:  \22name\22: \22OMReferenceTarget3\22,\0A \22value\22: \22OMReferenceTarget:~LocalTrackers|{{[{][{]1[}][}]}}>b\22
// CHECK-SAME:  \22name\22: \22OMReferenceTarget4\22,\0A \22value\22: \22OMReferenceTarget:~LocalTrackers|{{[{][{]1[}][}]}}>a\22
// CHECK-SAME:  #hw.output_file<"omir.json", excludeFromFileList>
// CHECK-SAME:  symbols = [@A, @LocalTrackers]

//===----------------------------------------------------------------------===//
// Trackers as Non-Local Annotations
//===----------------------------------------------------------------------===//

firrtl.circuit "NonLocalTrackers" attributes {annotations = [{
  class = "freechips.rocketchip.objectmodel.OMIRAnnotation",
  nodes = [{info = #loc, id = "OMID:0", fields = {
    OMReferenceTarget1 = {info = #loc, index = 1, id = "OMID:1", value = {omir.tracker, id = 0, type = "OMReferenceTarget"}}
  }}]
}]} {
  firrtl.nla @nla_0 [@NonLocalTrackers, @B, @A] ["b", "a", "A"]
  firrtl.module @A() attributes {annotations = [{circt.nonlocal = @nla_0, class = "freechips.rocketchip.objectmodel.OMIRTracker", id = 0}]} {}
  firrtl.module @B() {
    firrtl.instance a {annotations = [{circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @A()
  }
  firrtl.module @NonLocalTrackers() {
    firrtl.instance b {annotations = [{circt.nonlocal = @nla_0, class = "circt.nonlocal"}]} @B()
  }
}
// CHECK-LABEL: firrtl.circuit "NonLocalTrackers" {
// CHECK:       sv.verbatim
// CHECK-SAME:  \22name\22: \22OMReferenceTarget1\22,\0A \22value\22: \22OMReferenceTarget:~NonLocalTrackers|{{[{][{]0[}][}]}}/b:{{[{][{]1[}][}]}}/a:{{[{][{]2[}][}]}}\22
// CHECK-SAME:  #hw.output_file<"omir.json", excludeFromFileList>
// CHECK-SAME:  symbols = [@NonLocalTrackers, @B, @A]
