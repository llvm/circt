// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

module {
  // Keep the func dialect loaded for class allocation helper declarations.
  func.func private @load_func_dialect()

  moore.class.classdecl @ClassEventWait {
    moore.class.propertydecl @changed : !moore.i1
  }

  // All event operands are read out of class-backed storage that is created
  // inside the wait body, so observer analysis cannot see anything to observe
  // outside the region. Without observing the sampled pre-wait values, the
  // resulting `llhd.wait` would have no observed values and hang forever.
  // CHECK-LABEL: hw.module @ClassPropertyWaitEvent
  moore.module @ClassPropertyWaitEvent() {
    moore.procedure initial {
      // CHECK: llhd.process {
      // CHECK: cf.br ^[[WAIT:.+]]
      // CHECK: ^[[WAIT]]:
      // CHECK: func.call @malloc
      // CHECK: [[REF:%.+]] = builtin.unrealized_conversion_cast {{%.+}} : !llvm.ptr to !llhd.ref<i1>
      // CHECK: [[EVENT:%.+]] = llhd.prb [[REF]] : i1
      // CHECK: llhd.wait ([[EVENT]] : i1), ^[[CHECKBB:.+]]
      // CHECK: ^[[CHECKBB]]:
      // CHECK-NOT: comb.icmp
      // CHECK: cf.br
      moore.wait_event {
        %obj = moore.class.new : <@ClassEventWait>
        %ref = moore.class.property_ref %obj[@changed] : <@ClassEventWait> -> !moore.ref<i1>
        %v = moore.read %ref : <i1>
        moore.detect_event any %v : i1
      }
      moore.return
    }
    moore.output
  }

  // Same situation for edge-sensitive waits on storage behind a pointer that
  // observer analysis cannot see: the sampled pre-wait value must show up as
  // the observed value of the `llhd.wait`, and the edge detection compares it
  // against the value probed after the wait.
  // CHECK-LABEL: hw.module @DynamicRefPosedgeWait
  moore.module @DynamicRefPosedgeWait() {
    moore.procedure initial {
      // CHECK: llhd.process {
      // CHECK: cf.br ^[[WAIT:.+]]
      // CHECK: ^[[WAIT]]:
      // CHECK: [[BEFORE:%.+]] = llhd.prb {{%.+}} : i1
      // CHECK: llhd.wait ([[BEFORE]] : i1), ^[[CHECKBB:.+]]
      // CHECK: ^[[CHECKBB]]:
      // CHECK: [[AFTER:%.+]] = llhd.prb {{%.+}} : i1
      // CHECK: [[TRUE:%.+]] = hw.constant true
      // CHECK: [[TMP1:%.+]] = comb.xor bin [[BEFORE]], [[TRUE]]
      // CHECK: [[TMP2:%.+]] = comb.and bin [[TMP1]], [[AFTER]]
      // CHECK: cf.cond_br [[TMP2]]
      moore.wait_event {
        %arg0 = llvm.mlir.zero : !llvm.ptr
        %ref = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !moore.ref<i1>
        %v = moore.read %ref : <i1>
        moore.detect_event posedge %v : i1
      }
      moore.return
    }
    moore.output
  }
}
