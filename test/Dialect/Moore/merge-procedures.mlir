// RUN: circt-opt %s -moore-merge-procedures -split-input-file -verify-diagnostics | FileCheck %s
module {
  // CHECK-LABEL: module @merge_procedures
  moore.module @merge_procedures(in %wr_clk : !moore.l1, in %wr_data1 : !moore.l1, in %wr_data2 : !moore.l1, out mem : !moore.l2) {
    // CHECK-NEXT: moore.variable name "wr_clk"
    %wr_clk_0 = moore.variable name "wr_clk" : <l1>
    // CHECK-NEXT: moore.variable name "wr_data1"
    %wr_data1_1 = moore.variable name "wr_data1" : <l1>
    // CHECK-NEXT: moore.variable name "wr_data2"
    %wr_data2_2 = moore.variable name "wr_data2" : <l1>
    // CHECK-NEXT: moore.variable
    %mem = moore.variable : <l2>
    // CHECK-NEXT: moore.procedure always_ff
    moore.procedure always_ff {
      moore.wait_event {
        %3 = moore.read %wr_clk_0 : <l1>
        moore.detect_event posedge %3 : l1
      }
      %1 = moore.extract_ref %mem from 0 : <l2> -> <l1>
      %2 = moore.read %wr_data1_1 : <l1>
      moore.nonblocking_assign %1, %2 : l1
      moore.return
    }
    moore.procedure always_ff {
      moore.wait_event {
        %3 = moore.read %wr_clk_0 : <l1>
        moore.detect_event posedge %3 : l1
      }
      %1 = moore.extract_ref %mem from 1 : <l2> -> <l1>
      %2 = moore.read %wr_data2_2 : <l1>
      moore.nonblocking_assign %1, %2 : l1
      moore.return
    }
    moore.assign %wr_clk_0, %wr_clk : l1
    moore.assign %wr_data1_1, %wr_data1 : l1
    moore.assign %wr_data2_2, %wr_data2 : l1
    %0 = moore.read %mem : <l2>
    moore.output %0 : !moore.l2
  }
  // CHECK-NOT: moore.procedure
}

// -----


module {
  // Non-identical wait events should not be merged.
  // CHECK-LABEL: module @procedures_with_non_identical_wait_events
  // @expected-warning @+1 {{could not merge procedures}}
  moore.module @procedures_with_non_identical_wait_events() {
    %wr_clk_0 = moore.variable name "wr_clk_1" : <l1>
    %wr_clk_1 = moore.variable name "wr_clk_2" : <l1>
    // CHECK: moore.procedure always_ff
    // CHECK: moore.procedure always_ff
    moore.procedure always_ff {
      moore.wait_event {
        %3 = moore.read %wr_clk_0 : <l1>
        moore.detect_event posedge %3 : l1
      }
      moore.return
    }
    moore.procedure always_ff {
      moore.wait_event {
        %3 = moore.read %wr_clk_1 : <l1>
        moore.detect_event posedge %3 : l1
      }
      moore.return
    }
    moore.output
  }
  // CHECK-NOT: moore.procedure
}
