// RUN: arcilator %s | FileCheck %s

module {
  // CHECK-LABEL: define void @test_success_eval
  hw.module @test_success(in %clock: !seq.clock, in %cond: i1) {
    // CHECK: br i1 %{{.*}}, label %[[SET_FLAG:.*]], label %[[CONTINUE:.*]]
    // CHECK: [[CONTINUE]]:
    // CHECK: [[SET_FLAG]]:
    // CHECK-NEXT: %[[GEP:.*]] = getelementptr i8, ptr %0, i32 8
    // CHECK-NEXT: store i8 1, ptr %[[GEP]]{{.*}}
    // CHECK-NEXT: br label %[[CONTINUE]]
    sim.clocked_terminate %clock, %cond, success, verbose
  }

  // CHECK-LABEL: define void @test_failure_eval
  hw.module @test_failure(in %clock: !seq.clock, in %cond: i1) {
    // CHECK: br i1 %{{.*}}, label %[[SET_FLAG_FAIL:.*]], label %[[CONTINUE_FAIL:.*]]
    // CHECK: [[CONTINUE_FAIL]]:
    // CHECK: [[SET_FLAG_FAIL]]:
    // CHECK-NEXT: %[[GEP_FAIL:.*]] = getelementptr i8, ptr %0, i32 8
    // CHECK-NEXT: store i8 2, ptr %[[GEP_FAIL]]{{.*}}
    // CHECK-NEXT: br label %[[CONTINUE_FAIL]]
    sim.clocked_terminate %clock, %cond, failure, quiet
  }
}