// REQUIRES: host=aarch64{{.*}} || host=x86_64{{.*}} || host=arm64{{.*}}
// RUN: arcilator %s --until-before=state-alloc --add-data-layout-information=false                                    | FileCheck %s --check-prefix=DLTIOFF
// RUN: arcilator %s --until-before=state-alloc --add-data-layout-information=true                                     | FileCheck %s --check-prefix=DLTION
// RUN: arcilator %s --until-before=state-alloc --add-data-layout-information=true --target-triple=arm64-apple-macosx  | FileCheck %s --check-prefix=DLTION
// RUN: arcilator %s --until-before=state-alloc --add-data-layout-information=true --target-triple=x86_64              | FileCheck %s --check-prefix=DLTION
// RUN: not arcilator %s --add-data-layout-information=true --target-triple=i386
// RUN: not arcilator %s --add-data-layout-information=true --target-triple=notARealArch

// Check that DLTI information is available before state allocation, unless disabled

// DLTION-LABEL: module
// DLTION-SAME:  dlti.dl_spec = #dlti.dl_spec

// DLTIOFF-LABEL: module
// DLTIOFF-NOT:   dlti

module {
  hw.module @Dummy() {
  }
}
