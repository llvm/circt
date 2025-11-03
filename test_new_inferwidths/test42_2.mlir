firrtl.circuit "Issue4859" {
  firrtl.module @Issue4859() {
    %invalid = firrtl.invalidvalue : !firrtl.bundle<a: vector<uint, 2>>
    %0 = firrtl.subfield %invalid[a] : !firrtl.bundle<a: vector<uint, 2>>
    %1 = firrtl.subindex %0[0] : !firrtl.vector<uint, 2>
  }
}