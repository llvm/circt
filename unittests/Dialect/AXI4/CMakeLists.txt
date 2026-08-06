add_circt_unittest(CIRCTAXI4Tests
  BurstSetTest.cpp
)

target_link_libraries(CIRCTAXI4Tests
  PRIVATE
  CIRCTAXI4
  MLIRIR
)
