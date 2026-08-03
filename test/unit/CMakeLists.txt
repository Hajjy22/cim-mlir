add_executable(cim-unit-tests
  test_main.cpp
  placement_test.cpp
  target_spec_test.cpp
)

target_link_libraries(cim-unit-tests PRIVATE
  cimPlacement
  cimTarget
)

target_include_directories(cim-unit-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Run from the repo root so the target-file tests can find targets/*.yaml.
add_test(NAME cim-unit-tests
  COMMAND cim-unit-tests
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)

# `check-cim` is the one command that runs everything buildable in the
# current configuration.
add_custom_target(check-cim
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS cim-unit-tests
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  COMMENT "Running cim-mlir core unit tests"
)
