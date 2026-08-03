# Unit tests for the MLIR-free core: always built, and fast enough to be
# the first thing CI runs.
add_subdirectory(unit)

# Tests that need MLIR: the numerical end-to-end suite and the lit suite.
if(CIM_ENABLE_MLIR)
  add_subdirectory(mlir)

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

  set(CIM_TEST_DEPENDS
    FileCheck
    cim-opt
  )

  add_lit_testsuite(check-cim-opt "Running the cim-mlir dialect tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CIM_TEST_DEPENDS}
  )
  set_target_properties(check-cim-opt PROPERTIES FOLDER "Tests")

  add_lit_testsuites(CIM_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CIM_TEST_DEPENDS})
endif()
