# cimrt: the C ABI runtime. Depends only on the core target reader, so it
# builds without an LLVM/MLIR toolchain.
add_library(cimrt STATIC
  src/simulator/simulator.cpp
  src/erbium/erbium_backend.cpp
)

target_include_directories(cimrt PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include
)
target_include_directories(cimrt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/simulator
)

target_link_libraries(cimrt PUBLIC cimTarget)

set_target_properties(cimrt PROPERTIES POSITION_INDEPENDENT_CODE ON)
