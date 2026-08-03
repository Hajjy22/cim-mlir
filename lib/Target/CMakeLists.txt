# Target description reader. Dependency-free (see the comment at the top of
# TargetYAMLParser.cpp for why it does not use llvm::yaml).
add_library(cimTarget STATIC
  TargetYAMLParser.cpp
)

target_include_directories(cimTarget PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(cimTarget PROPERTIES POSITION_INDEPENDENT_CODE ON)
