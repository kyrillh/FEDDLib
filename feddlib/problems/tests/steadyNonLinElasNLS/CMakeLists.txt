TRIBITS_ADD_EXECUTABLE(
  steadyNonLinElasNLS
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
  steadyNonLinElasNLS
  NAME steadyNonLinElasNLS
	ARGS
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyNonLinElasNLS
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCE_FILES parametersProblem.xml parametersSolverNonLinSchwarz.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
  DEST_FILES parametersProblem.xml parametersSolverNonLinSchwarz.xml
  EXEDEPS steadyNonLinElasNLS
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyNonLinElasNLS
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
  SOURCE_FILES beam2D100.mesh
  SOURCE_FILES beam2D26k.mesh
  SOURCE_FILES beam2D59k.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
  DEST_FILES beam2D100.mesh
  DEST_FILES beam2D26k.mesh
  DEST_FILES beam2D59k.mesh
  EXEDEPS steadyNonLinElasNLS
)
