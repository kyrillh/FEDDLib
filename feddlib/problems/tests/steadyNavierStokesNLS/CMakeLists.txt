TRIBITS_ADD_EXECUTABLE(
  steadyNavierStokesNLS
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
  steadyNavierStokesNLS
  NAME steadyNavierStokesNLS
	ARGS
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyNavierStokesNLS
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCE_FILES parametersProblem.xml parametersSolverNonLinSchwarz.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
  DEST_FILES parametersProblem.xml parametersSolverNonLinSchwarz.xml
  EXEDEPS steadyNavierStokesNLS
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyNavierStokesNLS
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	SOURCE_FILES square.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES square.mesh
  EXEDEPS steadyNavierStokesNLS
)
