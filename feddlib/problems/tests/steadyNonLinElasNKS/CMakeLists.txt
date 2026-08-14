TRIBITS_ADD_EXECUTABLE(
  steadyNonLinElasNKS
	SOURCES main.cpp
)

TRIBITS_ADD_TEST(
  steadyNonLinElasNKS
  NAME steadyNonLinElasNKS
	ARGS
	NUM_MPI_PROCS 4
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(data_steadyNonLinElasNKS
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
	SOURCE_FILES parametersPrec.xml  parametersProblem.xml parametersSolver.xml
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES parametersPrec.xml  parametersProblem.xml parametersSolver.xml
  EXEDEPS steadyNonLinElasNKS
)

TRIBITS_COPY_FILES_TO_BINARY_DIR(mesh_steadyNonLinElasNKS
	SOURCE_DIR ${CMAKE_SOURCE_DIR}/meshes
	SOURCE_FILES square.mesh
  SOURCE_FILES beam2D100.mesh
  SOURCE_FILES beam2D59k.mesh
	DEST_DIR ${CMAKE_CURRENT_BINARY_DIR}
	DEST_FILES square.mesh
  DEST_FILES beam2D100.mesh
  DEST_FILES beam2D59k.mesh
  EXEDEPS steadyNonLinElasNKS
)
