cmake_minimum_required(VERSION 2.8)
PROJECT(ModelDicGen)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost COMPONENTS system filesystem REQUIRED)

if(Boost_FOUND)
  INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

  ADD_EXECUTABLE(${PROJECT_NAME} ./mainMod.cpp ../Headers/filterbank.h ../Headers/filterbank.cpp
  ./imgCollection.cpp ./imgCollection.h ./modelBuild.cpp ./modelBuild.h
  ./imgFunctions.cpp ./imgFunctions.h ./dictCreation.cpp ./dictCreation.h)

  TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Boost_FILESYSTEM_LIBRARY}
    ${BOOST_SYSTEM_LIBRARY}
    ${Boost_LIBRARIES}
    )
  ADD_DEFINITIONS(-std=c++11)
endif()
  
