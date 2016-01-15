#!/bin/bash

  # Function to output to stderr
  DEBUG=0
  if [ $DEBUG -eq 1 ]
  then
    echoerr(){ echo "$@" 1>&2; }
  else
    echoerr(){ echo ; }
  fi

  #---------Program and Parameters----------#
  #GOAL CLASS#
  goal="grass"

  #Image Scale and Segment Size
  scale=8
  cropSize=70

  # Check for an image path in input parameters, if none exit
  if [ "$#" -ne 1 ]; then
    echo no imageLocation detected, Exiting.
    exit
    imgLocation=""
  else
    echoerr input image path is: "$1"
    imgLocation="$1"
  fi

  # 1:"flag" 2:"image path" 3:"scale" 4:"cropSize" 5:"goal"
 echo $(singleImgEval $imgLocation $scale $cropSize $goal)
