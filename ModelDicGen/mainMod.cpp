 ///////////////////////////////////////
// Opencv BoW Texture classification //
///////////////////////////////////////

#include "opencv2/highgui/highgui.hpp" // Needed for HistCalc
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <iostream> // General io
#include <stdio.h> // General io
#include <string>
#include <boost/filesystem.hpp>
#include <assert.h>
#include <chrono>  // time measurement
#include <thread>  // time measurement

#include "dictCreation.h" // Generate and store Texton Dictionary
#include "modelBuild.h" // Generate models from class images

#define VERBOSE 0

#define DICTIONARY_BUILD 1
#define MODEL_BUILD 1

#define kmeansIteration 100000
#define kmeansEpsilon 0.000001
#define numClusters 10 // For model and test images
#define flags KMEANS_PP_CENTERS

#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

using namespace boost::filesystem;
using namespace cv;
using namespace std;

int main( int argc, char** argv ){

  cout << "\n.......Starting Program...... \n" ;

  // Print out number and values of inputs
  cout << "number of inputs: " << argc << endl;
  for(int i=0;i<argc;i++){
    cout << i << ": " << argv[i] << endl;
  }

  cout << "// --- Initialising Variables --- //\n";

  int scale = 8; // rescale input image to 256x144 pixels
  int attempts=35; // The number of kmeans attempts

  // Number of Model and Texton Dictionary repeats per image
  int modelRepeats = 1;
  int DictSize = 10;

  int cropsize = 70; // Allows for 6 segments per image(this is reduced to 4 during use)

  String folderName=argv[1]; // Set subfolder name to number of repeats

  double modOverlap = 0; // Percentage of crop which will overlap horizontally
  double modelOverlapDb = (modOverlap/100)*cropsize; // Calculate percentage of cropsize
  int modelOverlap = modelOverlapDb; // To convert to int for transfer to function

  int testimgOverlap =modelOverlap; // Have the same test and model overlap

  int dictDur, modDur, novDur;

  if(scale==0){
    ERR("Scale no set. Exiting.");
    exit(1);
  }

  cout << "\nDictionary Size: " << DictSize << "\nNumber of Clusters: " << numClusters << "\nAttempts: " << attempts << "\nIterations: "
  << kmeansIteration << "\nKmeans Epsilon: " << kmeansEpsilon << endl;
  cout << "This is the cropsize: " << cropsize << "\n";
  cout << "This is the scalesize: " << scale << endl;
  cout << "This is the number of modelRepeats: " << modelRepeats << endl;

  path clsPath = "../../TrainingImages/Models";
  path textonPath = "../../TrainingImages/TexDict";

  #if DICTIONARY_BUILD == 1
    ////////////////////////////////
    // Creating Texton vocabulary //
    ////////////////////////////////

    cout << "\n.......Generating Texton Dictionary...... \n" ;
    // Measure start time
    auto t1 = std::chrono::high_resolution_clock::now();

    dictCreateHandler(cropsize, scale, DictSize, flags, attempts, kmeansIteration, kmeansEpsilon, modelOverlap);

    // Measure time efficiency
    auto t2 = std::chrono::high_resolution_clock::now();
    dictDur = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    waitKey(500); // a a precausion to ensure all processes are finished
  #endif
  #if MODEL_BUILD == 1
    ///////////////////////////////////////////////////////////
    // Get histogram responses using vocabulary from Classes //
    ///////////////////////////////////////////////////////////

    cout << "\n........Generating Class Models from Imgs.........\n";
    // Measure start time
    auto t3 = std::chrono::high_resolution_clock::now();

    modelBuildHandle(cropsize, scale, numClusters, flags, attempts, kmeansIteration, kmeansEpsilon, modelOverlap, modelRepeats);

    // Measure time efficiency
    auto t4 = std::chrono::high_resolution_clock::now();
    modDur = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    waitKey(500); // a a precausion to ensure all processes are finished
  #endif

  int totalTime =0;
  if(DICTIONARY_BUILD == 1){
    cout << "\n";

    cout  << dictDur;
    totalTime +=dictDur;
  }
  if(MODEL_BUILD == 1){
    cout << "\n";
    cout << modDur;
    totalTime+=modDur;
  }
  cout << "\n";
  cout<< totalTime << "\n";

  cout <<"\nENDING RUN\n";
  cout << "This is the cropsize: " << cropsize << endl;
  cout << "This is the scalesize: " << scale << endl;
  cout << "Dictionary Size: " << DictSize << "\nNumber of Clusters: " << numClusters << "\nAttempts: " << attempts << "\nIterations: "
  << kmeansIteration << "\nKmeans Epsilon: " << kmeansEpsilon << endl;

  // Show Finished Sign //
  namedWindow("finished", CV_WINDOW_AUTOSIZE);
  moveWindow("finished", 500,200);
  Mat finished = Mat(300,500,CV_8UC1, Scalar(255,255,255));
  string fini = "...FINISHED...";
  Point org(100, 100);
  putText(finished, fini, org, CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 0), 2, 8 );
  imshow("finished", finished);

  cout << "..........ENDING PROGRAM.............\n";
  return 0;
}
