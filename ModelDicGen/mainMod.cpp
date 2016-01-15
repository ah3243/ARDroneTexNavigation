 ///////////////////////////////////////
// Opencv BoW Texture classification //
///////////////////////////////////////

#include "opencv2/core/core.hpp" // Needed for imports
#include <iostream> // General io
#include <stdio.h> // General io
#include <boost/filesystem.hpp> // For image dir path

#include "dictCreation.h" // Generate and store Texton Dictionary
#include "modelBuild.h" // Generate models from class images

#define VERBOSE 0

#define DICTIONARY_BUILD 1
#define MODEL_BUILD 1

#define flags KMEANS_PP_CENTERS // Kmeans initialisation method

// Error printout
#define ERR(msg) printf("\n\nERROR!: %s Line %d\nExiting.\n\n", msg, __LINE__);

using namespace boost::filesystem;
using namespace cv;
using namespace std;

int main( int argc, char** argv ){

  cout << "\n.......Starting Program...... \n" ;

  // Print out number and values of inputs //
    cout << "number of inputs: " << argc << endl;
    for(int i=0;i<argc;i++){
      cout << i << ": " << argv[i] << endl;
    }

  // Initialising Variables //
  cout << "\n......Initialising Variables......\n";

    path clsPath = "../../TrainingImages/Models";
    path textonPath = "../../TrainingImages/TexDict";

    // Kmeans parameters
      int kmeansIteration = 100000;
      double kmeansEpsilon = 0.000001;
      int numClusters = 10; // For model and test images
      int attempts = 35; // The number of kmeans attempts
    // Number of Model and Texton Dictionary repeats per image
      int modelRepeats = 1;
      int DictSize = 10;
    // Image parameters
      int scale = 8; // rescale input image to 256x144 pixels
      int cropsize = 70; // Allows for 6 segments per image(this is reduced to 4 during use)

    if(scale==0 || cropsize==0){
      ERR("Scale or cropping size set. Exiting.");
      exit(1);
    }

  // Print variables //
    cout << "\n.......Variables.......\n";
    cout << "\nDictionary Size: " << DictSize << "\nNumber of Clusters: " << numClusters << "\nAttempts: " << attempts << "\nIterations: "
    << kmeansIteration << "\nKmeans Epsilon: " << kmeansEpsilon << endl;
    cout << "This is the cropsize: " << cropsize << "\n";
    cout << "This is the scalesize: " << scale << endl;
    cout << "This is the number of modelRepeats: " << modelRepeats << endl;


  // Run Dictionary Module //
  #if DICTIONARY_BUILD == 1

    cout << "\n.......Generating Texton Dictionary...... \n" ;

    dictCreateHandler(cropsize, scale, DictSize, flags, attempts, kmeansIteration, kmeansEpsilon);

  #endif

  // Run Model Generation Module //
  #if MODEL_BUILD == 1

    cout << "\n........Generating Class Models from Imgs.........\n";

    modelBuildHandle(cropsize, scale, numClusters, flags, attempts, kmeansIteration, kmeansEpsilon, modelRepeats);

  #endif

  cout << "..........ENDING PROGRAM.............\n";
  return 0;
}
