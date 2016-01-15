
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <iostream> // General io
#include <stdio.h> // General io
#include <map>

// Headers //
  #include "../Headers/filterbank.h"
  #include "imgCollection.h"
  #include "imgFunctions.h"

#define DicDEBUG 0 // Show all debugging messages
#define printMdls 1 // Print out model values

using namespace cv;
using namespace std;

// Print out message and double for use in debugging
void dicDEBUG(string msg, double in){
  if(DicDEBUG){
    cout << msg;
    if(in!=0){
      cout << in;
    }
    cout << "\n";
  }
}

// Calculate bin limits for each textonDictionary Value
void binLimits(vector<float>& tex, int numClusters){
  dicDEBUG("inside binLimits", 0);

  // Create and populate bins //

    vector<float> bins;// Create vector to store bin limits

    bins.push_back(0); // Push back 0 as bottom end of bins

    // Push back top end of bin
    for(int i = 0;i <= tex.size()-1;i++){
        bins.push_back(tex[i] + 0.00001);
    }
    bins.push_back(256); // Push back maximum bin value

  // Print Bins if Flag true //
    if(printMdls){
      cout << "\n";
      for(int i=0;i<bins.size();i++){
           cout << bins[i] << endl;
         }
    }
    cout << "\n";

  // Clear and pass back generated bins //
    tex.clear();
    tex = bins;
}

// Assign vector to Set to remove duplicates
void removeDups(vector<float>& v){
  dicDEBUG("inside removeDups", 0);
  sort(v.begin(), v.end());
  auto last = unique(v.begin(), v.end());
  v.erase(last, v.end());
}

// Convert Matrix values to vector<float>
vector<float> matToVec(Mat m){
  vector<float> v;
  for(int i=0;i<m.rows;i++){
    v.push_back(m.at<float>(i,0));
  }
  return v;
}

// Pass converted Matrix values to binLimits returning result
vector<float> createBins(Mat texDic, int numClusters){
  vector<float> v = matToVec(texDic);
  dicDEBUG("\n\nThis is the bin vector size BEFORE binlimits: ", v.size());
  binLimits(v, numClusters);
  dicDEBUG("\n\nThis is the bin vector size AFTER binlimits: ", v.size());
  return v;
}

// Filter input images
void filterImgs(map<string, vector<Mat> > imgs, map<string, vector<Mat> > &filteredImgs){
  // Create Filterbank //
    int n_sigmas, n_orientations;
    vector<vector<Mat> > filterbank;
    createFilterbank(filterbank, n_sigmas, n_orientations);

  // Cycle through images and Apply Filterbank saving responses to output //
    for(auto const imgs1 : imgs){
      for(int curImg=0;curImg<imgs1.second.size();curImg++){
        Mat in, out;
        in = imgs1.second[curImg];
        filterHandle(in, out, filterbank, n_sigmas, n_orientations, curImg);
        filteredImgs[imgs1.first].push_back(out);
      }
    }
}

// Handle all dictionary generation and saving operations
void dictCreateHandler(int cropsize, int scale, int numClusters, int flags, int attempts,
  int kmeansIteration, double kmeansEpsilon, boost::filesystem::path p){

  // Initialise BOWTrainer object //
    TermCriteria tc(TermCriteria::MAX_ITER, kmeansIteration, kmeansEpsilon);
    BOWKMeansTrainer bowTrainer(numClusters, tc, attempts, flags);

  // Load and filter images //
    map<string, vector<Mat> > textonImgs, filteredImgs;
    loadClassImgs(p, textonImgs, scale);
    filterImgs(textonImgs, filteredImgs);

  Mat dictionary;
  // Segment and aggregate for each class, clustering the aggregates to make the texton Dictionary //
    for(auto const ent1 : filteredImgs){

      for(int j=0;j<ent1.second.size();j++){
        Mat curImg = Mat::zeros(ent1.second[j].cols, ent1.second[j].rows,CV_32F);
        curImg = ent1.second[j];

        // Segment and flatten 200x200pixel image //
          vector<Mat> test;
          segmentImg(test, curImg, cropsize);

        // Debug print out
        dicDEBUG("after segmenation: ", test.size());

        // Push all segments to bowTrainer //
          for(int k = 0; k < test.size(); k++){
            if(!test[k].empty()){
              bowTrainer.add(test[k]);
            }
        }

        dicDEBUG("This is the bowTrainer.size(): ", bowTrainer.descriptorsCount());
        }

        // Generate specified num of clusters per class and store in Mat //
          dictionary.push_back(bowTrainer.cluster());
          bowTrainer.clear();
    }

  // Generate bins from dictionary //
    vector<float> bins = createBins(dictionary, numClusters);
    // Remove duplicate bins
    removeDups(bins);

  // Save to file //
    dicDEBUG("Saving Dictionary..", 0);
    FileStorage fs("dictionary.xml",FileStorage::WRITE);
    fs << "cropSize" << cropsize;
    fs << "clustersPerClass" << numClusters;
    fs << "totalDictSize" << dictionary.size();
    fs << "flagType" << flags;
    fs << "attempts" << attempts;
    stringstream ss;
    for(auto const ent1 : textonImgs){
      ss << ent1.first << " ";
    }
    fs << "classes" << ss.str();
    fs << "Kmeans" << "{";
      fs << "Iterations" << kmeansIteration;
      fs << "Epsilon" << kmeansEpsilon;
    fs << "}";
    fs << "vocabulary" << dictionary;
    fs << "bins" << bins;
    fs.release();
}
