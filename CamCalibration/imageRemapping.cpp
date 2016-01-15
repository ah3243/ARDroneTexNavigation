// Practice for Clustering with Opencv BOW classes

#include "opencv2/highgui/highgui.hpp" // Needed for HistCalc
#include "opencv2/imgproc/imgproc.hpp" // Needed for HistCalc
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include <iostream> // General io
#include <stdlib.h>
#include <fstream>
#include <string>
#include <algorithm> // Maybe fix DescriptorExtractor doesn't have a member 'create'
#include <boost/filesystem.hpp>
#include <iostream>
#include <map>

using namespace cv;
using namespace std;

#define verbose 0
#define showImgs 0

// Check for input dir, exit if none
void checkInDir(string inDir){
  if(!boost::filesystem::exists(inDir) && boost::filesystem::is_directory(inDir)){
    cout << "Input image dir not able to be found. Exiting."<< endl;
    exit(1);
  }
}
// Regerate Output Dir
void checkImgDirs(string outDir){
  boost::filesystem::remove_all(outDir);
  if(boost::filesystem::create_directory(outDir)){
    if(verbose){
      cout << "Directory Created: " << outDir << endl;
    }
  }else{
    if(verbose){
      cout << "Directory creation failed" << outDir << endl;
    }
  }
}

// Extract Class Names from Dir Paths
string extractClsNmes(string path){
  size_t found = path.find_last_of('/');
  return path.substr(found+1);
}

// Get all Dirs at path
void getDirNmes(string inDir, vector<string>& clsPaths){
  boost::filesystem::directory_iterator end_iter;

  for( boost::filesystem::directory_iterator dir_iter(inDir) ; dir_iter != end_iter ; ++dir_iter)
  {
    if (boost::filesystem::is_directory(*dir_iter) )
    {
      clsPaths.push_back(dir_iter->path().string());
    }
  }
}

// Creates individual output Class Folders
void checkClsoutDirs(string dir, vector<string> clsNmes){
  for(int j=0;j<clsNmes.size();j++){
    stringstream ss;
    ss << dir << "/" << clsNmes[j];
    checkImgDirs(ss.str());
  }
}

// getting all .jpg fileNames in directory
void getImgsinDir(string inDir, vector<string> &filePaths){
  // Loop through input dir
  boost::filesystem::directory_iterator end_iter;
  for( boost::filesystem::directory_iterator dir_iter(inDir) ; dir_iter != end_iter ; ++dir_iter)
  {
    // Confirm file is regular file
    if (boost::filesystem::is_regular_file(*dir_iter) )
    {
      // Confirm file is .jpg
      if(dir_iter->path().extension().string() == ".jpg"){
        if(verbose){
          cout << "This is the filename: " << *dir_iter << endl;
        }
        // Save to vector<string>
        filePaths.push_back(dir_iter->path().string());
      }
    }
  }
}

void importImages(map<string, map<string, Mat> > &ClassImgs, vector<string> clsPaths){
  if(verbose){
    cout << "Getting Classes." << endl;
  }
  // Get all image Paths
  map< string, vector<string> > filePaths;
  for(int i=0;i<clsPaths.size();i++){
    // Get Overall Class Name
    string cls;
    cls = extractClsNmes(clsPaths[i]);
    // Get all FileNames from specific Class
    vector<string> a;
    getImgsinDir(clsPaths[i], a);
    filePaths[cls] = a;
  }
  // Import Images
  for(auto const ent : filePaths){
    map<string, Mat> tmpClsImgs; // Store each images fileName and Mat
    for(int j=0;j<ent.second.size();j++){
      // Read in and save image to Mat
      Mat tmp;
      tmp = imread(ent.second[j], CV_LOAD_IMAGE_COLOR);
      string fileName = extractClsNmes(ent.second[j]); // Store individual File Name
      if(verbose){
        cout << "This is the fileName: " << fileName << endl;
      }
      tmpClsImgs[fileName] = tmp; // Save each image to map with filename
    }
    ClassImgs[ent.first] = tmpClsImgs; // save each classes files to map
  }
}

void getRemapMats(Mat &map1, Mat &map2){
  Mat cameraMatrix, distCoeffs;
  double image_Width, image_Height;

  // Get Distortion Coefficients and Camera Matrix from file
  cout << "Start.." << endl;
  FileStorage fs("/home/james-tt/Desktop/MyFilterbankCode/imageRemapping/calFile.xml", FileStorage::READ);
  fs["Camera_Matrix"] >> cameraMatrix;
  fs["Distortion_Coefficients"] >> distCoeffs;
  fs["image_Width"] >> image_Width;
  fs["image_Height"] >> image_Height;

  fs.release();
  // Set size
  Size imageSize(image_Width, image_Height);

  // Create maps
  initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
      getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0, imageSize, 0),
      imageSize, CV_16SC2, map1, map2);
}

int main(int argc, char** argv){
  string inDir = "/home/james-tt/Desktop/TEST_IMAGES/ARImgs/BACKUP";
  string outDir = "/home/james-tt/Desktop/TEST_IMAGES/ARImgs/Normalised";
  vector<string> clsPaths;
  vector<string> clsNmes;
  checkInDir(inDir); // Check for input img Dir

  getDirNmes(inDir, clsPaths); // Get all class dir paths
  for(int i=0;i<clsPaths.size();i++){
    clsNmes.push_back(extractClsNmes(clsPaths[i])); // Extract Class Names from Paths
  }
  if(verbose){
    cout << "This is the number of Classes: " << clsPaths.size() << endl;
  }

  checkImgDirs(outDir); // Remove old output Dir and create New Dir
  checkClsoutDirs(outDir, clsNmes); // Create new individual class output Dirs

  // Import All Images
  map<string, map<string, Mat> > ClassImgs; // Multiple Classes < each Class Name, multiple images per class < each imageName, Image > >
  importImages(ClassImgs, clsPaths); // Import all images and store

  // Create variables
  Mat map1, map2;

  getRemapMats(map1, map2);

  // Create remap variables
  Mat view, rview;

  // Create display Windows
  namedWindow("before", 0);
  namedWindow("after", 0);

  // map<string, map<string, Mat> > ClassImgs; // Multiple Classes < each Class Name, multiple images per class < each imageName, Image > >

  // Loop through imgs in dir
  for(auto const ent1 : ClassImgs){
    cout << "going through Class: " << ent1.first << endl;
    for(auto const ent2 : ent1.second){
      stringstream ss;
      cout << "this is the FileNme: " << ent2.first << " and size: " << ent2.second.size() << endl;
      view = ent2.second;
      // Remap image
      remap(view, rview, map1, map2, INTER_LINEAR);
      if(showImgs){
       imshow("before", view);
       imshow("after", rview);
        waitKey(1000);
      }
      ss << outDir << "/" << ent1.first << "/" << ent2.first;
      imwrite(ss.str(), rview);
      char c = waitKey(30);
      if(c=='q'){
        break;
      }
    }
  }
  return 0;
}
