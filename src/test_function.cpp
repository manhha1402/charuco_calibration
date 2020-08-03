#include <sys/stat.h>
#include <sys/types.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
#include <dirent.h>
#include <string>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/imgproc.hpp>
std::vector<std::string> getFilesAt(const std::string& path, const std::unordered_set<std::string>& extensions = {})
{
  DIR* dir = opendir(path.c_str());
  if (dir == NULL)
  {
    std::cout<<"Could not open directory at "<< path<<std::endl;
    return {};
  }

  std::vector<std::string> fileList;

  for (dirent* entry = readdir(dir); entry != NULL; entry = readdir(dir))
  {
    if (entry->d_type == DT_REG)
    {
      std::string fileName = entry->d_name;
      if (extensions.size() > 0)
      {
        const auto dotPos = fileName.find_last_of('.');
        const std::string extension = fileName.substr(dotPos);
        if (extensions.find(extension) == extensions.end())
        {
          continue;
        }
      }
      fileList.push_back(fileName);
    }
  }

  closedir(dir);
  return fileList;
}

using namespace cv;
int main(int argc, char* argv[])
{ int squaresX = 14;
  int squaresY = 10;
  float squareLength = 0.028;
  float markerLength = 0.017;
  int dictionaryId = 1;
  bool showChessboardCorners = true;
  int calibrationFlags = 0;
  float aspectRatio = 1;
  calibrationFlags |= CALIB_FIX_ASPECT_RATIO;
  std::cout<<calibrationFlags<<std::endl;
  calibrationFlags |= CALIB_ZERO_TANGENT_DIST;
  std::cout<<calibrationFlags<<std::endl;

  calibrationFlags |= CALIB_FIX_PRINCIPAL_POINT;
  std::cout<<calibrationFlags<<std::endl;
  cv::Mat image = cv::imread("/home/manhha/iphone_calibration_images_png/IMG_3926.png");

  cv::Ptr<cv::aruco::Dictionary> dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));
  Ptr<aruco::CharucoBoard> charucoboard =
      aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
  Ptr<aruco::Board> board = charucoboard.staticCast<aruco::Board>();
  Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
  std::vector< int > ids;
  std::vector< std::vector< Point2f > > corners, rejected;
  // detect markers
  aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
  // refind strategy to detect more markers
  aruco::refineDetectedMarkers(image, board, corners, ids, rejected);

  // interpolate charuco corners
  Mat currentCharucoCorners, currentCharucoIds;
  if(ids.size() > 0)
    aruco::interpolateCornersCharuco(corners, ids, image, charucoboard, currentCharucoCorners,
                                     currentCharucoIds);
  cv::Mat imageCopy;
  // draw results
  image.copyTo(imageCopy);
  if(ids.size() > 0) aruco::drawDetectedMarkers(imageCopy, corners);

  if(currentCharucoCorners.total() > 0)
    aruco::drawDetectedCornersCharuco(imageCopy, currentCharucoCorners, currentCharucoIds);
  putText(imageCopy, "Press 'c' to add current frame. 'ESC' to finish and calibrate",
          Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 2);
  cv::namedWindow("out", cv::WINDOW_NORMAL);
  imshow("out", imageCopy);
  char key = (char)waitKey(0);
  if(key == 27) cv::destroyAllWindows();


  return 0;
}
