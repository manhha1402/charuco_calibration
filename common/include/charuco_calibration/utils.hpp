#pragma once

#include "charuco_calibration/calibrator.hpp"

#include <ros/ros.h>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>

#include <iostream>

namespace charuco_calibration
{
bool readDetectorParameters(ros::NodeHandle& nh, cv::Ptr<cv::aruco::DetectorParameters>& params);
bool readCalibrationFlags(ros::NodeHandle& nh, int& calibrationFlags);
void readCalibratorParams(ros::NodeHandle& nh, Calibrator& calibrator);

std::ostream& saveCameraInfo(std::ostream& output, CalibrationResult& result);

void setLoggerName(const std::string& loggerName);

}  // namespace charuco_calibration
