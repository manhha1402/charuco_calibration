#include "charuco_calibration/calibrator.hpp"
#include "charuco_calibration/utils.hpp"

#include <ros/ros.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <dirent.h>

#include <string>
#include <unordered_set>
#include <vector>

#include <fstream>

/**
 * Get a list of files at a given path.
 *
 * @param path A path to search at.
 * @param extensions A set of allowed extensions (empty set means return all files).
 * @return A vector of std::string containing all matched file names at a path.
 */
std::vector<std::string> getFilesAt(const std::string& path, const std::unordered_set<std::string>& extensions = {})
{
	DIR* dir = opendir(path.c_str());
	if (dir == NULL)
	{
		ROS_WARN_STREAM("Could not open directory at " << path);
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

static void logFunction(charuco_calibration::LogLevel logLevel, const std::string& message)
{
	ros::console::levels::Level rosLogLevel;
	switch (logLevel)
	{
		case charuco_calibration::LogLevel::DEBUG:
			rosLogLevel = ros::console::levels::Level::Debug;
			break;
		case charuco_calibration::LogLevel::INFO:
			rosLogLevel = ros::console::levels::Level::Info;
			break;
		case charuco_calibration::LogLevel::WARN:
			rosLogLevel = ros::console::levels::Level::Warn;
			break;
		case charuco_calibration::LogLevel::ERROR:
			rosLogLevel = ros::console::levels::Level::Error;
			break;
		case charuco_calibration::LogLevel::FATAL:
			rosLogLevel = ros::console::levels::Level::Fatal;
			break;
		default:
			// This should never happen, but if it does, everything went south
			rosLogLevel = ros::console::levels::Level::Fatal;
	}
	ROS_LOG(rosLogLevel, ROSCONSOLE_DEFAULT_NAME, "%s", message.c_str());
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "cv_calib_files");

	ros::NodeHandle nh, nhPriv("~");
	ros::NodeHandle nhDetector("~detector_parameters");

	charuco_calibration::Calibrator calibrator;
	charuco_calibration::readCalibratorParams(nhPriv, calibrator);
  //charuco_calibration::readDetectorParameters(nhDetector, calibrator.aruco_detector_params_);
	calibrator.setLogger(logFunction);

	std::string imgPath = nhPriv.param<std::string>("images_path", "calibration_images");
	std::string outputFile = nhPriv.param<std::string>("output_file", "calibration.yaml");

	auto imageFiles = getFilesAt(imgPath, {".png", ".jpg", ".jpeg"});

	if (imageFiles.size() == 0)
	{
		ROS_ERROR_STREAM("Could not get files at " << imgPath);
	}
	else
	{
		ROS_INFO_STREAM("Using the following files at " << imgPath << " for calibration:");
		for (const auto& fileName : imageFiles)
		{
			ROS_INFO_STREAM(" - " << fileName);
		}
	}

	for (const auto& fileName : imageFiles)
	{
		cv::Mat image = cv::imread(imgPath + "/" + fileName);
		if (image.data == nullptr)
		{
			ROS_WARN_STREAM("Could not open " << fileName << " as a CV image, skipping");
			continue;
		}
		auto detectionResult = calibrator.processImage(image);
    auto displayedImage = calibrator.drawDetectionResults(detectionResult);

		if (detectionResult.isValid())
		{
			ROS_INFO_STREAM("Added " << fileName << " to calibration set");       
			calibrator.addToCalibrationList(detectionResult);
		}
		else
		{
			ROS_INFO_STREAM("Rejected " << fileName << " due to insufficient features");
		}
	}


	auto calibResult = calibrator.performCalibration();
	if (calibResult.isValid)
	{
		std::string outputFilePath = imgPath + "/" + outputFile;
		std::ofstream outFile(outputFilePath);
		outFile << "# File generated by charuco_calibration" << std::endl;
		outFile << "# ArUco reprojection error: " << calibResult.arucoReprojectionError << std::endl;
		outFile << "# Reprojection error: " << calibResult.reprojectionError << std::endl;
		saveCameraInfo(outFile, calibResult);
		if (!outFile)
		{
			ROS_ERROR("Encountered an error while writing result");
		}
		else
		{
			ROS_INFO_STREAM("Calibration saved to " << outputFilePath);
		}
	}

	return 0;
}