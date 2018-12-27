#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <mutex>
#include <chrono>
#include "Frame.h"
#include <vector>

class MotionCapture {
  cv::VideoCapture capture;
  std::chrono::milliseconds currentTime;
  const int timeRange; // in milliseconds
  int fps;

  std::map<chrono::milliseconds, Frame> _frames;
  std::vector<std::map<chrono::milliseconds, std::vector<cv::Point>>> allTracks;
  cv::Ptr<cv::BackgroundSubtractor> pBgs;
  void getFeaturePoints(const std::vector<cv::Point> &in, std::vector<cv::Point2f> &out);
  void uniteContours(std::vector<std::vector<cv::Point>> &cnts);

public:
  explicit MotionCapture(std::string &fileName);
  ~MotionCapture();
  void find();
  void display();

  void fill_tracks(std::vector<std::map<std::chrono::milliseconds, std::vector<cv::Point>>> &allTracks,
                   std::vector<std::vector<cv::Point>> &allContours) const;
};
