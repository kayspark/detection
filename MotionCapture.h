#pragma once
#include "Frame.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <chrono>
#include <mutex>
#include <vector>

class MotionCapture {
  cv::VideoCapture *capture = nullptr;
  std::chrono::milliseconds currentTime;
  const int timeRange; // in milliseconds
  int fps;
  std::vector<cv::Vec4i> hierarchy;
  cv::TermCriteria criteria;
  cv::Size winSize;
  cv::Mat gray;
  cv::Mat prevGray;
  cv::Mat savemask;

  std::map<std::chrono::milliseconds, Frame> _frames;
  std::vector<std::map<std::chrono::milliseconds, std::vector<cv::Point>>> allTracks;
  cv::Ptr<cv::BackgroundSubtractor> pBgs;
  void getFeaturePoints(const std::vector<cv::Point> &in,
                        std::vector<cv::Point2f> &out);
  void uniteContours(std::vector<std::vector<cv::Point>> &cnts);

public:
  explicit MotionCapture(cv::VideoCapture &cap);
  ~MotionCapture();
  void find();
  void display();

  void fill_tracks(
      std::vector<std::map<std::chrono::milliseconds, std::vector<cv::Point>>>
      &allTracks,
      std::vector<std::vector<cv::Point>> &allContours) const;
};
