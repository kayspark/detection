#pragma once
#include "Frame.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <chrono>
#include <mutex>
#include <vector>

class MotionCapture {
  cv::VideoCapture *_capture = nullptr;
  std::chrono::milliseconds _currentTime;
  const int _timeRange; // in milliseconds
  int _fps;
  std::vector<cv::Vec4i> _hierarchy;
  cv::TermCriteria _criteria;
  cv::Size _winSize;
  cv::Mat _gray;
  cv::Mat _prevGray;
  cv::Mat _saved_mask;

  std::map<std::chrono::milliseconds, Frame> _frames;
  std::vector<std::map<std::chrono::milliseconds, std::vector<cv::Point>>> _allTracks;
  cv::Ptr<cv::BackgroundSubtractor> _pBgs;
  void getFeaturePoints(const std::vector<cv::Point> &in,
                        std::vector<cv::Point2f> &out);
  void uniteContours(std::vector<std::vector<cv::Point>> &cnts);

public:
  explicit MotionCapture(cv::VideoCapture &cap);
  ~MotionCapture();
  void detect_motions(std::vector<cv::Rect>& ROIs);
  void display();

  void fill_tracks(
      std::vector<std::map<std::chrono::milliseconds, std::vector<cv::Point>>>
      &allTracks,
      std::vector<std::vector<cv::Point>> &allContours) const;
};
