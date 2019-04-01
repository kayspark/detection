#include <opencv2/highgui.hpp>
#include "MotionCapture.hpp"
#include <filesystem>
#include <string_view>
using namespace std;
using namespace cv;
using namespace chrono;
// namespace fs = std::__fs::filesystem;
namespace fs = std::filesystem;

int main() {
  // const int cameraNumber = 0;
#ifdef _WIN32
  const string_view fileName("Y://Downloads//hanjun//hanjun3.avi");
#else
  const string_view fileName("/Users/kspark/Documents/GitHub/detection/TB.mp4");
  // const string_view fileName("~/Downloads/hanjun/hanjun3.avi");
#endif //_WIN32
  fs::path p = fileName;
  if (p.empty()) {
    cout << "Error: File " << fileName << " is not found." << endl;
    return -1;
  }
  std::vector<cv::Rect> ROIs;

  cv::VideoCapture capture(fileName.data());
  MotionCapture mc(capture);
  while (capture.isOpened()) {
    mc.detect_motions(ROIs);
    mc.display();
    if (cv::waitKey(5) == 27)
      break;
  }

  return 0;
}
