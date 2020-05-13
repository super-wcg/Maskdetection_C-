#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <stdint.h>
using namespace std;
using namespace cv;

typedef struct FaceInfo {
	Rect2f rect;
	float score;
	int id;
} FaceInfo;

class maskdetection {
public:
	void init_mask_modle();
	//bool increase(const FaceInfo& a, const FaceInfo& b);
	//vector<int> do_nms(vector<FaceInfo>& bboxes, float thresh, char methodType);
	//vector<int> single_class_non_max_suppression(vector<cv::Rect2f>& rects, float* confidences, int c_len, vector<int>& classes, vector <float>& bbox_max_scores);
	vector<cv::Rect2f> decode_bbox(vector<vector<float>>& anchors, float* raw);
	vector<vector<float>> generate_anchors(const vector<float>& ratios, const vector<int>& scales, vector<float>& anchor_base);
	bool mask_detection(cv::Mat img);
private:
	cv::Mat img;
	dnn::Net PNet_;
};