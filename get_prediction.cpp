/***
 * This is a sample program to demonstrate the use of the model_loader.h files
 ***/

#include "./saved_model_loader.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define THRESHOLD 0.8


int main(int argc, char* argv[]){
	if (argc != 4){
		std::cout << "Error! Usage: <path/to_saved_model> <path/to_input/image.jpg> <path/to/output/image.jpg>" << std::endl;
		return 1;
	}

	// Make a Prediction instance
	Prediction out_pred;
	out_pred.boxes = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
	out_pred.scores = std::unique_ptr<std::vector<float>>(new std::vector<float>());
	out_pred.labels = std::unique_ptr<std::vector<int>>(new std::vector<int>());

	const string model_path = argv[1]; 
	const string test_image_file  = argv[2];
	const string test_prediction_image = argv[3];

	// Load the saved_model
	ModelLoader model(model_path);

	//Predict on the input image
	model.predict(test_image_file, out_pred);

	using namespace cv;
	Mat img = imread(test_image_file, IMREAD_COLOR);

	Size size = img.size();
	int height = size.height;
	int width = size.width;

	auto boxes = (*out_pred.boxes);
	auto scores = (*out_pred.scores);

	for (int i=0; i < boxes.size(); i++){
	    auto box = boxes[i];
	    auto score = scores[i];
	    if (score < THRESHOLD){
	        continue;
	    }
		int ymin = (int) (box[0] * height);
		int xmin = (int) (box[1] * width);
		int h = (int) (box[2] * height) - ymin;
		int w = (int) (box[3] * width) - xmin;
		Rect rect = Rect(xmin, ymin, w, h);
		rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
	}

	if (img.empty()){
		std::cout <<" Failed to read image" << std::endl;
	}

	imwrite(test_prediction_image, img);
}
