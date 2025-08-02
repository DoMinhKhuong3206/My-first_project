#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <filesystem>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

Ort::Value create_tensor_from_image(Ort::Env& env, const std::string& image_path) {
    // 1. Read image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        throw std::runtime_error("Khong the doc anh " + image_path);
    }

    // 2. Resize image -> 28x28
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(28, 28));

    // 3. Normalization
    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32F, 1.0 / 255.0); //[0, 255] -> [0, 1]
    float_image = (float_image - 0.5) / 0.5; //[0, 1] -> [-1, 1]
    std::vector<float> input_tensor_values;
    input_tensor_values.assign(float_image.begin<float>(), float_image.end<float>());

    std::vector<int64_t> input_tensor_shape = {1, 1, 28, 28};

    // 4. Create tensor ONNX
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_tensor_shape.data(), input_tensor_shape.size());
}

// Read label from file labels.txt
std::vector<int> read_labels(const std::string& labels_path) {
    std::vector<int> labels;
    std::ifstream labels_file(labels_path);
    if (labels_file.is_open()) {
        std::string line;
        while (std::getline(labels_file, line)) {
            try {
                labels.push_back(std::stoi(line));
            } catch (const std::invalid_argument& e) {
                std::cerr << "Loi khi chuyen doi nhan tu dong: " << line << std::endl;
            }
        }
        labels_file.close();
    } else {
        throw std::runtime_error("Khong the mo file labels.txt.");
    }
    return labels;
}

int main() {
    std::cout << "--- Bat dau chuong trinh suy luan MNIST ---" << std::endl;
    
    try {
        // 1. Initialize ONNX Runtime Environment and Session Options
        std::cout << "1. Khoi tao ONNX Runtime..." << std::endl;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_inference");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // 2. Load Model
        std::string model_path = "mnist_inference/model.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);
        std::cout << "2. Tai mo hinh ONNX thanh cong." << std::endl;

        // 3. Access the model's input/output data
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();

        std::vector<std::string> input_node_names_str;
        std::vector<std::string> output_node_names_str;

        for (size_t i = 0; i < num_input_nodes; i++) {
            Ort::AllocatedStringPtr name_ptr = session.GetInputNameAllocated(i, allocator);
            input_node_names_str.push_back(name_ptr.get());
        }
        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::AllocatedStringPtr name_ptr = session.GetOutputNameAllocated(i, allocator);
            output_node_names_str.push_back(name_ptr.get());
        }

        std::vector<const char*> input_names_c;
        for (const auto& name : input_node_names_str) {
            input_names_c.push_back(name.c_str());
        }
        std::vector<const char*> output_names_c;
        for (const auto& name : output_node_names_str) {
            output_names_c.push_back(name.c_str());
        }
        
        std::cout << "   - Ten input: " << input_names_c[0] << std::endl;
        std::cout << "   - Ten output: " << output_names_c[0] << std::endl;

        // 4.Prepare data for inference
        std::cout << "4. Dang chuan bi du lieu va bat dau suy luan cho 10000 anh..." << std::endl;
        std::vector<int> actual_labels = read_labels("mnist_inference/labels.txt");
        int correct_predictions = 0;
        int total_images = actual_labels.size();

        for (int i = 0; i < total_images; ++i) {
            std::string image_path = "mnist_inference/test_images/" + std::to_string(i) + ".png";
            //Preprocess
            Ort::Value input_tensor = create_tensor_from_image(env, image_path);
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(input_tensor));
            //Inference
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_c.data(), input_tensors.data(), input_tensors.size(), output_names_c.data(), output_names_c.size());
            //Postprocess
            const float* raw_output = output_tensors[0].GetTensorMutableData<float>();
            
            int predicted_class = 0;
            float max_prob = 0.0f;
            for (int j = 0; j < 10; ++j) {
                if (raw_output[j] > max_prob) {
                    max_prob = raw_output[j];
                    predicted_class = j;
                }
            }

            if (predicted_class == actual_labels[i]) {
                correct_predictions++;
            }
        }

        // Find accuracy
        std::cout << "\n--- Ket qua suy luan toan bo 10000 anh ---" << std::endl;
        std::cout << "So luong anh du doan dung: " << correct_predictions << "/" << total_images << std::endl;
        double accuracy = static_cast<double>(correct_predictions) / total_images;
        std::cout << "Do chinh xac: " << accuracy * 100 << "%" << std::endl;

    } catch (const Ort::Exception& exception) {
        std::cerr << "Loi ONNX Runtime: " << exception.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Mot loi chung da xay ra: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "--- Chuong trinh ket thuc. ---" << std::endl;
    return 0;
}


