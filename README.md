# Tài liệu tóm tắt thông tin về mã nguồn môn Khai phá dữ liệu - Nhóm 6

Danh sách thành viên trong nhóm

  - Đoàn Thị Hoài Thu - Nhóm trưởng
  - Trần Quang Tuấn
  - Đỗ Tuấn Anh
  - Nguyễn Văn Dương
  - Nguyễn Văn Hoàn

# Cấu trúc thư mục

  - model: các mô hình sử dụng để huấn luyện. Gồm có svm, random forest, dnn.
  - transformer: package chứa một số tác vụ trích xuất đặc trưng.
  - testing: package chứa mã nguồn huấn luyện các mô hình
  - testing/svm: mã nguồn huấn luyện và test cho mô hình SVM
  - testing/random_forest: mã nguồn huấn luyện và test cho mô hình Random Forest
  - testing/dnn: mã nguồn huấn luyện và test cho mô hình DNN
  - linh_tinh: chưa các đoạn mã nguồn mà nhóm tạo ra trong quá trình học và làm việc, dùng để thử một số thứ, không quan trọng.
  - utils.py: chứa các thao tác liên quan đến việc lấy kết quả dự đoán của mô hình, xuất ra file.


# Hướng dẫn chạy mô hình SVM
Các mô hình khác tương tự:
  - testing/svm/svm_final_training.py: train mô hình trên 100% dữ liệu ban đầu thầy gửi, lưu model vào cùng vị trí thư mục.
  - testing/svm/model_testing.py: sử dụng mô hình để dự doán, đầu vào là tên file test và tên model đã lưu ở bước trước. Kết quả sẽ lưu vào file result.txt và result.xlsx.
 - cần chú ý đến các dòng mã nguồn liên quan đến tên model file, tên file test, tên file train để có kết quả chính xác.


