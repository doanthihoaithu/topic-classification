from src.utils import get_project_root, predict_and_save_to_txt, predict_and_save_to_xlsx

ROOT_DIR = get_project_root()
working_dir = str(ROOT_DIR) + "/src/testing/svm"
print(ROOT_DIR)
test_file = str(ROOT_DIR) + "/resources/data.txt"
model_file = working_dir + "/" + "svm_final_model_official.sav"

txt_result_file = working_dir + "/" + "result.txt"
predict_and_save_to_txt(test_file, model_file, txt_result_file)

xlsx_result_file = working_dir + "/" + "result.xlsx"
predict_and_save_to_xlsx(test_file,model_file, xlsx_result_file)
