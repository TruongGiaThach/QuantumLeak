import os
import nbformat as nbf
import sys

# --- Cấu hình ---
PROJECT_ROOT = '.'  # Thư mục gốc của dự án
NOTEBOOK_FILENAME = 'QuantumLeak_Notebook.ipynb'
MAIN_FILE = 'main.py'

# ==============================================================================
# BIẾN TÙY CHỈNH: Liệt kê các thư mục bạn muốn BỎ QUA khi tạo notebook.
# Thêm tên thư mục vào danh sách này.
# Ví dụ: 'venv', '.git', 'data', '__pycache__'
# ==============================================================================
EXCLUDE_DIRS = [
    '__pycache__',
    '.git',
    '.idea',
    '.vscode',
    'venv',
    'env',
    'data',  # Thường không cần đưa dữ liệu vào notebook, trừ khi nó nhỏ
    # Thêm thư mục môi trường Python của bạn vào đây để tránh lỗi
    'Python311' 
]
# ==============================================================================


# Tạo một notebook mới
notebook = nbf.v4.new_notebook()
notebook['cells'] = []

# --- Cell 1: Giới thiệu và Cài đặt thư viện ---
intro_text = """
# QuantumLeak Project Notebook

Đây là phiên bản Jupyter Notebook được tạo tự động từ dự án Python gốc.
Notebook này chứa tất cả mã nguồn cần thiết và được cấu trúc để chạy trong các môi trường như Google Colab.

**Hướng dẫn:**
1.  Chạy các cell theo thứ tự từ trên xuống dưới.
2.  Cell đầu tiên sẽ cài đặt tất cả các thư viện cần thiết.
3.  Các cell tiếp theo sẽ sử dụng "magic command" `%%writefile` để tái tạo lại cấu trúc file của dự án.
4.  Cell cuối cùng chứa mã từ `main.py` để bạn có thể chạy các thử nghiệm.
"""
notebook['cells'].append(nbf.v4.new_markdown_cell(intro_text))

dependencies = [
    "pennylane",
    "pennylane-lightning",
    "torch",
    "torchvision",
    "pandas",
    "numpy",
    "scikit-learn",
    "tqdm",
    "joblib",
    "matplotlib"
]
install_code = f"import sys\n!{{sys.executable}} -m pip install {' '.join(dependencies)}"
notebook['cells'].append(nbf.v4.new_code_cell(install_code, metadata={"colab_type": "code"}))


# --- Cell 2: Tái tạo cấu trúc file ---
file_structure_header = """
## 1. Tái tạo cấu trúc file của dự án

Các cell dưới đây sẽ tạo ra các thư mục và file `.py` cần thiết cho dự án.
"""
notebook['cells'].append(nbf.v4.new_markdown_cell(file_structure_header))

def read_file_with_fallback(file_path):
    """Đọc file với các encoding dự phòng."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        print(f"Cảnh báo: File '{file_path}' không phải là UTF-8. Thử đọc với 'latin-1'.")
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Lỗi nghiêm trọng: Không thể đọc file '{file_path}'. Lỗi: {e}")
            return None

# Tìm tất cả các file .py trong dự án
files_to_write = []
for root, dirs, files in os.walk(PROJECT_ROOT, topdown=True):
    # Loại bỏ các thư mục không mong muốn từ việc duyệt tiếp
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    
    for file in files:
        if file.endswith('.py') and file not in ['create_notebook.py', MAIN_FILE]:
            files_to_write.append(os.path.join(root, file))

# Sắp xếp để đảm bảo các file config được ghi trước
files_to_write.sort()

# Tạo các cell để ghi file
created_dirs = set()
for file_path in files_to_write:
    normalized_path = file_path.replace('\\', '/')
    
    dir_name = os.path.dirname(normalized_path)
    if dir_name and dir_name != '.' and dir_name not in created_dirs:
        notebook['cells'].append(nbf.v4.new_code_cell(f"!mkdir -p {dir_name}"))
        created_dirs.add(dir_name)
        
    file_content = read_file_with_fallback(file_path)
    if file_content is None:
        continue
    
    write_cell_content = f"%%writefile {normalized_path}\n\n{file_content}"
    notebook['cells'].append(nbf.v4.new_code_cell(write_cell_content))

# Thêm file main.py cuối cùng
main_content = read_file_with_fallback(os.path.join(PROJECT_ROOT, MAIN_FILE))
if main_content:
    write_cell_content = f"%%writefile {MAIN_FILE}\n\n{main_content}"
    notebook['cells'].append(nbf.v4.new_code_cell(write_cell_content))


# --- Cell 3: Chạy thử nghiệm ---
run_header = """
## 2. Chạy các thử nghiệm

Cell dưới đây cho phép bạn chạy các thử nghiệm chính của dự án.
Bỏ comment (xóa dấu `#`) ở dòng tương ứng với thử nghiệm bạn muốn chạy.

**Lưu ý:**
- Một bản vá lỗi cho `QuantumLayer.forward` đã được áp dụng tự động để tránh lỗi `ValueError` khi chuẩn hóa vector.
- Bạn có thể chạy lại cell này nhiều lần để thực hiện các thử nghiệm khác nhau mà không cần chạy lại các cell phía trên.
"""
notebook['cells'].append(nbf.v4.new_markdown_cell(run_header))

if main_content:
    final_run_code = f"""
# Import các hàm từ file main.py vừa được tạo
from main import (
    run_quanv_experiment,
    run_basic_qnn_experiment,
    run_circuit14_experiment,
    run_transfer_learning_experiment,
    run_leak_experiment
)

# --- BẢN VÁ LỖI TỰ ĐỘNG ---
# Áp dụng bản vá cho lỗi ValueError khi chuẩn hóa vector trong QuantumLayer
try:
    import torch
    from models.circuit14 import QuantumLayer

    def forward_patched(self, inputs):
        x = torch.relu(self.conv(inputs))
        x = x.view(-1, 16 * 15 * 15)
        x = torch.tanh(self.pre_net(x))
        # Chuẩn hóa để tránh lỗi chia cho 0
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)
        
        batch_size = x.size(0)
        outputs = []
        for i in range(batch_size):
            q_out = self.quantum_circuit(x[i], self.weights, self.crx_weights)
            q_out = torch.stack(q_out).float()
            outputs.append(q_out)
        outputs = torch.stack(outputs)
        probs = self.fc(outputs).sigmoid()
        return probs

    QuantumLayer.forward = forward_patched
    print("✅ Đã áp dụng bản vá cho QuantumLayer.forward để sửa lỗi ValueError.")
except ImportError:
    print("⚠️ Không tìm thấy QuantumLayer, bỏ qua việc vá lỗi.")
except Exception as e:
    print(f"❌ Lỗi khi áp dụng bản vá: {{e}}")


# --- CHỌN THỬ NGHIỆM ĐỂ CHẠY ---
# Bỏ comment (xóa #) ở dòng bạn muốn thực thi.
# Chỉ nên chạy một thử nghiệm mỗi lần để dễ theo dõi.

print("\\n--- Bắt đầu chạy thử nghiệm Circuit14 ---")
run_circuit14_experiment()

# print("\\n--- Bắt đầu chạy thử nghiệm Transfer Learning ---")
# run_transfer_learning_experiment()

# print("\\n--- Bắt đầu chạy thử nghiệm Quanv ---")
# run_quanv_experiment()

# print("\\n--- Bắt đầu chạy thử nghiệm Basic QNN ---")
# run_basic_qnn_experiment()

# print("\\n--- Bắt đầu chạy thử nghiệm Leak trên Basic QNN ---")
# run_leak_experiment("basic_qnn")

# print("\\n--- Bắt đầu chạy thử nghiệm Leak trên Circuit14 ---")
# run_leak_experiment("circuit14")

# print("\\n--- Bắt đầu chạy thử nghiệm Leak trên Transfer Learning ---")
# run_leak_experiment("transfer_learning")
"""
    notebook['cells'].append(nbf.v4.new_code_cell(final_run_code))


# --- Lưu notebook ---
try:
    with open(NOTEBOOK_FILENAME, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    print(f"\n✅ Tạo notebook thành công! File đã được lưu tại: {NOTEBOOK_FILENAME}")
except Exception as e:
    print(f"\n❌ Lỗi khi ghi file notebook: {e}")