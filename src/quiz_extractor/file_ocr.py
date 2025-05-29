import sys
import os

from langchain_community.document_loaders import PyPDFLoader
from tkinter import filedialog, Tk
from tkinter import ttk
from tkinter import messagebox
from utils.utils import Utils
from aliyun_ocr import Sample


class FileOCR:
    def __init__(self) -> None:
        """
        :rtype: None
        """
        # 初始化Tkinter, 不显示主窗口
        self.root = None

        # 初始化文件路径和文件加载器
        self.file_path = None
        self.is_img_pdf = False
        self._initialize()

    def _initialize(self):
        """初始化并加载PDF"""
        try:
            # 初始化Tkinter
            self.root = Tk()
            self.root.withdraw()
            # 选择文件
            self.file_path = filedialog.askopenfilename(
                initialdir="/",
                title="请选择PDF文件",
                filetypes=(("PDF", "*.pdf"), ("All Files", "*.*"))
            )
            if not self.file_path:
                messagebox.showwarning("取消", "未选择文件")
                sys.exit(0)
            # 自动检测PDF类型
            self._detect_pdf_type()
        except Exception as e:
            messagebox.showerror("程序初始化错误", f"初始化失败：{e}")
            self._cleanup()
            sys.exit(1)

    def _detect_pdf_type(self):
        """检测PDF类型是否为图片型"""
        try:
            # 尝试使用PDF加载器
            test_loader = PyPDFLoader(self.file_path)
            test_page = test_loader.load()[0]

            # 如果识别文本过少(可能为图片型PDF)
            if len(test_page.page_content.strip()) < 50:
                self.is_img_pdf = True
                print("识别为图片型pdf")

        except Exception as e:
            print(e)
            self.is_img_pdf = True

    def _ocr_process(self):
        """处理图片类型的PDF"""
        try:

            pic_path = '../../pdf_pics'
            Utils.pdf2img(self.file_path, pic_path)   # 将PDF转换为图片
            pic_files = [pic_path + '/' + f for f in os.listdir(pic_path) if f.lower().endswith(".png")]

            print(pic_files)
            for p, pic_file in enumerate(pic_files):
                Sample.main(sys.argv[1:], pic_file)
                print(f" ==== 正在转换第{p+1}页内容")

        except Exception as e:
            raise Exception(f"OCR处理失败：{e}")

    def get_pages(self):
        """获取处理后的内容"""
        if self.is_img_pdf:
            # 使用ocr处理图片PDF
            self._ocr_process()
            with open('../../output/ocr_text.txt', 'r', encoding='utf-8') as f:
                text = f.read()
                return text
        else:
            # 如果是普通PDF
            pdf_loader = PyPDFLoader(self.file_path)
            return pdf_loader.load_and_split()

    def _cleanup(self):
        """清理资源"""
        if hasattr(self, 'root') and self.root:
            self.root.destroy()


if __name__ == '__main__':
    # 删除文件夹
    if os.path.exists("../../output/ocr_text.txt"):
        os.remove("../../output/ocr_text.txt")
    loader = None
    # 分情况进行处理
    try:
        loader = FileOCR()
        pages = loader.get_pages()
        # 1 如果是图片型PDF
        if loader.is_img_pdf:
            print(pages)
        # 2 如果是文字型PDF
        else:
            for i, page in enumerate(pages):
                print(f"\n=== 第{i}页 ===")
                print(page.page_content[:300] + "...")  # 限制打印长度
                with open("../../output/ocr_text.txt", "a", encoding="utf-8") as f:
                    f.write(page.page_content)

    except Exception as e:
        print(f"发生了错误：{str(e)}")
    finally:
        if loader is not None:
            loader._cleanup()
