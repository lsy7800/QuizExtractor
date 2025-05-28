import sys

from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path
from tkinter import filedialog, Tk
from tkinter import ttk
from tkinter import messagebox


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
            # 将PDF转换为图片
            images = convert_from_path(self.file_path, dpi=300)
            print(images)

            breakpoint()
            # 识别结果存储
            all_text = []

            for i, image in enumerate(images):
                # 调用阿里云小语种识别接口进行内容识别
                pass

            return "\n\n".join(all_text)

        except Exception as e:
            raise Exception(f"OCR处理失败：{e}")

    def get_pages(self):
        """获取处理后的内容"""
        if self.is_img_pdf:
            # 使用ocr处理图片PDF
            ocr_text = self._ocr_process()
            return [Document(page_content=ocr_text)]
        else:
            # 如果是普通PDF
            pdf_loader = PyPDFLoader(self.file_path)
            return pdf_loader.load_and_split()

    def _cleanup(self):
        """清理资源"""
        if hasattr(self, 'root') and self.root:
            self.root.destroy()


if __name__ == '__main__':
    loader = None
    try:
        loader = FileOCR()
        pages = loader.get_pages()

        for i, page in enumerate(pages):
            print(f"\n=== 第{i}页 ===")
            print(page.page_content[:300] + "...")  # 限制打印长度

    except Exception as e:
        print(f"发生了错误：{str(e)}")
    finally:
        if loader is not None:
            loader._cleanup()
