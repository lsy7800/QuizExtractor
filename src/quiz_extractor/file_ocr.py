import sys

from langchain_community.document_loaders import PyPDFLoader
from tkinter import filedialog, Tk
from tkinter import ttk
from tkinter import messagebox


class FileOCR:
    def __init__(self):
        # 初始化Tkinter, 不显示主窗口
        self.root = Tk()
        self.root.withdraw()

        # 初始化文件路径和文件加载器
        self.file_path = None
        self.pdf_loader = None

        # 运行内部方法选择路径
        self._select_file()

        # 如果路径有效, 初始化加载器
        if self.file_path:
            try:
                self.pdf_loader = PyPDFLoader(self.file_path)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                sys.exit(1)

    def _select_file(self):
        """选择PDF文件"""
        self.file_path = filedialog.askopenfilename(
            initialdir="/",
            title="Select file",
            filetypes=(("PDF files", "*.pdf"),)
        )
        if not self.file_path:
            messagebox.showerror("Error", "No file selected")
            sys.exit(0)

    def get_pages(self):
        """加载并分隔PDF页面"""
        if not self.pdf_loader:
            raise ValueError("PDF loader not initialized")

        try:
            return self.pdf_loader.load_and_split()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            sys.exit(1)


if __name__ == '__main__':
    try:
        file_ocr = FileOCR()
        pages = file_ocr.get_pages()
        for index, page in enumerate(pages, start=1):
            print(f"\n=== Page {index} ===")
            print(page.page_content[:200] + "...")
    except Exception as e:
        print(e)

