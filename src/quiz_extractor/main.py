# 主程序
from file_ocr import FileOCR
from extractor import ExamExtractor


def main_process():

    """
    1. 将PDF文件处理为TXT文件
    2. 调用大模型提取试卷内容
    :return:
    """

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
        # 删除pdf转换的图片文件
        # pic_list = [p for p in os.listdir(os.path.abspath("../../pdf_pics")) if p.lower().endswith(".png")]
        # if pic_list:
        #     for p in pic_list:
        #         os.remove(f"../../pdf_pics/{p}")
        # 释放资源
        if loader is not None:
            loader._cleanup()

    # 初始化提取器
    extractor = ExamExtractor()

    try:
        # 处理试卷
        result = extractor.process_exam(
            input_file="../../output/ocr_text.txt",
            output_file="../../output/ocr_result.json"
        )
        print("\n试卷处理完成！")
    except Exception as e:
        print(f"\n处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_process()
