import os  
import pdfplumber  
from typing import List, Optional  

class PDFReader:  
    def __init__(self, pdf_path: Optional[str] = None):  
        """  
        PDF 读取器初始化  
        
        :param pdf_path: PDF 文件路径  
        """  
        self.pdf_path = pdf_path  
        self.total_pages = 0  
        self.text_content = []  
    
    def load_pdf(self, pdf_path: Optional[str] = None) -> List[str]:  
        """  
        加载 PDF 文件并提取文本  
        
        :param pdf_path: PDF 文件路径（可选）  
        :return: 每页文本的列表  
        """  
        # 使用传入的路径或初始化时的路径  
        path = pdf_path or self.pdf_path  
        
        # 检查文件是否存在  
        if not path or not os.path.exists(path):  
            raise FileNotFoundError(f"PDF 文件不存在: {path}")  
        
        with pdfplumber.open(path) as pdf:  
            # 记录总页数  
            self.total_pages = len(pdf.pages)  
            
            # 提取每页文本  
            self.text_content = []  
            for page in pdf.pages:  
                # 提取文本，去除多余空白  
                page_text = page.extract_text()  
                if page_text:  
                    self.text_content.append(page_text.strip())  
            
            return self.text_content  
    
    def get_page_text(self, page_number: int) -> Optional[str]:  
        """  
        获取指定页面的文本  
        
        :param page_number: 页码（从1开始）  
        :return: 页面文本或 None  
        """  
        if 1 <= page_number <= len(self.text_content):  
            return self.text_content[page_number - 1]  
        return None  

def main():  
    # PDF 文件路径  
    pdf_path = "./test.pdf"  
    
    # 创建 PDF 读取器  
    pdf_reader = PDFReader(pdf_path)  
    
    # 加载 PDF  
    pages_text = pdf_reader.load_pdf()  
    
    # 打印基本信息  
    print(f"总页数: {pdf_reader.total_pages}")  
    print(f"成功提取文本页数: {len(pages_text)}")  
    
    # 打印第一页文本预览  
    if pages_text:  
        print("\n第一页文本预览:")  
        print(pages_text[0][:500])  # 预览前500个字符  

if __name__ == "__main__":  
    main()  