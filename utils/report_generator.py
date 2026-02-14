class ContentProcessor:
    """
    报告内容处理器，用于处理和优化报告内容
    """
    
    def __init__(self):
        """初始化内容处理器"""
        self.chinese = True  # 默认使用中文
    
    def check_for_duplicate_content(self, content: str) -> list:
        """
        检查内容中是否有重复的段落
        
        Args:
            content: 要检查的内容
            
        Returns:
            list: 重复段落的列表
        """
        # 简单的重复检查实现
        paragraphs = content.split('\n\n')
        seen = set()
        duplicates = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                if para in seen:
                    duplicates.append((i, para))
                else:
                    seen.add(para)
        
        return duplicates
    
    def remove_duplicate_content(self, content: str) -> str:
        """
        移除内容中的重复段落
        
        Args:
            content: 要处理的内容
            
        Returns:
            str: 移除重复后的内容
        """
        paragraphs = content.split('\n\n')
        seen = set()
        result = []
        
        for para in paragraphs:
            if para.strip():
                if para not in seen:
                    seen.add(para)
                    result.append(para)
            else:
                result.append(para)
        
        return '\n\n'.join(result)
    
    def format_tables(self, content: str) -> str:
        """
        格式化内容中的表格
        
        Args:
            content: 要处理的内容
            
        Returns:
            str: 格式化后的内容
        """
        # 简单的表格格式化实现
        # 这里可以根据实际需要扩展表格格式化逻辑
        lines = content.split('\n')
        formatted_lines = []
        
        in_table = False
        table_lines = []
        
        for line in lines:
            # 检测表格行（包含|的行）
            if '|' in line:
                if not in_table:
                    in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    # 处理表格
                    formatted_lines.extend(self._format_table(table_lines))
                    in_table = False
                    table_lines = []
                formatted_lines.append(line)
        
        # 处理最后一个表格
        if in_table:
            formatted_lines.extend(self._format_table(table_lines))
        
        return '\n'.join(formatted_lines)
    
    def _format_table(self, table_lines: list) -> list:
        """
        格式化单个表格
        
        Args:
            table_lines: 表格的行列表
            
        Returns:
            list: 格式化后的表格行列表
        """
        # 简单的表格格式化，确保列对齐
        if not table_lines:
            return table_lines
        
        # 计算每列的最大宽度
        columns = []
        for line in table_lines:
            if '|' in line:
                cols = [col.strip() for col in line.split('|')]
                if len(cols) > len(columns):
                    columns.extend([0] * (len(cols) - len(columns)))
                for i, col in enumerate(cols):
                    if len(col) > columns[i]:
                        columns[i] = len(col)
        
        # 格式化每一行
        formatted_lines = []
        for line in table_lines:
            if '|' in line:
                cols = [col.strip() for col in line.split('|')]
                formatted_cols = []
                for i, col in enumerate(cols):
                    if i < len(columns):
                        # 左对齐，添加填充
                        formatted_cols.append(col.ljust(columns[i]))
                    else:
                        formatted_cols.append(col)
                formatted_line = ' | '.join(formatted_cols)
                formatted_lines.append(formatted_line)
            else:
                formatted_lines.append(line)
        
        return formatted_lines
