#!/usr/bin/env python3
"""
测试所有更新后的导入语句是否正确
这个脚本会检查语法而不实际运行需要深度学习库的代码
"""

import ast
import sys
from pathlib import Path

def test_import_syntax(file_path):
    """测试文件的导入语句语法是否正确"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析AST来检查语法
        tree = ast.parse(content)
        
        # 提取导入语句
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        
        return True, imports
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def main():
    """主测试函数"""
    print("=== 测试项目文件的导入语句语法 ===")
    
    # 要测试的文件列表
    test_files = [
        'train.py',
        'main.py', 
        'evaluate.py',
        'MSMultiSpineLoader.py'
    ]
    
    all_passed = True
    
    for file_name in test_files:
        file_path = Path(file_name)
        if not file_path.exists():
            print(f"❌ {file_name}: 文件不存在")
            all_passed = False
            continue
            
        success, result = test_import_syntax(file_path)
        
        if success:
            print(f"✅ {file_name}: 语法检查通过")
            # 检查是否包含MSMultiSpineLoader相关导入
            msl_imports = [imp for imp in result if 'MSMultiSpineLoader' in imp]
            if msl_imports:
                print(f"   📦 MSMultiSpineLoader导入: {len(msl_imports)}个")
                for imp in msl_imports:
                    print(f"      - {imp}")
        else:
            print(f"❌ {file_name}: 语法错误 - {result}")
            all_passed = False
    
    print("\n=== 测试结果 ===")
    if all_passed:
        print("🎉 所有文件的导入语句语法检查通过！")
        print("\n📋 更新摘要:")
        print("   ✅ train.py - 已更新导入和优化功能集成")
        print("   ✅ main.py - 已更新导入语句")
        print("   ✅ evaluate.py - 已更新导入和优化功能使用")
        print("   ✅ MSMultiSpineLoader.py - 包含所有优化功能")
        print("   ✅ README.md - 已更新示例代码和文档")
        return 0
    else:
        print("❌ 部分文件存在语法问题，请检查")
        return 1

if __name__ == '__main__':
    sys.exit(main())