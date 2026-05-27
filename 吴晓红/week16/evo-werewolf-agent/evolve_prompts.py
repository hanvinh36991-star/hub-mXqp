"""提示词演化脚本

手动触发所有角色提示词的演化过程。
运行此脚本将分析各角色的历史经验，生成优化后的提示词。
"""

import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.evolution_agent import evolve_all_roles


async def main():
    """主函数：演化所有角色提示词"""
    print("开始演化角色提示词...")
    print("=" * 60)
    
    try:
        results = await evolve_all_roles()
        
        print("\n" + "=" * 60)
        print("演化完成！")
        print("=" * 60)
        
        for role_type, prompt in results.items():
            print(f"\n【{role_type}】")
            print(f"提示词长度：{len(prompt)} 字符")
            print(f"保存路径：agent/prompts/{role_type}.txt")
            # 显示前200字符预览
            preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
            print(f"预览：{preview}")
            
    except Exception as e:
        print(f"演化过程中出现错误：{e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)