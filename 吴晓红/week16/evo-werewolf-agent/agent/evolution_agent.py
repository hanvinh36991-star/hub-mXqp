"""演化代理

在游戏结束后，分析玩家表现，优化角色提示词，实现自我改进。
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from agents import Agent, Runner
from agents import set_default_openai_api, set_tracing_disabled
from schema.system_config import load_system_config

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

config = load_system_config("config/system_config.json")

# 提示词模板存储目录
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
os.makedirs(PROMPTS_DIR, exist_ok=True)


class EvolutionAgent:
    """演化代理
    
    负责分析游戏经验，生成优化后的角色提示词，实现智能体的自我演化。
    """

    def __init__(self):
        self.agent = Agent(
            name="EvolutionAgent",
            model=config.default_model,
            instructions="你是一个狼人杀游戏策略优化专家。你的任务是分析玩家的游戏表现，生成改进后的角色提示词，帮助智能体在未来游戏中表现得更好。",
        )

    async def evolve_prompt(self, role_type: str, experiences: List[Dict[str, Any]]) -> str:
        """为指定角色类型生成优化后的提示词
        
        Args:
            role_type: 角色类型，如 "werewolf", "seer"
            experiences: 该角色的经验列表，每条包含 summary, strategies, mistakes, lessons 等
            
        Returns:
            优化后的提示词文本
        """
        if not experiences:
            return ""
            
        # 提取最近5条经验
        recent = experiences[-5:]
        
        # 分析经验，找出常见错误和有效策略
        analysis = self._analyze_experiences(recent)
        
        prompt = self._build_evolution_prompt(role_type, recent, analysis)
        result = await Runner.run(self.agent, prompt)
        return self._extract_prompt(result.final_output)
    
    def _analyze_experiences(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析经验数据，提取关键模式"""
        total = len(experiences)
        wins = sum(1 for exp in experiences if exp.get("is_winner", False))
        win_rate = wins / total if total > 0 else 0
        
        common_mistakes = []
        effective_strategies = []
        
        for exp in experiences:
            if exp.get("mistakes"):
                common_mistakes.append(exp["mistakes"])
            if exp.get("strategies"):
                effective_strategies.append(exp["strategies"])
        
        return {
            "total_games": total,
            "wins": wins,
            "win_rate": win_rate,
            "common_mistakes": common_mistakes[:3],  # 取前3个
            "effective_strategies": effective_strategies[:3],
        }
    
    def _build_evolution_prompt(self, role_type: str, experiences: List[Dict[str, Any]], 
                               analysis: Dict[str, Any]) -> str:
        """构建演化提示词"""
        role_names = {
            "werewolf": "狼人",
            "seer": "预言家", 
            "witch": "女巫",
            "hunter": "猎人",
            "villager": "村民",
        }
        role_name = role_names.get(role_type, role_type)
        
        # 格式化经验
        exp_text = ""
        for i, exp in enumerate(experiences, 1):
            outcome = "胜利" if exp.get("is_winner", False) else "失败"
            exp_text += f"\n--- 经验{i} ({outcome}) ---\n"
            exp_text += f"总结：{exp.get('summary', '无')}\n"
            exp_text += f"策略：{exp.get('strategies', '无')}\n"
            exp_text += f"错误：{exp.get('mistakes', '无')}\n"
            exp_text += f"建议：{exp.get('lessons', '无')}\n"
        
        return f"""你正在优化狼人杀游戏中{role_name}角色的AI提示词。

## 角色背景
{role_name}是狼人杀游戏中的一个角色，拥有特定的能力和目标。

## 历史表现分析
- 总对局数：{analysis['total_games']}
- 胜利次数：{analysis['wins']}
- 胜率：{analysis['win_rate']:.1%}
- 常见错误：{analysis['common_mistakes']}
- 有效策略：{analysis['effective_strategies']}

## 详细经验记录
{exp_text}

## 任务
基于以上分析，请生成一个**优化后的提示词**，用于指导{role_name}AI在游戏中的决策。
这个提示词将替换原有的提示词，帮助AI在未来游戏中避免常见错误、采用有效策略。

## 提示词要求
1. 保持原有提示词的基本结构：角色介绍、游戏规则、胜利条件
2. 融入从经验中学到的关键教训
3. 强调成功的策略
4. 警告常见的陷阱
5. 使用清晰、具体的指导语言
6. 长度适中（300-500字）

## 输出格式
请只输出优化后的提示词文本，不要包含任何额外的解释或标记。
提示词应以第一人称编写（如"你是一个狼人杀游戏中的玩家。你的角色是：{role_name}..."）。
"""
    
    def _extract_prompt(self, output: str) -> str:
        """从LLM输出中提取提示词"""
        # 去除可能的代码块标记
        output = re.sub(r'```(?:json|text)?\n?', '', output)
        output = re.sub(r'\n*```', '', output)
        return output.strip()
    
    def save_prompt(self, role_type: str, prompt: str) -> None:
        """保存优化后的提示词到文件"""
        filepath = os.path.join(PROMPTS_DIR, f"{role_type}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt)
    
    def load_prompt(self, role_type: str) -> Optional[str]:
        """加载优化后的提示词"""
        filepath = os.path.join(PROMPTS_DIR, f"{role_type}.txt")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        return None


async def evolve_all_roles() -> Dict[str, str]:
    """演化所有角色类型的提示词
    
    Returns:
        字典：角色类型 -> 新提示词
    """
    from memory.experience import load_experiences
    
    evolution_agent = EvolutionAgent()
    results = {}
    
    role_types = ["werewolf", "seer", "witch", "hunter", "villager"]
    
    for role_type in role_types:
        experiences = load_experiences(role_type)
        if experiences:
            new_prompt = await evolution_agent.evolve_prompt(role_type, experiences)
            evolution_agent.save_prompt(role_type, new_prompt)
            results[role_type] = new_prompt
            print(f"角色 {role_type} 的提示词已演化并保存")
        else:
            print(f"角色 {role_type} 尚无经验，跳过演化")
    
    return results


__all__ = ["EvolutionAgent", "evolve_all_roles"]