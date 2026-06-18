import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import json
import os
from datetime import datetime
import matplotlib as mpl


class ReportVisualizer:
    def __init__(self):
        """初始化报告可视化工具"""
        # 配置matplotlib中文字体
        self._setup_chinese_font()

        self.dimensions = {
            'technical_knowledge': '技术知识',
            'problem_solving': '问题解决',
            'communication': '沟通表达',
            'team_collaboration': '团队协作',
            'code_quality': '代码质量',
            'system_design': '系统设计'
        }

    def _setup_chinese_font(self):
        """配置中文字体支持"""
        try:
            # 尝试使用系统中文字体
            if os.name == 'nt':  # Windows系统
                font_paths = [
                    'C:/Windows/Fonts/simhei.ttf',  # 黑体
                    'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
                    'C:/Windows/Fonts/simsun.ttc'  # 宋体
                ]
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        plt.rcParams['font.family'] = ['sans-serif']
                        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
                        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                        break
            else:  # Linux/Mac系统
                plt.rcParams['font.family'] = ['sans-serif']
                plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
                plt.rcParams['axes.unicode_minus'] = False

            # 验证字体是否生效
            mpl.font_manager.findfont('SimHei', rebuild_if_missing=True)

        except Exception as e:
            print(f"配置中文字体时出现警告: {str(e)}")
            print("将使用默认字体，中文显示可能不正确")

    def generate_radar_chart(self, scores: Dict[str, float], output_path: str):
        """生成雷达图
        
        Args:
            scores: 各维度得分字典
            output_path: 输出文件路径
        """
        try:
            # 准备数据
            categories = list(self.dimensions.values())
            values = [scores[dim] for dim in self.dimensions.keys()]

            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            values = np.concatenate((values, [values[0]]))  # 闭合图形
            angles = np.concatenate((angles, [angles[0]]))  # 闭合图形

            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)

            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)
            ax.set_ylim(0, 100)

            # 添加标题
            plt.title('面试评估雷达图', pad=20, fontsize=14)

            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.7)

            # 保存图形
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"生成雷达图时出现错误: {str(e)}")
            # 创建一个简单的错误提示图
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.text(0.5, 0.5, '生成雷达图时出现错误\n请检查数据格式',
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path)
            plt.close()

    def generate_html_report(self, interview_data: Dict, output_path: str):
        """生成HTML格式的面试报告
        
        Args:
            interview_data: 面试数据
            output_path: 输出文件路径
        """
        # 生成雷达图
        radar_chart_path = os.path.join(os.path.dirname(output_path), 'radar_chart.png')
        self.generate_radar_chart(interview_data['final_scores'], radar_chart_path)
        print("-----------雷达图已生成------------")
        # HTML模板
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>面试评估报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .section {{ margin-bottom: 30px; }}
        .score-item {{ margin: 10px 0; }}
        .question-item {{ margin: 20px 0; padding: 10px; background: #f5f5f5; }}
        .radar-chart {{ text-align: center; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; border: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>面试评估报告</h1>
    
    <div class="section">
        <h2>基本信息</h2>
        <p>面试职位：{position}</p>
        <p>面试级别：{level}</p>
        <p>面试时间：{start_time} - {end_time}</p>
        <p>总轮次：{total_rounds}</p>
    </div>
    
    <div class="section">
        <h2>综合评分</h2>
        <div class="radar-chart">
            <img src="radar_chart.png" alt="评估雷达图">
        </div>
        <table>
            <tr>
                <th>评估维度</th>
                <th>得分</th>
            </tr>
            {score_rows}
        </table>
    </div>
    
    <div class="section">
        <h2>详细问答记录</h2>
        {question_items}
    </div>
</body>
</html>
"""

        # 生成评分行
        score_rows = ""
        for dim, score in interview_data['final_scores'].items():
            score_rows += f"""
            <tr>
                <td>{self.dimensions[dim]}</td>
                <td>{score:.1f}</td>
            </tr>
            """

        # 生成问题记录
        question_items = ""
        for q in interview_data['questions']:
            question_items += f"""
            <div class="question-item">
                <h3>问题 {q['round']}</h3>
                <p><strong>问题：</strong>{q['question']}</p>
                <p><strong>回答：</strong>{q.get('answer', '无回答')}</p>
                <p><strong>得分：</strong>{q['score']['final_score']:.1f}</p>
                <p><strong>评估：</strong>{q['score']['details']['content_score']['reason']}</p>
            </div>
            """

        # 填充模板
        html_content = html_template.format(
            position=interview_data['position'],
            level=interview_data['level'],
            start_time=interview_data['start_time'],
            end_time=interview_data['end_time'] or '未完成',
            total_rounds=interview_data.get('total_rounds', len(interview_data['questions'])),
            score_rows=score_rows,
            question_items=question_items
        )

        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def calculate_dimension_scores(self, questions: List[Dict]) -> Dict[str, float]:
        """计算各维度的最终得分
        
        Args:
            questions: 问题列表
            
        Returns:
            Dict[str, float]: 各维度得分
        """
        # 初始化维度得分
        dimension_scores = {dim: [] for dim in self.dimensions.keys()}

        # 根据问题类型和内容分配得分
        for q in questions:
            score = q['score']['final_score']
            q_type = q['type']

            if q_type == 'tech':
                # 技术问题影响技术知识、问题解决、代码质量和系统设计
                dimension_scores['technical_knowledge'].append(score)
                dimension_scores['problem_solving'].append(score)
                dimension_scores['code_quality'].append(score)
                dimension_scores['system_design'].append(score)
            else:
                # 行为问题影响沟通表达和团队协作
                dimension_scores['communication'].append(score)
                dimension_scores['team_collaboration'].append(score)

        # 计算每个维度的平均分
        final_scores = {}
        for dim, scores in dimension_scores.items():
            if scores:
                final_scores[dim] = sum(scores) / len(scores)
            else:
                final_scores[dim] = 0.0

        return final_scores


def test_report_visualizer():
    """测试报告可视化功能"""
    print("\n=== 开始测试报告可视化功能 ===")

    # 初始化可视化工具
    visualizer = ReportVisualizer()

    # 测试数据
    test_data = {
        'position': 'Java开发工程师',
        'level': '高级',
        'start_time': '2024-03-20 10:00:00',
        'end_time': '2024-03-20 11:00:00',
        'total_rounds': 5,
        'questions': [
            {
                'round': 1,
                'type': 'tech',
                'question': '请详细说明MyBatis的工作原理',
                'answer': 'MyBatis是一个优秀的持久层框架...',
                'score': {
                    'final_score': 85.0,
                    'details': {
                        'content_score': {
                            'score': 85.0,
                            'reason': '回答全面，技术准确'
                        }
                    }
                }
            },
            {
                'round': 2,
                'type': 'behavior',
                'question': '请描述一次团队协作经历',
                'answer': '在之前的项目中...',
                'score': {
                    'final_score': 90.0,
                    'details': {
                        'content_score': {
                            'score': 90.0,
                            'reason': '表达清晰，案例具体'
                        }
                    }
                }
            }
        ]
    }

    # 计算维度得分
    test_data['final_scores'] = visualizer.calculate_dimension_scores(test_data['questions'])

    # 生成报告
    output_dir = 'interview_reports'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'interview_report_{timestamp}.html')

    visualizer.generate_html_report(test_data, report_path)
    print(f"报告已生成: {report_path}")


if __name__ == "__main__":
    test_report_visualizer()
