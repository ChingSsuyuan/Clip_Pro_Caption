import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

def draw_architecture():
    """绘制架构图"""
    
    fig, ax = plt.subplots(figsize=(14, 18), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # 颜色方案
    color_frozen = '#E1F5FE'      # CLIP (冻结)
    color_trainable = '#FFF3E0'   # 可训练模块
    color_data = '#F3E5F5'        # 数据流
    color_process = '#E8F5E9'     # 处理过程
    
    # ==================== 辅助函数 ====================
    def add_box(x, y, w, h, text, color, edge_color='#333', lw=2, fontsize=10):
        """添加圆角矩形框"""
        box = FancyBboxPatch(
            (x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=lw,
            zorder=2
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', 
                fontsize=fontsize, fontweight='bold', zorder=3)
        return box
    
    def add_arrow(x1, y1, x2, y2, label='', color='#333', lw=2):
        """添加箭头"""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.4,head_length=0.8',
            color=color,
            linewidth=lw,
            zorder=1
        )
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax.text(mid_x+0.3, mid_y, label, fontsize=8, 
                   style='italic', color=color)
    
    # ==================== 绘制流程图 ====================
    
    # 2. CLIP 编码器区域 - 调整背景框包含所有组件（不包括Input Image）
    clip_bg = FancyBboxPatch(
        (0.5, 13.5), 9, 5,
        boxstyle="round,pad=0.15",
        facecolor=color_frozen,
        edgecolor='#01579B',
        linewidth=3,
        alpha=0.3,
        zorder=0
    )
    ax.add_patch(clip_bg)
    ax.text(5, 18, 'CLIP Encoder', 
           ha='center', fontsize=11, fontweight='bold', color='#01579B')
    
    # 1. 输入图像 - 放在CLIP Encoder背景框之上
    add_box(5, 19.5, 2, 0.8, 'Input Image\n[B, 3, 224, 224]', 
            '#E3F2FD', '#1976D2', 2, 9)
    
    # Vision Encoder
    add_box(3, 16.5, 2, 0.8, 'ResNet-50x4\nVision Encoder', 
            '#B3E5FC', '#01579B', 3, 9)
    
    # Text Encoder
    add_box(7, 16.5, 2, 0.8, 'Text Encoder', 
            '#B3E5FC', '#01579B', 3, 9)
    
    # Weather Labels
    add_box(7, 15.3, 2.5, 0.6, 
            '[sunny, rainy, cloudy, snowy, foggy, stormy, overcast, clear]',  
            '#B3E5FC', '#01579B', 2, 7)
    
    # 箭头: Input → Vision Encoder (完全连接)
    add_arrow(5, 19.1, 3, 17.3, color='#1976D2', lw=2.5)
    # 箭头: Input → Text Encoder (通过天气标签，完全连接)
    add_arrow(5, 19.1, 7, 17.3, color='#1976D2', lw=2.5, label='weather labels')
    
    # 3. 特征输出
    add_box(3, 14.5, 2, 0.6, 'Image Features\n[B, 640]', 
            color_data, '#6A1B9A', 2, 9)
    add_box(7, 14.5, 2, 0.6, 'Text Features\n[8, 640]', 
            color_data, '#6A1B9A', 2, 9)
    
    add_arrow(3, 16.1, 3, 14.9, color='#01579B', lw=2)
    add_arrow(7, 16.1, 7, 14.9, color='#01579B', lw=2)
    add_arrow(7, 14.9, 7, 14.5, color='#01579B', lw=2)
    
    # 4. Zero-shot 分类区域 - 与Vision Mapping Network平齐，相同高度
    zs_bg = FancyBboxPatch(
        (5.3, 9.5), 4.4, 4.2,
        boxstyle="round,pad=0.15",
        facecolor=color_process,
        edgecolor='#33691E',
        linewidth=3,
        alpha=0.3,
        zorder=0
    )
    ax.add_patch(zs_bg)
    ax.text(7.5, 13.8, ' Zero-shot Classification', 
           ha='center', fontsize=10, fontweight='bold', color='#33691E')
    
    add_box(7.5, 13, 2, 0.5, 'Cosine Similarity\n[B, 8]', 
            '#C5E1A5', '#33691E', 2, 8)
    add_box(7.5, 12.2, 1.5, 0.5, 'Softmax', 
            '#C5E1A5', '#33691E', 2, 8)
    add_box(7.5, 11.4, 2, 0.5, 'Weather Pred\n[B]', 
            '#C5E1A5', '#33691E', 2, 8)
    
    # 箭头: 特征 → 相似度
    add_arrow(3.5, 14.2, 6.5, 13.2, color='#6A1B9A', lw=2)
    add_arrow(7.5, 14.2, 7.5, 13.3, color='#6A1B9A', lw=2)
    add_arrow(7.5, 12.7, 7.5, 12.5, color='#33691E', lw=2)
    add_arrow(7.5, 11.9, 7.5, 11.7, color='#33691E', lw=2)
    
    # 5. Weather Embedding - 作为Zero-shot的输出，在框底部
    add_box(7.5, 10.2, 2.2, 0.6, 'Weather Embedding\n[B, 1, 768]', 
            color_trainable, '#E65100', 3, 9)
    
    # 箭头: Weather Pred → Weather Embedding
    add_arrow(7.5, 11.1, 7.5, 10.5, color='#33691E', lw=2)
    
    # 6. Vision Mapping Network - 调整背景框包含所有四个block
    map_bg = FancyBboxPatch(
        (0.3, 9.5), 4.4, 4.2,
        boxstyle="round,pad=0.15",
        facecolor=color_trainable,
        edgecolor='#E65100',
        linewidth=3,
        alpha=0.3,
        zorder=0
    )
    ax.add_patch(map_bg)
    ax.text(2.5, 13.8, 'Vision Mapping Network', 
           ha='center', fontsize=10, fontweight='bold', color='#E65100')
    
    add_box(2.5, 13, 2.5, 0.6, 'Linear Projection\n[640]→[768x40]', 
            '#FFCC80', '#E65100', 3, 8)
    add_box(2.5, 12, 2.5, 0.6, 'Transformer\n8 Layers', 
            '#FFCC80', '#E65100', 3, 9)
    add_box(2.5, 11, 2.5, 0.6, 'Reshape\n[B, 40, 768]', 
            '#FFCC80', '#E65100', 3, 9)
    add_box(2.5, 10, 2.5, 0.6, 'Vision Prefix\n[B, 40, 768]', 
            color_data, '#6A1B9A', 2, 9)
    
    # 箭头: Image Features → Mapping
    add_arrow(3, 14.2, 2.5, 13.3, color='#6A1B9A', lw=2)
    add_arrow(2.5, 12.7, 2.5, 12.3, color='#E65100', lw=2)
    add_arrow(2.5, 11.7, 2.5, 11.3, color='#E65100', lw=2)
    add_arrow(2.5, 10.7, 2.5, 10.3, color='#E65100', lw=2)
    
    # 7. 拼接 Weather + Vision
    add_box(5, 9, 2.5, 0.7, 'Concatenate\n[B, 41, 768]', 
            '#FFF9C4', '#F57F17', 2, 9)
    add_arrow(3.8, 10, 4, 9.4, color='#6A1B9A', lw=2)
    add_arrow(6.3, 10.2, 6, 9.4, color='#E65100', lw=2)
    
    # 8. 文本输入
    add_box(8, 7.5, 2, 0.6, 'Text Tokens\n[B, seq_len]', 
            '#E3F2FD', '#1976D2', 2, 9)
    add_box(8, 6.5, 2.3, 0.6, 'GPT-2 Embedding\n[B, seq_len, 768]', 
            color_data, '#6A1B9A', 2, 9)
    add_arrow(8, 7.2, 8, 6.8, color='#1976D2', lw=2)
    
    # 9. 最终拼接
    add_box(5, 6.5, 3, 0.7, 'Final Concatenate\n[B, 41+seq_len, 768]', 
            '#FFF9C4', '#F57F17', 2, 9)
    add_arrow(5, 8.7, 5, 6.85, color='#F57F17', lw=2.5)
    add_arrow(6.85, 6.5, 6.5, 6.5, color='#6A1B9A', lw=2)
    
    # 10. GPT-2 解码器
    gpt2_bg = FancyBboxPatch(
        (3, 3.5), 4, 2.3,
        boxstyle="round,pad=0.15",
        facecolor=color_trainable,
        edgecolor='#E65100',
        linewidth=3,
        alpha=0.3,
        zorder=0
    )
    ax.add_patch(gpt2_bg)
    ax.text(5, 5.9, 'GPT-2 Decoder (Fine-tuned)', 
           ha='center', fontsize=10, fontweight='bold', color='#E65100')
    
    add_box(5, 5, 2.5, 0.6, 'Transformer Blocks\n12 Layers', 
            '#FFCC80', '#E65100', 3, 9)
    add_box(5, 4, 2.5, 0.6, 'LM Head', 
            '#FFCC80', '#E65100', 3, 9)
    
    add_arrow(5, 6.15, 5, 5.3, color='#F57F17', lw=2.5)
    add_arrow(5, 4.7, 5, 4.3, color='#E65100', lw=2)
    
    # 11. 输出
    add_box(5, 2.5, 2.5, 0.6, 'Logits\n[B, 41+seq_len, 50257]', 
            color_data, '#6A1B9A', 2, 9)
    add_box(5, 1, 3.5, 0.8, ' Generated Caption\n"A sunny day with clear skies..."', 
            '#C8E6C9', '#2E7D32', 3, 9)
    
    add_arrow(5, 3.7, 5, 2.8, color='#E65100', lw=2)
    add_arrow(5, 2.2, 5, 1.4, color='#2E7D32', lw=2.5)
    
    # ==================== 图例 ====================
    legend_elements = [
        mpatches.Patch(facecolor=color_frozen, edgecolor='#01579B', 
                      linewidth=2, label='Frozen (CLIP)'),
        mpatches.Patch(facecolor=color_trainable, edgecolor='#E65100', 
                      linewidth=2, label='Trainable'),
        mpatches.Patch(facecolor=color_data, edgecolor='#6A1B9A', 
                      linewidth=2, label='Data Flow'),
        mpatches.Patch(facecolor=color_process, edgecolor='#33691E', 
                      linewidth=2, label='Process')
    ]
    ax.legend(handles=legend_elements, loc='upper right', 
             fontsize=9, framealpha=0.9)
    
    # 标题
    plt.title('Weather-aware Image Captioning Architecture\nCLIP + Zero-shot Weather + GPT-2', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('weather_caption_architecture.png', dpi=300, bbox_inches='tight')
    print("架构图已保存为 weather_caption_architecture.png")
    plt.show()

# 绘制
draw_architecture()