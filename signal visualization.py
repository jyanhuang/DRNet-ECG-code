import os
import wfdb
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.colors as mcolors

# ---------- 1. 路径与记录列表 ----------
DB_DIR = r'D:\python\python项目\医学图像\mit-bih-arrhythmia-database-1.0.0'
record_names = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# ---------- 2. 符号→标签映射 ----------
BEAT2DX = {
    'N': 'NOR',
    'L': 'LBBB',
    'R': 'RBBB',
    '!': 'VF',
    'A': 'PAC',
    'V': 'PVC',
    'E': 'VEB',
    '/': 'PB',
}


# ---------- 3. 加载数据 ----------
def load_all_dx():
    dx_list = []
    for rec in record_names:
        try:
            ann = wfdb.rdann(os.path.join(DB_DIR, rec), 'atr')
            for sym in ann.symbol:
                dx = BEAT2DX.get(sym, 'Others')
                dx_list.append(dx)
        except Exception as e:
            print(f'Error loading {rec}: {e}')
    return dx_list

dx_counts = Counter(load_all_dx())

# ---------- 4. 指定顺序（VF 与 VEB 隔开） ----------
order = ['NOR', 'LBBB', 'RBBB', 'PAC', 'PVC', 'VF', 'PB', 'VEB', 'Others']
ordered_values = [dx_counts.get(label, 0) for label in order]
ordered_labels = order

# ---------- 5. 浅色配色 ----------
colors = [mcolors.to_hex(c) for c in plt.cm.Pastel1(range(len(order)))]

# ---------- 6. 绘图 ----------
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    ordered_values,
    labels=ordered_labels,
    colors=colors,
    autopct='%1.1f%%',
    pctdistance=0.8,      # 百分比更靠外
    labeldistance=1.08,    # 标签更靠近圆心
    startangle=90,
    rotatelabels=True,
    wedgeprops = dict(width=0.5)
)

# ---------- 7. 字体微调 ----------
for text in texts:
    text.set_fontsize(10)
    text.set_color('black')

for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('black')

plt.title('MIT-BIH Arrhythmia Database')
plt.tight_layout()
plt.show()


import os
import wfdb
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.colors as mcolors

# ---------- 1. 路径与记录列表 ----------
DB_DIR = r'D:\python\python项目\医学图像\st-petersburg-incart-12-lead-arrhythmia-database-1.0.0\files'
record_names = [
    'I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08', 'I09', 'I10',
    'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18', 'I19', 'I20',
    'I21', 'I22', 'I23', 'I24', 'I25', 'I26', 'I27', 'I28', 'I29', 'I30',
    'I31', 'I32', 'I33', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'I40',
    'I41', 'I42', 'I43', 'I44', 'I45', 'I46', 'I47', 'I48', 'I49', 'I50',
    'I51', 'I52', 'I53', 'I54', 'I55', 'I56', 'I57', 'I58', 'I59', 'I60',
    'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69', 'I70',
    'I71', 'I72', 'I73', 'I74', 'I75'
]

BEAT2DX = {
    "N": "N",
    "V": "P",
    "R": "R",
    "A": "A",
    "F": "F",
    "j": "j",
    "S": "S",
    "Q": "Q",
    "B": "B"
}

# ---------- 3. 加载数据 ----------
def load_all_dx():
    dx_list = []
    for rec in record_names:
        try:
            ann = wfdb.rdann(os.path.join(DB_DIR, rec), 'atr')
            for sym in ann.symbol:
                dx = BEAT2DX.get(sym, 'Others')
                dx_list.append(dx)
        except Exception as e:
            print(f'Error loading {rec}: {e}')
    return dx_list

dx_counts = Counter(load_all_dx())

order = [
    "N",
    "B",
    "Others",
    "P",
    "Q",
    "R",
    "j",
    "A",
    "S",
    "F",

]
ordered_values = [dx_counts.get(label, 0) for label in order]
ordered_labels = order
# ---------- 4. 打印每个类别的数量 ----------
print("Beat category counts:")
for label in order:
    print(f"{label}: {dx_counts.get(label, 0)}")
# ---------- 5. 浅色配色 ----------
colors = [mcolors.to_hex(c) for c in plt.cm.Pastel1(range(len(order)))]
import numpy as np

# ---------- 6. 交叉排序 ----------
# 按数量降序
sorted_by_cnt = sorted(order, key=lambda x: dx_counts[x], reverse=True)
# 交叉合并
interleave = []
for a, b in zip(sorted_by_cnt[::2], sorted_by_cnt[1::2]):
    interleave.extend([a, b])
if len(sorted_by_cnt) % 2:
    interleave.append(sorted_by_cnt[-1])

# 根据新顺序取值
ordered_labels = interleave
ordered_values = [dx_counts[l] for l in ordered_labels]
colors = [mcolors.to_hex(c) for c in plt.cm.Pastel1(range(len(order)))]

# ---------- 7. 绘图 ----------
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, _ = ax.pie(
    ordered_values,
    labels=None,
    colors=colors,
    autopct='',
    startangle=90,
    wedgeprops=dict(width=0.5)
)


# ---------- 9. 图例（含百分比） ----------
total = sum(ordered_values)
ax.legend([f'{l}: {dx_counts[l]}  ({v/total*100:.1f}%)'
           for l, v in zip(ordered_labels, ordered_values)],
          title='Beat Types', loc='center left',
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('st-petersburg-incart-12-lead-arrhythmia-database')
plt.tight_layout()
plt.show()


import os
import wfdb
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.colors as mcolors
import numpy as np

# ---------- 1. 路径与记录列表 ----------
DB_DIR = r'D:\python\python项目\医学图像\st-petersburg-incart-12-lead-arrhythmia-database-1.0.0\files'

record_names = [f'I{i:02d}' for i in range(1, 76)]   # I01 … I75

BEAT2DX = {"N": "N", "V": "P", "R": "R", "A": "A", "F": "F",
           "j": "j", "S": "S", "Q": "Q", "B": "B"}

# ---------- 2. 加载所有心拍 ----------
def load_all_dx():
    dx_list = []
    for rec in record_names:
        try:
            ann = wfdb.rdann(os.path.join(DB_DIR, rec), 'atr')
            dx_list += [BEAT2DX.get(sym, 'Others') for sym in ann.symbol]
        except Exception as e:
            print(f'Error loading {rec}: {e}')
    return dx_list

if __name__ == '__main__':
    dx_counts = Counter(load_all_dx())

    # ---------- 3. 交叉排序 ----------
    sorted_by_cnt = sorted(dx_counts.keys(), key=lambda x: dx_counts[x], reverse=True)
    interleave = []
    for a, b in zip(sorted_by_cnt[::2], sorted_by_cnt[1::2]):
        interleave.extend([a, b])
    if len(sorted_by_cnt) % 2:
        interleave.append(sorted_by_cnt[-1])

    ordered_labels = interleave
    ordered_values = [dx_counts[l] for l in ordered_labels]

    # ---------- 4. 浅色 -> 深色 ----------
    base_colors = plt.cm.Pastel1(np.linspace(0, 1, len(ordered_labels)))
    dark_labels = {'F', 'j', 'Others', 'S', 'Q', 'B'}
    colors = []
    for lab, c in zip(ordered_labels, base_colors):
        if lab in dark_labels:
            h, s, v = mcolors.rgb_to_hsv(c[:3])
            colors.append(mcolors.hsv_to_rgb((h, s, v * 0.7)))
        else:
            colors.append(c[:3])

    # ================= 8. 竖直条形图（0-2000 拉长） =================
    total = sum(ordered_values)
    pct = [v / total * 100 for v in ordered_values]

    fig, ax = plt.subplots(figsize=(11, 8))


    # 1. 映射：0-2000 放大 3 倍；2000 以上正常
    def forward(y):
        return np.where(y <= 2000, y * 5, y + 4000)  # 2000*3=6000，过渡点 6000


    def inverse(y):
        return np.where(y <= 6000, y / 5, y - 4000)


    ax.set_yscale('function', functions=(forward, inverse))

    # 2. 手动刻度（真实值），仍以 20000 为间距
    max_y = max(ordered_values) * 1.05
    tick_values = np.arange(0, max_y + 1, 20000)
    ax.set_yticks(forward(tick_values))
    ax.set_yticklabels([str(int(v)) for v in tick_values])

    # 3. 画柱子
    bars = ax.bar(ordered_labels, ordered_values, color=colors)

    # 4. 柱顶：数量 + 百分比
    for bar, v, p in zip(bars, ordered_values, pct):
        if v == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + total * 0.02,
                f'{int(v)}\n{p:.1f}%',
                ha='center', va='bottom', fontsize=9, color='black')

    ax.set_ylabel('Count')
    ax.set_xlabel('Beat Type')
    ax.set_title('st-petersburg-incart-12-lead-arrhythmia-database')
    plt.tight_layout()
    plt.show()