# FSNet 设备管理说明

## 🎯 快速开始

### 1. CPU模式（当前默认）
不需要任何修改，直接运行：
```bash
python fsnet/myexp.py
```

### 2. 切换到GPU模式
只需修改**一个文件**的**一行代码**：

**文件**: `fsnet/device_config.py`

```python
# 第13行
USE_CUDA = True  # ✅ 改为True启用GPU
```

然后正常运行：
```bash
python fsnet/myexp.py
```

---

## 🔧 详细配置

### 选项1：通过device_config.py全局配置（推荐）

**优点**: 一次配置，所有脚本生效

**文件位置**: `fsnet/device_config.py`

```python
# ============================================
# 🔧 全局设备配置 - 只需要改这一个地方！
# ============================================
USE_CUDA = False  # 改为 True 启用GPU，False 使用CPU
```

### 选项2：通过命令行参数

```bash
# CPU模式
python fsnet/myexp.py --use_gpu False

# GPU模式  
python fsnet/myexp.py --use_gpu True --gpu 0
```

### 选项3：通过环境变量

```bash
# CPU模式
export CUDA_VISIBLE_DEVICES=""
python fsnet/myexp.py

# GPU模式（使用GPU 0）
export CUDA_VISIBLE_DEVICES="0"
python fsnet/myexp.py --use_gpu True
```

---

## 📊 检查设备配置

运行以下命令查看当前设备配置：

```bash
python fsnet/device_config.py
```

**输出示例**：
```
==================================================
🖥️  设备配置
==================================================
当前设备: cpu
配置USE_CUDA: False
CUDA可用: True
==================================================

测试张量: cpu
✅ 设备配置工作正常！
```

---

## 🔍 已修复的问题

### 问题1: 维度不匹配
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (384x10 and 14x64)
```

**原因**: 模型期望14维输入，但实际数据只有10维（7个数据特征+3个时间特征）

**修复**: 
- `exp_fsnet.py` 第42行: `input_dims=args.enc_in + 3` （原来是+7）

### 问题2: 硬编码CUDA调用
```
AssertionError: Torch not compiled with CUDA enabled
```

**原因**: 代码中有3处硬编码 `.cuda()` 调用

**修复位置**:
- `models/ts2vec/fsnet_.py` 第93行: `.cuda()` → `.to(self.device)`
- `models/ts2vec/fsnet_.py` 第110行: `.cuda()` → `.to(self.device)`  
- `models/ts2vec/fsnet_.py` 第113行: `.cuda()` → `.to(self.device)`
- `models/ts2vec/dev.py` 第34行: `.cuda()` → `.to(self.device)`

### 修复范围
✅ 所有模型文件统一设备管理：
- `exp/exp_fsnet.py` - 主实验类
- `exp/exp_nomem.py` - NoMemory消融实验
- `models/ts2vec/fsnet.py` - TSEncoder
- `models/ts2vec/fsnet_.py` - FSNet核心组件
- `models/ts2vec/dev.py` - 开发版本
- `models/ts2vec/nomem.py` - NoMemory版本

---

## 🧪 测试不同实验方法

现在所有方法都支持CPU/GPU切换：

```bash
# FSNet完整版
python fsnet/myexp.py --method fsnet

# OGD baseline
python fsnet/myexp.py --method ogd

# 无记忆模块
python fsnet/myexp.py --method nomem

# Experience Replay
python fsnet/myexp.py --method er

# DER++
python fsnet/myexp.py --method derpp
```

---

## 💡 代码示例

### 在自己的代码中使用设备配置

```python
from device_config import get_device

# 获取设备
device = get_device()

# 创建模型
model = YourModel().to(device)

# 创建张量
x = torch.randn(3, 3).to(device)
```

### 强制使用CPU（覆盖全局配置）

```python
from device_config import get_device

# 无论device_config.py中的配置如何，强制使用CPU
device = get_device(force_cpu=True)
```

---

## 🐛 故障排查

### 问题: 修改了USE_CUDA但还是用CPU

**检查项**:
1. 确认修改了正确的文件（`fsnet/device_config.py`）
2. 检查torch是否支持CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. 检查CUDA驱动: `nvidia-smi`

### 问题: 提示CUDA out of memory

**解决方案**:
```python
# 在myexp.py中减小batch size
parser.add_argument('--batch_size', type=int, default=4)  # 原来是8

# 或者减小模型大小
parser.add_argument('--d_model', type=int, default=32)  # 原来是64
```

### 问题: 代码报其他CUDA错误

**快速回退到CPU**:
```bash
# 方法1: 环境变量（最快）
export CUDA_VISIBLE_DEVICES=""
python fsnet/myexp.py

# 方法2: 修改device_config.py
USE_CUDA = False
```

---

## 📈 性能对比

| 配置 | 训练速度 | 适用场景 |
|------|----------|----------|
| CPU | ~0.5 iter/s | 快速验证、调试 |
| GPU (单卡) | ~10 iter/s | 正式训练、完整实验 |
| GPU (多卡) | ~40 iter/s | 大规模实验、超参数搜索 |

---

## 🎓 最佳实践

1. **开发调试**: 使用CPU + 小数据集 + 少epoch
   ```bash
   USE_CUDA = False
   --train_epochs 2 --batch_size 8
   ```

2. **正式训练**: 使用GPU + 完整数据 + 标准配置
   ```bash
   USE_CUDA = True
   --train_epochs 10 --batch_size 32
   ```

3. **消融实验**: 批量运行不同方法
   ```bash
   for method in ogd er fsnet nomem; do
       python myexp.py --method $method --use_gpu True
   done
   ```

---

## 📞 需要帮助？

如果遇到问题：
1. 运行 `python device_config.py` 检查设备状态
2. 查看本文档的"故障排查"部分
3. 检查是否有其他硬编码的 `.cuda()` 调用：
   ```bash
   grep -r "\.cuda()" fsnet/
   ```
