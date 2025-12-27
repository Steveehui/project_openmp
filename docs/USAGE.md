# 简明使用指南

## 🚀 三步快速开始

### 步骤1: 编译项目
```powershell
cd D:\Desktop\project_openmp
.\scripts\build.ps1
```

### 步骤2: 运行测试
```powershell
# 串行版本
.\bin\conv_serial.exe 1024

# OpenMP 版本 (16线程)
.\bin\conv_omp.exe 1024 16
```

### 步骤3: 性能对比
```powershell
.\scripts\quick_test.ps1
```

---

## 📊 命令参数说明

### 串行版本
```powershell
.\bin\conv_serial.exe [图像尺寸] [图案类型] [选项]
```

**参数**:
- `图像尺寸`: 512, 1024, 2048 等
- `图案类型`: random(随机), gradient(渐变), checkerboard(棋盘)
- `选项`: --save (保存输出图像)

**示例**:
```powershell
.\bin\conv_serial.exe 512 random
.\bin\conv_serial.exe 2048 gradient --save
```

### OpenMP 版本
```powershell
.\bin\conv_omp.exe [图像尺寸] [线程数] [图案类型] [选项]
```

**参数**:
- `线程数`: 1, 2, 4, 8, 16 等
- `选项`: --batch (批处理模式), --verify (验证正确性)

**示例**:
```powershell
.\bin\conv_omp.exe 1024 8 random
.\bin\conv_omp.exe 2048 16 gradient --batch
.\bin\conv_omp.exe 512 4 random --verify
```

---

## 🧪 实验场景

### 场景1: 测试不同线程数的加速比
```powershell
# 串行基准
.\bin\conv_serial.exe 1024

# 依次测试 1, 2, 4, 8, 16 线程
.\bin\conv_omp.exe 1024 1
.\bin\conv_omp.exe 1024 2
.\bin\conv_omp.exe 1024 4
.\bin\conv_omp.exe 1024 8
.\bin\conv_omp.exe 1024 16
```

### 场景2: 测试不同图像尺寸的扩展性
```powershell
# 小图像
.\bin\conv_omp.exe 512 16

# 中图像
.\bin\conv_omp.exe 1024 16

# 大图像
.\bin\conv_omp.exe 2048 16
```

### 场景3: 正确性验证
```powershell
# 对比 OpenMP 和串行结果
.\bin\conv_omp.exe 512 8 random --verify
```

---

## 📈 记录实验数据

### 手动记录模板
```
实验1: 强扩展性测试 (1024×1024)
--------------------------------------
线程数 | 运行时间(秒) | 加速比 | 效率(%)
   1   |    ____     |  1.00  |  100
   2   |    ____     |  ____  |  ____
   4   |    ____     |  ____  |  ____
   8   |    ____     |  ____  |  ____
  16   |    ____     |  ____  |  ____
```

### 计算公式
- **加速比** = 串行时间 / 并行时间
- **并行效率** = 加速比 / 线程数 × 100%

---

## 🐛 故障排查

### 问题1: 找不到 conv_serial.exe
**解决**: 重新运行构建脚本
```powershell
.\scripts\build.ps1
```

### 问题2: 中文输出乱码
**解决**: 切换终端编码(临时)
```powershell
chcp 65001
```

### 问题3: OpenMP 不工作(运行时间相同)
**检查**: 确认线程数参数正确
```powershell
# 错误: 缺少线程数参数
.\bin\conv_omp.exe 1024

# 正确: 指定线程数
.\bin\conv_omp.exe 1024 8
```

---

## 💡 优化建议

### 提高串行基准时间(>10秒)
```powershell
# 方法1: 增大图像尺寸
.\bin\conv_serial.exe 4096

# 方法2: 多次运行取平均
for ($i=1; $i -le 5; $i++) {
    .\bin\conv_serial.exe 2048
}
```

### 最佳线程数选择
- **经验值**: CPU 核心数的 1-1.5 倍
- **i7-13700H**: 建议 8-16 线程
- **测试方法**: 逐步增加观察收益递减点

---

## 📝 报告用图表建议

### 图表1: 加速比曲线
- **X轴**: 线程数 (1, 2, 4, 8, 16)
- **Y轴**: 加速比
- **对比线**: 理想线性加速 (y=x)

### 图表2: 并行效率
- **X轴**: 线程数
- **Y轴**: 效率百分比
- **参考线**: 100% 理想效率

### 图表3: 不同尺寸对比
- **柱状图**: 串行 vs OpenMP
- **数据**: 512, 1024, 2048 三组

---

## ⏱️ 预计测试时间

- **单次测试**: 0.01 - 0.5 秒
- **完整场景1**: 约 2 分钟
- **完整场景2**: 约 3 分钟
- **quick_test.ps1**: 约 5 分钟

---

**提示**: 实验过程中注意关闭其他占用 CPU 的程序,确保测试准确性!
