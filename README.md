# 🎞️ Film Lab - Python Computational Photography

这是一个基于 Python (OpenCV + NumPy) 的胶片模拟实验室。
它不是简单的滤镜，而是通过计算图片的光感、色彩耦合矩阵和物理颗粒模拟，还原 Kodak, Fuji, Cinestill 等经典胶片的质感。

## ✨ 特性
- **物理光感模拟**: 线性光空间下的光晕 (Bloom) 和光线漫射。
- **动态颗粒引擎**: 基于亮度的自适应银盐颗粒合成。
- **经典胶片预设**: Fuji Pro 400H, Kodak Portra 400, Cinestill 800T, Ricoh GR 等。
- **隐私安全**: 图片仅在内存中处理，处理完即销毁，不保存到硬盘。

## 🛠️ 安装与运行

1. 克隆项目
   ```bash
   git clone https://github.com/你的用户名/film-lab.git
   cd filab
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 运行服务器
   ```bash
   python app.py
   ```

4. 打开浏览器访问
   `http://127.0.0.1:5000`

## 📝 胶片模型
- **Fuji Pro 400H**: 青色阴影，日系通透感。
- **Cinestill 800T**: 钨丝灯白平衡，特有的红色光晕 (Halation)。
- **Ilford HP5**: 经典高感黑白，粗颗粒。
```
