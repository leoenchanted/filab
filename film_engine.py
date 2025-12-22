import cv2
import numpy as np

# ==================== 光学基础函数 ====================

def srgb_to_linear(img):
    """
    sRGB → 线性光空间 (符合人眼感知的 gamma 2.2)
    这是光学计算的基础!所有混合/叠加都应在线性空间进行
    """
    img_norm = img.astype(np.float32) / 255.0
    linear = np.where(
        img_norm <= 0.04045,
        img_norm / 12.92,
        np.power((img_norm + 0.055) / 1.055, 2.4)
    )
    return linear

def linear_to_srgb(linear):
    """线性光 → sRGB (用于显示)"""
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1/2.4) - 0.055
    )
    return np.clip(srgb * 255, 0, 255).astype(np.uint8)

def get_luminance(linear_img):
    """计算感知亮度 (ITU-R BT.709 标准)"""
    return (0.2126 * linear_img[:,:,2] + 
            0.7152 * linear_img[:,:,1] + 
            0.0722 * linear_img[:,:,0])

# ==================== 胶片物理引擎 ====================

class FilmPhysics:
    @staticmethod
    def apply_lut_curve(img, points):
        """色调曲线映射 (兼容多通道)"""
        x_points, y_points = zip(*points)
        lut = np.interp(np.arange(256), x_points, y_points).astype(np.uint8)
        lut = lut.reshape(256, 1)
        return cv2.LUT(img, lut)

    @staticmethod
    def channel_mix_linear(img, matrix):
        """
        [优化] 在线性光空间进行色彩混合
        模拟胶片染料层的光谱吸收特性
        """
        linear = srgb_to_linear(img)
        result = linear @ np.array(matrix).T
        result = np.clip(result, 0, 1)
        return linear_to_srgb(result)

    @staticmethod
    def spectral_grain(img, intensity=0.05, softness=0.5):
        """
        [优化] 基于亮度的单层颗粒 (更接近银盐物理)
        真实胶片的颗粒是银盐/染料颗粒,主要影响亮度而非色相
        """
        if intensity <= 0.001:
            return img
        
        h, w, c = img.shape
        linear = srgb_to_linear(img)
        
        # 1. 计算感知亮度
        luminance = get_luminance(linear)
        
        # 2. 生成单层亮度噪声
        grain = np.random.normal(0, intensity, (h, w)).astype(np.float32)
        
        # 3. 柔化处理 (模拟颗粒大小)
        if softness > 0:
            ksize = max(3, int(softness * 2) | 1)  # 奇数
            grain = cv2.GaussianBlur(grain, (ksize, ksize), softness)
        
        # 4. 亮度掩码 (中间调颗粒最强)
        grain_mask = 1.0 - 3.5 * (luminance - 0.5) ** 2
        grain_mask = np.clip(grain_mask, 0.2, 1.0)
        
        # 5. 应用到亮度通道
        luminance_with_grain = luminance + grain * grain_mask
        luminance_with_grain = np.clip(luminance_with_grain, 0, 1)
        
        # 6. 保持色度,只改变亮度
        scale = np.divide(
            luminance_with_grain, 
            luminance + 1e-6,
            out=np.ones_like(luminance),
            where=luminance > 1e-6
        )
        scale = np.stack([scale] * 3, axis=2)
        result = linear * scale
        
        return linear_to_srgb(np.clip(result, 0, 1))

    @staticmethod
    def chromatic_grain(img, intensity=0.06, color_variance=0.25):
        """
        [可选] 带轻微色彩偏移的颗粒
        适用于彩色负片 (染料颗粒不均导致的色彩噪声)
        """
        if intensity <= 0.001:
            return img
        
        h, w, c = img.shape
        linear = srgb_to_linear(img)
        luminance = get_luminance(linear)
        
        # 主亮度噪声
        luma_grain = np.random.normal(0, intensity, (h, w, 1)).astype(np.float32)
        
        # 次色度噪声 (颗粒更大,强度更低)
        chroma_grain = np.random.normal(0, intensity * color_variance, (h, w, 3)).astype(np.float32)
        chroma_grain = cv2.GaussianBlur(chroma_grain, (0, 0), sigmaX=1.8)
        
        # 合并
        grain = luma_grain + chroma_grain
        grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.7)
        
        # 亮度掩码
        mask = 1.0 - 3.5 * (luminance - 0.5) ** 2
        mask = np.clip(mask, 0.2, 1.0)
        mask = np.stack([mask] * 3, axis=2)
        
        result = linear + grain * mask
        return linear_to_srgb(np.clip(result, 0, 1))

    @staticmethod
    def spectral_halation(img, threshold=0.85, strength=0.6, color_shift=(1.5, 0.3, 0.2)):
        """
        [优化] 光谱级光晕效应
        模拟光在胶片乳剂层中的散射
        color_shift: (R, G, B) 各通道的光晕强度
        """
        linear = srgb_to_linear(img)
        luminance = get_luminance(linear)
        
        # 1. 提取高光区域
        mask = np.maximum(0, luminance - threshold) / (1 - threshold + 1e-6)
        
        # 2. 多尺度光晕 (模拟不同深度的散射)
        halo_small = cv2.GaussianBlur(mask, (0, 0), sigmaX=8)
        halo_large = cv2.GaussianBlur(mask, (0, 0), sigmaX=20)
        halo = 0.7 * halo_small + 0.3 * halo_large
        
        # 3. 分通道应用 (BGR 顺序)
        halo_3ch = np.stack([halo] * 3, axis=2)
        color_weight = np.array([color_shift[2], color_shift[1], color_shift[0]]).reshape(1, 1, 3)
        linear += halo_3ch * color_weight * strength
        
        return linear_to_srgb(np.clip(linear, 0, 1))

    @staticmethod
    def standard_bloom(img, radius=20, strength=0.3):
        """柔光/辉光效果 (Screen 混合模式)"""
        if radius % 2 == 0:
            radius += 1
        
        linear = srgb_to_linear(img)
        blur = srgb_to_linear(cv2.GaussianBlur(img, (0, 0), sigmaX=radius))
        
        # Screen 混合: 1 - (1-A)*(1-B)
        result = 1.0 - (1.0 - linear) * (1.0 - blur * strength)
        return linear_to_srgb(np.clip(result, 0, 1))

    @staticmethod
    def film_log_curve(img, shoulder=0.92, toe=0.08, contrast=1.15):
        """
        [可选] 对数响应曲线 (更符合胶片物理)
        可替代 LUT 查找表
        """
        linear = srgb_to_linear(img)
        
        # 对数曲线
        k = 10 ** contrast
        log_curve = np.log(1 + linear * k) / np.log(1 + k)
        
        # Toe 提亮暗部
        log_curve = np.where(
            log_curve < toe,
            toe + np.power(log_curve / toe, 1.5) * (toe * 0.5),
            log_curve
        )
        
        # Shoulder 压缩高光
        log_curve = np.where(
            log_curve > shoulder,
            shoulder + (log_curve - shoulder) * 0.3,
            log_curve
        )
        
        return linear_to_srgb(np.clip(log_curve, 0, 1))

# ==================== 胶片配方库 ====================

def process_fuji_pro400h(img, grain_scale=1.0):
    """富士 Pro 400H - 人像负片之王"""
    fp = FilmPhysics()
    
    # 1. 富士染料矩阵 (青色偏移特性)
    matrix = [[0.92, 0.06, 0.02], [0.03, 0.95, 0.02], [0.05, 0.08, 0.87]]
    img = fp.channel_mix_linear(img, matrix)
    
    # 2. 曲线调整 (降低对比度)
    curve_master = [(0, 0), (40, 38), (128, 135), (220, 232), (255, 252)]
    img = fp.apply_lut_curve(img, curve_master)
    
    b, g, r = cv2.split(img)
    b = fp.apply_lut_curve(b, [(0, 18), (128, 128), (255, 248)])
    r = fp.apply_lut_curve(r, [(0, 0), (128, 125), (255, 242)])
    img = cv2.merge([b, g, r])
    
    # 3. 质感处理 (光谱级颗粒)
    img = fp.standard_bloom(img, radius=13, strength=0.25)
    img = fp.spectral_grain(img, intensity=0.06 * grain_scale, softness=0.7)
    
    return img

def process_kodak_portra400(img, grain_scale=1.0):
    """柯达 Portra 400 - 肤色之王"""
    fp = FilmPhysics()
    
    # Kodak 染料矩阵 (暖色倾向)
    matrix = [[0.88, 0.08, 0.04], [0.02, 0.98, 0.00], [0.03, 0.05, 0.92]]
    img = fp.channel_mix_linear(img, matrix)
    
    curve = [(0, 8), (60, 68), (180, 188), (255, 238)]
    img = fp.apply_lut_curve(img, curve)
    
    # Portra 颗粒极细
    img = fp.spectral_grain(img, intensity=0.04 * grain_scale, softness=0.9)
    
    return img

def process_cinestill_800t(img, grain_scale=1.0):
    """Cinestill 800T - 霓虹之王"""
    fp = FilmPhysics()
    
    # 色温调整 (钨丝灯平衡)
    b, g, r = cv2.split(img)
    b = cv2.addWeighted(b, 1.12, np.zeros_like(b), 0, 0)
    r = cv2.addWeighted(r, 0.88, np.zeros_like(r), 0, 0)
    img = cv2.merge([b, g, r])
    
    curve = [(0, 0), (50, 42), (200, 218), (255, 242)]
    img = fp.apply_lut_curve(img, curve)
    
    # 标志性红色光晕 (去防光晕层导致)
    img = fp.spectral_halation(img, threshold=0.85, strength=0.6, 
                                color_shift=(1.8, 0.4, 0.1))
    
    # 带色彩偏移的颗粒
    img = fp.chromatic_grain(img, intensity=0.08 * grain_scale, color_variance=0.3)
    
    return img

def process_polaroid(img, grain_scale=1.0):
    """宝丽来 - 复古即显"""
    fp = FilmPhysics()
    
    # 宝丽来染料特性
    matrix = [[0.85, 0.10, 0.05], [0.05, 0.95, 0.00], [0.00, 0.05, 1.05]]
    img = fp.channel_mix_linear(img, matrix)
    
    curve = [(0, 35), (50, 65), (128, 138), (255, 235)]
    img = fp.apply_lut_curve(img, curve)
    
    b, g, r = cv2.split(img)
    b = fp.apply_lut_curve(b, [(0, 0), (255, 215)])
    img = cv2.merge([b, g, r])
    
    # 强烈的柔光效果
    img = fp.standard_bloom(img, radius=28, strength=0.55)
    
    # 粗颗粒
    img = fp.chromatic_grain(img, intensity=0.10 * grain_scale, color_variance=0.35)
    
    return img

def process_ilford_hp5(img, grain_scale=1.0):
    """Ilford HP5 Plus - 经典黑白"""
    fp = FilmPhysics()
    
    # 自定义灰度转换 (偏重绿色和红色通道)
    linear = srgb_to_linear(img)
    b, g, r = cv2.split(linear)
    gray = 0.10 * b + 0.50 * g + 0.40 * r
    gray = np.clip(gray, 0, 1)
    gray_srgb = linear_to_srgb(np.stack([gray] * 3, axis=2))
    
    # 高对比度曲线
    curve = [(0, 0), (60, 48), (190, 215), (255, 255)]
    gray_srgb = fp.apply_lut_curve(gray_srgb, curve)
    
    # 粗颗粒 (黑白卷特色)
    gray_srgb = fp.spectral_grain(gray_srgb, intensity=0.15 * grain_scale, softness=0.3)
    
    return gray_srgb

def process_ricoh_gr(img, grain_scale=1.0):
    """Ricoh GR - 街拍高反差"""
    fp = FilmPhysics()
    
    # 高对比度 S 曲线
    img = fp.apply_lut_curve(img, [(0, 0), (40, 22), (128, 145), (210, 242), (255, 255)])
    
    # 饱和度大增
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.38
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 边缘暗角
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X ** 2 + Y ** 2)
    mask = 1 - np.clip(radius - 0.5, 0, 1) * 0.85
    mask = np.stack([mask] * 3, axis=2)
    img = (img.astype(np.float32) * mask).astype(np.uint8)
    
    # 锐利颗粒 (不柔化)
    img = fp.spectral_grain(img, intensity=0.12 * grain_scale, softness=0.0)
    
    return img

# ==================== 统一入口 ====================

def process_style_v2(img, style_name, grain_scale=1.0):
    """
    主处理函数
    
    参数:
        img: BGR 图像 (OpenCV 格式)
        style_name: 胶片风格 ('fuji', 'kodak', 'cinestill', 'polaroid', 'ilford', 'ricoh')
        grain_scale: 颗粒强度倍数 (0.0 = 无颗粒, 1.0 = 标准, 2.0 = 加倍)
    
    返回:
        处理后的 BGR 图像
    """
    style_map = {
        'fuji': process_fuji_pro400h,
        'kodak': process_kodak_portra400,
        'cinestill': process_cinestill_800t,
        'polaroid': process_polaroid,
        'ilford': process_ilford_hp5,
        'ricoh': process_ricoh_gr
    }
    
    process_func = style_map.get(style_name, process_fuji_pro400h)
    return process_func(img, grain_scale)
