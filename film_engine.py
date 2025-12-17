import cv2
import numpy as np

class FilmPhysics:
    @staticmethod
    def apply_lut_curve(img, points):
        """[修复版] 曲线查找表"""
        x_points, y_points = zip(*points)
        lut = np.interp(np.arange(256), x_points, y_points).astype(np.uint8)
        # 必须 reshape 才能兼容多通道
        lut = lut.reshape(256, 1)
        return cv2.LUT(img, lut)

    @staticmethod
    def channel_mix(img, matrix):
        """色彩耦合"""
        img_float = img.astype(np.float32) / 255.0
        result = img_float @ np.array(matrix).T 
        return np.clip(result, 0, 1) * 255

    @staticmethod
    def smart_grain(img, intensity=0.05, softness=0.5):
        """
        [自适应颗粒]
        """
        if intensity <= 0.001: return img # 如果强度太低直接返回原图
        
        h, w, c = img.shape
        # 基础噪声
        noise = np.random.normal(0, intensity * 255, (h, w, c)).astype(np.float32)
        
        # 柔化
        if softness > 0:
            k = 3
            noise = cv2.GaussianBlur(noise, (k, k), softness)

        # 亮度掩码 (中间调颗粒最强)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        grain_mask = 1.0 - 3.5 * (gray - 0.5) ** 2
        grain_mask = np.clip(grain_mask, 0.2, 1.0) 
        grain_mask = np.stack([grain_mask]*3, axis=2)

        return np.clip(img.astype(np.float32) + noise * grain_mask, 0, 255).astype(np.uint8)

    @staticmethod
    def red_halation(img, threshold=0.9, strength=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mask = np.maximum(0, gray - threshold) / (1 - threshold + 1e-6)
        halo = cv2.GaussianBlur(mask, (0, 0), sigmaX=15, sigmaY=15)
        halo = np.stack([halo]*3, axis=2)
        img_float = img.astype(np.float32)
        img_float[:,:,2] += halo[:,:,2] * 255 * strength * 1.5 
        img_float[:,:,1] += halo[:,:,1] * 255 * strength * 0.2 
        return np.clip(img_float, 0, 255).astype(np.uint8)

    @staticmethod
    def standard_bloom(img, radius=20, strength=0.3):
        if radius % 2 == 0: radius += 1
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)
        img_f = img.astype(np.float32) / 255.0
        blur_f = blur.astype(np.float32) / 255.0
        res = 1.0 - (1.0 - img_f) * (1.0 - blur_f * strength)
        return np.clip(res * 255, 0, 255).astype(np.uint8)

# ================= 胶片配方库 (增加了 grain_scale 参数) =================

def process_fuji_pro400h(img, grain_scale=1.0):
    fp = FilmPhysics()
    # 1. 色彩
    matrix = [[0.92, 0.05, 0.03], [0.00, 1.02, 0.00], [0.02, 0.05, 0.93]]
    img = fp.channel_mix(img, matrix).astype(np.uint8)

    # 2. 曲线
    curve_master = [(0,0), (40, 35), (128, 135), (220, 235), (255, 255)]
    img = fp.apply_lut_curve(img, curve_master)
    b, g, r = cv2.split(img)
    b = fp.apply_lut_curve(b, [(0, 15), (128, 128), (255, 250)]) 
    r = fp.apply_lut_curve(r, [(0, 0), (128, 125), (255, 245)])
    img = cv2.merge([b, g, r])

    # 3. 质感 (基础强度 0.06 * 用户缩放)
    img = fp.standard_bloom(img, radius=13, strength=0.25)
    img = fp.smart_grain(img, intensity=0.06 * grain_scale, softness=0.7)
    return img

def process_kodak_portra400(img, grain_scale=1.0):
    fp = FilmPhysics()
    matrix = [[0.85, 0.10, 0.05], [0.02, 1.00, -0.02], [-0.02, 0.05, 1.05]]
    img = fp.channel_mix(img, matrix).astype(np.uint8)
    curve = [(0, 5), (60, 65), (180, 190), (255, 240)]
    img = fp.apply_lut_curve(img, curve)
    # Portra 颗粒很细 (基础 0.04)
    img = fp.smart_grain(img, intensity=0.04 * grain_scale, softness=0.8)
    return img

def process_cinestill_800t(img, grain_scale=1.0):
    fp = FilmPhysics()
    b, g, r = cv2.split(img)
    b = cv2.addWeighted(b, 1.1, np.zeros_like(b), 0, 0)
    r = cv2.addWeighted(r, 0.9, np.zeros_like(r), 0, 0)
    img = cv2.merge([b, g, r])
    curve = [(0, 0), (50, 40), (200, 220), (255, 245)]
    img = fp.apply_lut_curve(img, curve)
    img = fp.red_halation(img, threshold=0.85, strength=0.6)
    # 基础 0.08
    img = fp.smart_grain(img, intensity=0.08 * grain_scale, softness=0.5)
    return img

def process_polaroid(img, grain_scale=1.0):
    fp = FilmPhysics()
    matrix = [[0.85, 0.10, 0.05], [0.05, 0.95, 0.00], [0.00, 0.05, 1.05]]
    img = fp.channel_mix(img, matrix).astype(np.uint8)
    curve = [(0, 30), (50, 60), (128, 140), (255, 240)]
    img = fp.apply_lut_curve(img, curve)
    b, g, r = cv2.split(img)
    b = fp.apply_lut_curve(b, [(0,0), (255, 220)])
    img = cv2.merge([b, g, r])
    img = fp.standard_bloom(img, radius=25, strength=0.5)
    # 宝丽来颗粒较重 (基础 0.10)
    img = fp.smart_grain(img, intensity=0.10 * grain_scale, softness=0.4)
    return img

def process_ilford_hp5(img, grain_scale=1.0):
    fp = FilmPhysics()
    b, g, r = cv2.split(img.astype(np.float32))
    gray = 0.1*b + 0.5*g + 0.4*r 
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    img = cv2.merge([gray, gray, gray])
    curve = [(0, 0), (60, 50), (190, 210), (255, 255)]
    img = fp.apply_lut_curve(img, curve)
    # 黑白卷颗粒最粗 (基础 0.15)
    img = fp.smart_grain(img, intensity=0.15 * grain_scale, softness=0.3)
    return img

def process_ricoh_gr(img, grain_scale=1.0):
    fp = FilmPhysics()
    img = fp.apply_lut_curve(img, [(0,0), (40, 20), (128, 145), (210, 245), (255, 255)])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= 1.35
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    h, w = img.shape[:2]
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X**2 + Y**2)
    mask = 1 - np.clip(radius - 0.5, 0, 1) * 0.9
    mask = np.stack([mask]*3, axis=2)
    img = (img.astype(np.float32) * mask).astype(np.uint8)
    # GR 颗粒 (基础 0.12)
    img = fp.smart_grain(img, intensity=0.12 * grain_scale, softness=0.0)
    return img

# 统一入口：接收 grain_scale
def process_style_v2(img, style_name, grain_scale=1.0):
    if style_name == 'fuji': return process_fuji_pro400h(img, grain_scale)
    elif style_name == 'kodak': return process_kodak_portra400(img, grain_scale)
    elif style_name == 'cinestill': return process_cinestill_800t(img, grain_scale)
    elif style_name == 'polaroid': return process_polaroid(img, grain_scale)
    elif style_name == 'ilford': return process_ilford_hp5(img, grain_scale)
    elif style_name == 'ricoh': return process_ricoh_gr(img, grain_scale)
    else: return process_fuji_pro400h(img, grain_scale)