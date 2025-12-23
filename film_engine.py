import cv2
import numpy as np

# ==================== 光学基础函数 ====================

def srgb_to_linear(img):
img_norm = img.astype(np.float32) / 255.0
linear = np.where(
img_norm <= 0.04045,
img_norm / 12.92,
np.power((img_norm + 0.055) / 1.055, 2.4)
)
return linear

def linear_to_srgb(linear):
srgb = np.where(
linear <= 0.0031308,
linear * 12.92,
1.055 * np.power(linear, 1/2.4) - 0.055
)
return np.clip(srgb * 255, 0, 255).astype(np.uint8)

def get_luminance(linear_img):
return (0.2126 * linear_img[:,:,2] +
0.7152 * linear_img[:,:,1] +
0.0722 * linear_img[:,:,0])

# ==================== 胶片物理引擎 ====================

class FilmPhysics:
@staticmethod
def apply_lut_curve(img, points):
x_points, y_points = zip(*points)
lut = np.interp(np.arange(256), x_points, y_points).astype(np.uint8)
lut = lut.reshape(256, 1)
return cv2.LUT(img, lut)

```
@staticmethod
def channel_mix_linear(img, matrix):
    linear = srgb_to_linear(img)
    result = linear @ np.array(matrix).T
    result = np.clip(result, 0, 1)
    return linear_to_srgb(result)

@staticmethod
def spectral_grain(img, intensity=0.05, softness=0.5):
    if intensity <= 0.001:
        return img
    
    h, w, c = img.shape
    linear = srgb_to_linear(img)
    luminance = get_luminance(linear)
    
    grain = np.random.normal(0, intensity, (h, w)).astype(np.float32)
    
    if softness > 0:
        ksize = max(3, int(softness * 2) | 1)
        grain = cv2.GaussianBlur(grain, (ksize, ksize), softness)
    
    grain_mask = 1.0 - 3.5 * (luminance - 0.5) ** 2
    grain_mask = np.clip(grain_mask, 0.2, 1.0)
    
    luminance_with_grain = luminance + grain * grain_mask
    luminance_with_grain = np.clip(luminance_with_grain, 0, 1)
    
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
    if intensity <= 0.001:
        return img
    
    h, w, c = img.shape
    linear = srgb_to_linear(img)
    luminance = get_luminance(linear)
    
    luma_grain = np.random.normal(0, intensity, (h, w, 1)).astype(np.float32)
    chroma_grain = np.random.normal(0, intensity * color_variance, (h, w, 3)).astype(np.float32)
    chroma_grain = cv2.GaussianBlur(chroma_grain, (0, 0), sigmaX=1.8)
    
    grain = luma_grain + chroma_grain
    grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.7)
    
    mask = 1.0 - 3.5 * (luminance - 0.5) ** 2
    mask = np.clip(mask, 0.2, 1.0)
    mask = np.stack([mask] * 3, axis=2)
    
    result = linear + grain * mask
    return linear_to_srgb(np.clip(result, 0, 1))

@staticmethod
def spectral_halation(img, threshold=0.85, strength=0.6, color_shift=(1.5, 0.3, 0.2)):
    linear = srgb_to_linear(img)
    luminance = get_luminance(linear)
    
    mask = np.maximum(0, luminance - threshold) / (1 - threshold + 1e-6)
    
    halo_small = cv2.GaussianBlur(mask, (0, 0), sigmaX=8)
    halo_large = cv2.GaussianBlur(mask, (0, 0), sigmaX=20)
    halo = 0.7 * halo_small + 0.3 * halo_large
    
    halo_3ch = np.stack([halo] * 3, axis=2)
    color_weight = np.array([color_shift[2], color_shift[1], color_shift[0]]).reshape(1, 1, 3)
    linear += halo_3ch * color_weight * strength
    
    return linear_to_srgb(np.clip(linear, 0, 1))

@staticmethod
def standard_bloom(img, radius=20, strength=0.3):
    if radius % 2 == 0:
        radius += 1
    
    linear = srgb_to_linear(img)
    blur = srgb_to_linear(cv2.GaussianBlur(img, (0, 0), sigmaX=radius))
    
    result = 1.0 - (1.0 - linear) * (1.0 - blur * strength)
    return linear_to_srgb(np.clip(result, 0, 1))
```

# ==================== 胶片配方库 (带进度回调) ====================

def process_fuji_pro400h(img, grain_scale=1.0, callback=None):
“”“富士 Pro 400H”””
fp = FilmPhysics()

```
if callback: callback('fuji', 30, '应用富士染料特性...')
matrix = [[0.92, 0.06, 0.02], [0.03, 0.95, 0.02], [0.05, 0.08, 0.87]]
img = fp.channel_mix_linear(img, matrix)

if callback: callback('fuji', 50, '调整色调曲线...')
curve_master = [(0, 0), (40, 38), (128, 135), (220, 232), (255, 252)]
img = fp.apply_lut_curve(img, curve_master)

b, g, r = cv2.split(img)
b = fp.apply_lut_curve(b, [(0, 18), (128, 128), (255, 248)])
r = fp.apply_lut_curve(r, [(0, 0), (128, 125), (255, 242)])
img = cv2.merge([b, g, r])

if callback: callback('fuji', 70, '添加柔光效果...')
img = fp.standard_bloom(img, radius=13, strength=0.25)

if callback: callback('fuji', 85, '生成胶片颗粒...')
img = fp.spectral_grain(img, intensity=0.06 * grain_scale, softness=0.7)

return img
```

def process_kodak_portra400(img, grain_scale=1.0, callback=None):
“”“柯达 Portra 400”””
fp = FilmPhysics()

```
if callback: callback('kodak', 30, '应用柯达染料特性...')
matrix = [[0.88, 0.08, 0.04], [0.02, 0.98, 0.00], [0.03, 0.05, 0.92]]
img = fp.channel_mix_linear(img, matrix)

if callback: callback('kodak', 60, '优化肤色表现...')
curve = [(0, 8), (60, 68), (180, 188), (255, 238)]
img = fp.apply_lut_curve(img, curve)

if callback: callback('kodak', 85, '添加细腻颗粒...')
img = fp.spectral_grain(img, intensity=0.04 * grain_scale, softness=0.9)

return img
```

def process_cinestill_800t(img, grain_scale=1.0, callback=None):
“”“Cinestill 800T”””
fp = FilmPhysics()

```
if callback: callback('cinestill', 30, '调整色温平衡...')
b, g, r = cv2.split(img)
b = cv2.addWeighted(b, 1.12, np.zeros_like(b), 0, 0)
r = cv2.addWeighted(r, 0.88, np.zeros_like(r), 0, 0)
img = cv2.merge([b, g, r])

if callback: callback('cinestill', 50, '应用电影曲线...')
curve = [(0, 0), (50, 42), (200, 218), (255, 242)]
img = fp.apply_lut_curve(img, curve)

if callback: callback('cinestill', 70, '生成标志性光晕...')
img = fp.spectral_halation(img, threshold=0.85, strength=0.6, 
                            color_shift=(1.8, 0.4, 0.1))

if callback: callback('cinestill', 85, '添加色彩颗粒...')
img = fp.chromatic_grain(img, intensity=0.08 * grain_scale, color_variance=0.3)

return img
```

def process_polaroid(img, grain_scale=1.0, callback=None):
“”“宝丽来”””
fp = FilmPhysics()

```
if callback: callback('polaroid', 30, '应用宝丽来染料...')
matrix = [[0.85, 0.10, 0.05], [0.05, 0.95, 0.00], [0.00, 0.05, 1.05]]
img = fp.channel_mix_linear(img, matrix)

if callback: callback('polaroid', 50, '调整复古色调...')
curve = [(0, 35), (50, 65), (128, 138), (255, 235)]
img = fp.apply_lut_curve(img, curve)

b, g, r = cv2.split(img)
b = fp.apply_lut_curve(b, [(0, 0), (255, 215)])
img = cv2.merge([b, g, r])

if callback: callback('polaroid', 70, '添加强柔光...')
img = fp.standard_bloom(img, radius=28, strength=0.55)

if callback: callback('polaroid', 85, '生成粗颗粒...')
img = fp.chromatic_grain(img, intensity=0.10 * grain_scale, color_variance=0.35)

return img
```

def process_ilford_hp5(img, grain_scale=1.0, callback=None):
“”“Ilford HP5 Plus”””
fp = FilmPhysics()

```
if callback: callback('ilford', 30, '转换为黑白...')
linear = srgb_to_linear(img)
b, g, r = cv2.split(linear)
gray = 0.10 * b + 0.50 * g + 0.40 * r
gray = np.clip(gray, 0, 1)
gray_srgb = linear_to_srgb(np.stack([gray] * 3, axis=2))

if callback: callback('ilford', 60, '提升对比度...')
curve = [(0, 0), (60, 48), (190, 215), (255, 255)]
gray_srgb = fp.apply_lut_curve(gray_srgb, curve)

if callback: callback('ilford', 85, '添加银盐颗粒...')
gray_srgb = fp.spectral_grain(gray_srgb, intensity=0.15 * grain_scale, softness=0.3)

return gray_srgb
```

def process_ricoh_gr(img, grain_scale=1.0, callback=None):
“”“Ricoh GR”””
fp = FilmPhysics()

```
if callback: callback('ricoh', 30, '应用高反差曲线...')
img = fp.apply_lut_curve(img, [(0, 0), (40, 22), (128, 145), (210, 242), (255, 255)])

if callback: callback('ricoh', 55, '提升饱和度...')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[:, :, 1] *= 1.38
img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

if callback: callback('ricoh', 75, '添加暗角效果...')
h, w = img.shape[:2]
X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
radius = np.sqrt(X ** 2 + Y ** 2)
mask = 1 - np.clip(radius - 0.5, 0, 1) * 0.85
mask = np.stack([mask] * 3, axis=2)
img = (img.astype(np.float32) * mask).astype(np.uint8)

if callback: callback('ricoh', 85, '生成锐利颗粒...')
img = fp.spectral_grain(img, intensity=0.12 * grain_scale, softness=0.0)

return img
```

# ==================== 统一入口 (带进度) ====================

def process_style_v2_with_progress(img, style_name, grain_scale=1.0, callback=None):
“””
主处理函数 (带进度回调)

```
参数:
    img: BGR 图像
    style_name: 胶片风格
    grain_scale: 颗粒强度
    callback: 进度回调函数 callback(step, progress, message)
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
return process_func(img, grain_scale, callback)
```

# ==================== 兼容旧接口 ====================

def process_style_v2(img, style_name, grain_scale=1.0):
“”“无进度回调的版本 (向后兼容)”””
return process_style_v2_with_progress(img, style_name, grain_scale, callback=None)
