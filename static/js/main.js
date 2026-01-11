// ==================== 胶片数据 ====================
const FILMS = {
    "All": [], 
    "Fuji": [{ id: 'fuji', name: 'Pro 400H', img: 'fuji_pro400h.png' }],
    "Kodak": [{ id: 'kodak', name: 'Portra 400', img: 'kodak_portra400.png' }, { id: 'cinestill', name: 'CineStill 800T', img: 'cinestill_800t.png' }],
    "B&W": [{ id: 'ilford', name: 'Ilford HP5', img: 'ilford_hp5.png' }, { id: 'ricoh', name: 'Ricoh GR', img: 'ricoh_gr.png' }],
    "Vintage": [{ id: 'polaroid', name: 'Polaroid', img: 'polaroid.png' }]
};
Object.keys(FILMS).forEach(k => { if(k !== "All") FILMS["All"].push(...FILMS[k]); });

// ★★★ 修改1: 提升至 3200万像素，保证绝大多数照片不缩像素 ★★★
const MAX_UPLOAD_MP = 12;

const app = {
    state: {
        currentBrand: 'All',
        currentFilm: null, 
        params: { opacity: 100, exposure: 0, contrast: 0, highlights: 0, shadows: 0, whites: 0, blacks: 0, temp: 0, tint: 0, saturation: 0, grain: 0, vignette: 0 },
        isProcessed: false,
        isComparing: false
    },
    images: { original: new Image(), film: new Image() },
    socket: null, sessionId: null, gl: null, uploadBlob: null,

    init() {
        this.socket = io(window.location.origin);
        this.setupSocket();
        this.renderTabs();
        this.renderFilmStrip();
        this.setupListeners();
        this.gl = new WebGLProcessor(document.getElementById('glCanvas'));
        this.switchAdjustCat('color');
        
        document.querySelectorAll('input[type=range]').forEach(inp => {
            inp.addEventListener('dblclick', () => {
                inp.value = (inp.dataset.param === 'opacity') ? 100 : 0;
                this.state.params[inp.dataset.param] = parseFloat(inp.value);
                this.updateText(`val_${inp.dataset.param}`, inp.value);
                this.requestRender();
            });
        });
    },

    updateText(id, text) { const el = document.getElementById(id); if(el) el.innerText = text; },

    switchAdjustCat(cat) {
        ['color', 'light', 'tone', 'effects'].forEach(c => {
            document.getElementById(`cat_${c}`).classList.remove('active');
            document.getElementById(`group_${c}`).classList.add('hidden');
        });
        document.getElementById(`cat_${cat}`).classList.add('active');
        document.getElementById(`group_${cat}`).classList.remove('hidden');
    },

    renderTabs() {
        const container = document.getElementById('brandTabs');
        if(!container) return;
        container.innerHTML = '';
        Object.keys(FILMS).forEach(brand => {
            const el = document.createElement('div');
            el.innerText = brand.toUpperCase();
            el.className = `cursor-pointer transition-colors hover:text-white ${this.state.currentBrand === brand ? 'text-white border-b-2 border-accent' : 'text-gray-600'}`;
            el.onclick = () => { this.state.currentBrand = brand; this.renderTabs(); this.renderFilmStrip(); };
            container.appendChild(el);
        });
    },

    renderFilmStrip() {
        const strip = document.getElementById('filmStrip');
        if(!strip) return;
        strip.innerHTML = '';
        const origDiv = document.createElement('div');
        origDiv.className = `film-item flex flex-col items-center gap-2 cursor-pointer shrink-0 ${this.state.currentFilm === null ? 'active' : ''}`;
        origDiv.innerHTML = `<div class="w-16 h-24 flex items-center justify-center relative bg-white/5 border border-white/10 rounded overflow-hidden"><span class="text-[10px] font-bold text-gray-500">NONE</span></div><span class="text-[9px] font-bold text-gray-500 uppercase tracking-wider">Original</span>`;
        origDiv.onclick = () => this.selectFilm(null);
        strip.appendChild(origDiv);

        FILMS[this.state.currentBrand].forEach(f => {
            const active = f.id === this.state.currentFilm;
            const el = document.createElement('div');
            el.className = `film-item flex flex-col items-center gap-2 cursor-pointer shrink-0 ${active ? 'active' : ''}`;
            el.innerHTML = `<div class="w-16 h-24 relative shadow-lg"><img src="/static/films/${f.img}" class="w-full h-full object-cover"></div><span class="text-[9px] font-bold ${active ? 'text-white' : 'text-gray-500'} text-center w-20 leading-tight">${f.name}</span>`;
            el.onclick = () => this.selectFilm(f.id);
            strip.appendChild(el);
        });
    },

    selectFilm(id) {
        if(this.state.currentFilm === id) return;
        this.state.currentFilm = id;
        this.renderFilmStrip();
        if (id === null) {
            this.state.params.opacity = 0;
            this.updateText('val_opacity', '0');
            const opRange = document.querySelector('input[data-param="opacity"]');
            if(opRange) opRange.value = 0;
            this.requestRender();
        } else {
            this.processImage();
        }
    },

    switchMode(mode) {
        const pFilm = document.getElementById('panelPresets');
        const pAdjust = document.getElementById('panelAdjust');
        const indicator = document.getElementById('tabIndicator');
        const btnFilm = document.getElementById('modePresets');
        const btnAdjust = document.getElementById('modeAdjust');

        if(mode === 'adjust') {
            if(!this.images.original.src) return;
            
            pFilm.classList.remove('active');
            pFilm.classList.add('hidden-panel');
            
            pAdjust.classList.remove('hidden-panel');
            pAdjust.classList.add('active');
            
            indicator.style.transform = 'translateX(100%)';
            btnAdjust.classList.replace('text-gray-600', 'text-white');
            btnFilm.classList.replace('text-white', 'text-gray-600');
        } else {
            pAdjust.classList.remove('active');
            pAdjust.classList.add('hidden-panel');
            
            pFilm.classList.remove('hidden-panel');
            pFilm.classList.add('active');
            
            indicator.style.transform = 'translateX(0%)';
            btnFilm.classList.replace('text-gray-600', 'text-white');
            btnAdjust.classList.replace('text-white', 'text-gray-600');
        }
    },

    handleFile(file) {
        if(!file) return;
        document.getElementById('emptyState').classList.add('hidden');
        document.getElementById('glCanvas').classList.remove('opacity-0');
        document.getElementById('fileInfo').classList.remove('hidden');
        document.getElementById('fileInfo').classList.add('flex');
        
        const url = URL.createObjectURL(file);
        this.images.original.src = url;
        this.images.film.src = ""; 

        this.images.original.onload = () => {
            const w = this.images.original.naturalWidth;
            const h = this.images.original.naturalHeight;
            const mp = (w * h / 1e6);
            this.updateText('stat_in', `${w}×${h}`);

            // ★★★ 修改2: 统一使用 Canvas 进行格式转换 (PNG -> JPG 0.8) ★★★
            // 无论图片是否超过阈值，都进行一次 "无损像素 + 0.8质量" 的转换
            // 这样既保证了像素尺寸不变，又大幅降低了文件体积 (MB)
            
            let targetW = w;
            let targetH = h;

            // 只有当图片真的大到离谱 (>3200万像素) 时才缩放像素，防止服务器内存溢出
            if (mp > MAX_UPLOAD_MP) {
                const scale = Math.sqrt(MAX_UPLOAD_MP / mp);
                targetW = Math.floor(w * scale);
                targetH = Math.floor(h * scale);
            }

            const cvs = document.createElement('canvas');
            cvs.width = targetW;
            cvs.height = targetH;
            const ctx = cvs.getContext('2d');
            ctx.drawImage(this.images.original, 0, 0, targetW, targetH);
            
            // 导出为 0.8 质量的 JPEG
            cvs.toBlob(b => { 
                this.uploadBlob = b; 
                const nMp = (targetW*targetH/1e6).toFixed(1);
                
                // 更新 UI
                if (mp > MAX_UPLOAD_MP) {
                    this.updateText('stat_out', `${targetW}×${targetH}`);
                } else {
                    this.updateText('stat_out', "ORIG (HQ)");
                }
                
                // 估算时间 (0.8质量上传很快，但后端处理大像素需要时间，调整系数)
                this.updateText('stat_time', (nMp * 0.8 + 2.0).toFixed(1));
                
            }, 'image/jpeg', 0.8); // <--- 这里设为 0.8

            this.gl.uploadTexture(0, this.images.original);
            this.gl.uploadTexture(1, this.images.original);
            this.state.params.opacity = 0;
            this.requestRender();
            this.state.currentFilm = null;
            this.renderFilmStrip();
            document.getElementById('floatingTools').classList.remove('opacity-0', 'pointer-events-none');
        };
    },

    async processImage() {
        if(!this.images.original.src || !this.state.currentFilm || !this.uploadBlob) return;
        const modal = document.getElementById('progressModal');
        modal.classList.remove('hidden'); modal.classList.add('flex');
        document.getElementById('progressBar').style.width = '0%';
        document.body.style.pointerEvents = 'none';
        
        this.sessionId = 'sess_' + Date.now();
        const fd = new FormData();
        fd.append('file', this.uploadBlob);
        fd.append('style', this.state.currentFilm);
        fd.append('grain', 'off'); 
        fd.append('session_id', this.sessionId);

        try {
            const res = await fetch('/process', { method: 'POST', body: fd });
            if(!res.ok) throw new Error("Error");
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            this.images.film.src = url;
            this.images.film.onload = () => {
                this.state.isProcessed = true;
                this.gl.uploadTexture(1, this.images.film);
                this.state.params.opacity = 100;
                this.updateText('val_opacity', '100');
                const opRange = document.querySelector('input[data-param="opacity"]');
                if(opRange) opRange.value = 100;
                modal.classList.add('hidden'); modal.classList.remove('flex');
                document.body.style.pointerEvents = 'auto';
                this.requestRender();
            };
        } catch (e) {
            alert(e.message); modal.classList.add('hidden'); document.body.style.pointerEvents = 'auto';
        }
    },

    requestRender() {
        if(!this.images.original.src) return;
        const max = 1600;
        let w = this.images.original.naturalWidth, h = this.images.original.naturalHeight;
        if(w > max || h > max) { const s = max / Math.max(w, h); w *= s; h *= s; }
        requestAnimationFrame(() => this.gl.render(w, h, this.state.params, this.state.isComparing));
    },

    download() {
        const btn = document.getElementById('downloadBtn');
        const oldText = btn.innerHTML;
        btn.innerHTML = "SAVING...";
        btn.disabled = true;

        setTimeout(() => {
            const w = this.images.original.naturalWidth;
            const h = this.images.original.naturalHeight;
            this.gl.render(w, h, this.state.params, false);
            this.gl.canvas.toBlob((blob) => {
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                const fname = this.state.currentFilm ? this.state.currentFilm.toUpperCase() : "EDIT";
                a.download = `FILM_${fname}_${Date.now()}.jpg`;
                a.click();
                this.requestRender();
                btn.innerHTML = oldText;
                btn.disabled = false;
            }, 'image/jpeg', 0.95);
        }, 50);
    },

    setupListeners() {
        document.getElementById('fileInput').addEventListener('change', (e) => this.handleFile(e.target.files[0]));
        document.querySelectorAll('input[type=range]').forEach(inp => {
            inp.addEventListener('input', (e) => {
                const key = e.target.dataset.param;
                this.state.params[key] = parseFloat(e.target.value);
                this.updateText(`val_${key}`, e.target.value);
                this.requestRender();
            });
        });
        const btn = document.getElementById('compareBtn');
        const start = (e) => { e.preventDefault(); this.state.isComparing = true; this.requestRender(); document.getElementById('originalLabel').classList.remove('opacity-0'); };
        const end = (e) => { e.preventDefault(); this.state.isComparing = false; this.requestRender(); document.getElementById('originalLabel').classList.add('opacity-0'); };
        ['mousedown', 'touchstart'].forEach(ev => btn.addEventListener(ev, start));
        ['mouseup', 'mouseleave', 'touchend'].forEach(ev => btn.addEventListener(ev, end));
    },

    setupSocket() {
        this.socket.on('connect', () => document.getElementById('statusDot').classList.replace('bg-gray-600', 'bg-green-500'));
        this.socket.on('disconnect', () => document.getElementById('statusDot').classList.replace('bg-green-500', 'bg-red-500'));
        this.socket.on('progress_update', (d) => {
            if(d.session_id === this.sessionId) {
                document.getElementById('progressBar').style.width = d.progress + '%';
                document.getElementById('progressPercent').innerText = Math.round(d.progress) + '%';
                document.getElementById('progressStep').innerText = d.message;
            }
        });
    }
};

window.onload = () => app.init();