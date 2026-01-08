class WebGLProcessor {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true });
        if (!this.gl) return;
        this.program = this.createProgram(this.vsSource, this.fsSource);
        this.initBuffers();
        this.initTextures();
        this.cacheUniforms();
    }

    vsSource = `#version 300 es
        in vec2 a_position;
        out vec2 v_texCoord;
        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
            v_texCoord = a_position * 0.5 + 0.5;
            v_texCoord.y = 1.0 - v_texCoord.y; 
        }`;

    // === v11.0 Shader: 修复变量名 ===
    fsSource = `#version 300 es
        precision highp float;
        
        uniform sampler2D u_image0; 
        uniform sampler2D u_image1; 
        
        uniform float u_opacity;
        uniform float u_exposure;
        uniform float u_contrast;
        uniform float u_highlights;
        uniform float u_shadows;
        uniform float u_whites;
        uniform float u_blacks;
        uniform float u_temp;
        uniform float u_tint;
        uniform float u_saturation; // 之前漏了这里导致失效
        uniform float u_grain;
        uniform float u_vignette;
        uniform float u_rand;
        
        uniform bool u_is_compare;

        in vec2 v_texCoord;
        out vec4 outColor;

        // 统一命名为 luminance
        float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

        float rand(vec2 co) {
            return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
        }

        vec3 applyTemp(vec3 color, float temp, float tint) {
            vec3 res = color;
            res.r += temp * 0.15; res.b -= temp * 0.15; res.g += tint * 0.15;
            return res;
        }

        void main() {
            vec4 orig = texture(u_image0, v_texCoord);
            if (u_is_compare) { outColor = orig; return; }

            vec4 film = texture(u_image1, v_texCoord);
            vec3 color = mix(orig.rgb, film.rgb, u_opacity);

            // 1. Exposure
            color *= pow(2.0, u_exposure);

            // 2. WB
            color = applyTemp(color, u_temp, u_tint);

            // 3. Contrast
            color = (color - 0.5) * (u_contrast + 1.0) + 0.5;

            // 4. Tone Mapping
            float luma = luminance(color);
            float sm = 1.0 - smoothstep(0.0, 0.5, luma);
            if(u_shadows > 0.0) color = mix(color, pow(color, vec3(1.0 - u_shadows * 0.4)), sm);
            else color = mix(color, color * (1.0 + u_shadows * 0.5), sm);
            
            float hm = smoothstep(0.5, 1.0, luma);
            if(u_highlights < 0.0) color = mix(color, color * (1.0 + u_highlights * 0.4), hm);
            else color = mix(color, 1.0 - (1.0 - color) * (1.0 - u_highlights * 0.4), hm);

            // 5. Levels
            float wp = 1.0 - u_whites * 0.15;
            float bp = u_blacks * 0.15;
            if(wp <= bp) wp = bp + 0.001;
            color = (color - bp) / (wp - bp);

            // 6. Saturation (修复！)
            // 必须重新计算 luma 因为前面的操作改变了颜色
            float satLuma = luminance(color);
            color = mix(vec3(satLuma), color, 1.0 + u_saturation);

            // 7. Grain
            if (u_grain > 0.0) {
                float noise = rand(v_texCoord * 2.0 + u_rand); 
                float grainStrength = u_grain * 0.4 * (1.0 - luma * 0.6); // 稍微降低亮部颗粒
                
                // Overlay Blend for Grain
                vec3 gColor = vec3(noise);
                vec3 finalG = vec3(0.0);
                // 简化版 Overlay
                finalG = mix(2.0 * color * gColor, 1.0 - 2.0 * (1.0 - color) * (1.0 - gColor), step(0.5, color));
                
                color = mix(color, finalG, grainStrength * 0.5); // 0.5 混合强度，防止颗粒太燥
            }

            // 8. Vignette
            if (u_vignette > 0.0) {
                float dist = distance(v_texCoord, vec2(0.5));
                color = mix(color, vec3(0.0), smoothstep(0.4, 1.0, dist) * u_vignette);
            }

            outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
        }`;

    createProgram(vs, fs) { const p = this.gl.createProgram(); const v = this.compile(this.gl.VERTEX_SHADER, vs); const f = this.compile(this.gl.FRAGMENT_SHADER, fs); this.gl.attachShader(p, v); this.gl.attachShader(p, f); this.gl.linkProgram(p); return p; }
    compile(t, s) { const h = this.gl.createShader(t); this.gl.shaderSource(h, s); this.gl.compileShader(h); return h; }
    initBuffers() { const p = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]); const b = this.gl.createBuffer(); this.gl.bindBuffer(this.gl.ARRAY_BUFFER, b); this.gl.bufferData(this.gl.ARRAY_BUFFER, p, this.gl.STATIC_DRAW); const l = this.gl.getAttribLocation(this.program, "a_position"); this.gl.enableVertexAttribArray(l); this.gl.vertexAttribPointer(l, 2, this.gl.FLOAT, false, 0, 0); }
    initTextures() { this.tex0 = this.createTex(); this.tex1 = this.createTex(); }
    createTex() { const t = this.gl.createTexture(); this.gl.bindTexture(this.gl.TEXTURE_2D, t); this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE); this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE); this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR); this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR); return t; }

    cacheUniforms() {
        this.u = {};
        const names = ['u_image0', 'u_image1', 'u_opacity', 'u_exposure', 'u_contrast', 
                       'u_highlights', 'u_shadows', 'u_whites', 'u_blacks', 'u_temp', 'u_tint', 
                       'u_saturation', 'u_grain', 'u_vignette', 'u_rand', 'u_is_compare'];
        names.forEach(n => this.u[n] = this.gl.getUniformLocation(this.program, n));
    }

    uploadTexture(u, i) {
        this.gl.activeTexture(u===0?this.gl.TEXTURE0:this.gl.TEXTURE1);
        this.gl.bindTexture(this.gl.TEXTURE_2D, u===0?this.tex0:this.tex1);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, i);
    }

    render(w, h, p, cmp=false) {
        this.canvas.width = w; this.canvas.height = h;
        this.gl.viewport(0, 0, w, h);
        this.gl.useProgram(this.program);
        this.gl.uniform1i(this.u['u_image0'], 0);
        this.gl.uniform1i(this.u['u_image1'], 1);
        this.gl.uniform1f(this.u['u_opacity'], p.opacity/100);
        this.gl.uniform1f(this.u['u_exposure'], p.exposure/50);
        this.gl.uniform1f(this.u['u_contrast'], p.contrast/100);
        this.gl.uniform1f(this.u['u_highlights'], p.highlights/100);
        this.gl.uniform1f(this.u['u_shadows'], p.shadows/100);
        this.gl.uniform1f(this.u['u_whites'], p.whites/100);
        this.gl.uniform1f(this.u['u_blacks'], p.blacks/100);
        this.gl.uniform1f(this.u['u_temp'], p.temp/100);
        this.gl.uniform1f(this.u['u_tint'], p.tint/100);
        this.gl.uniform1f(this.u['u_saturation'], p.saturation/100);
        this.gl.uniform1f(this.u['u_grain'], p.grain/100);
        this.gl.uniform1f(this.u['u_vignette'], p.vignette/100);
        this.gl.uniform1f(this.u['u_rand'], 123.45); // 固定种子
        this.gl.uniform1i(this.u['u_is_compare'], cmp?1:0);
        this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);
    }
}