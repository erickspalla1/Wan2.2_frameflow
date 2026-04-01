# FrameFlow Engine — Roadmap de Testes no ComfyUI

Data: 2026-04-01
Status: Pendente — requer Pod temporário no RunPod com GPU

---

## Objetivo

Testar e validar features que dependem do ComfyUI visual antes de integrar no handler.py. Cada teste gera (ou não) um JSON exportado que será integrado como injection layer no motor.

---

## Fase 1: Setup do Pod

### 1.1 Subir Pod temporário no RunPod

- **Template:** RunPod PyTorch 2.1 ou similar com CUDA 12.1
- **GPU:** A100 SXM 80GB (mesma que o serverless vai usar)
- **Network Volume:** `frameflow-models` (US-KS-2) — já tem os modelos I2V/Fun Control/CLIP/VAE/LoRA
- **Disk:** 50GB container volume (pra ComfyUI + custom nodes)
- **Expose:** Porta 8188 (ComfyUI web UI)

### 1.2 Instalar ComfyUI no Pod

```bash
# Clonar ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/comfyui
cd /workspace/comfyui
pip install -r requirements.txt

# Linkar modelos do network volume
ln -sf /runpod-volume/models/diffusion_models/* /workspace/comfyui/models/diffusion_models/
ln -sf /runpod-volume/models/clip/* /workspace/comfyui/models/clip/
ln -sf /runpod-volume/models/vae/* /workspace/comfyui/models/vae/
ln -sf /runpod-volume/models/loras/* /workspace/comfyui/models/loras/
ln -sf /runpod-volume/models/upscale_models/* /workspace/comfyui/models/upscale_models/

# Iniciar ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

### 1.3 Instalar custom nodes existentes (mesmos do Dockerfile)

```bash
cd /workspace/comfyui/custom_nodes
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
git clone https://github.com/Jonseed/ComfyUI-Image-Filters.git
git clone https://github.com/EllangoK/ComfyUI-post-processing-nodes.git
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# Instalar deps de cada um
for dir in */; do
    [ -f "$dir/requirements.txt" ] && pip install -r "$dir/requirements.txt" || true
done
```

### 1.4 Instalar custom nodes NOVOS (a serem testados)

```bash
cd /workspace/comfyui/custom_nodes

# RIFE Frame Interpolation (16fps → 24/30fps)
git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git
pip install -r ComfyUI-Frame-Interpolation/requirements.txt || true

# Face Restore (GFPGAN / CodeFormer)
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
pip install -r ComfyUI-Impact-Pack/requirements.txt || true

# Motion Deblur (ConvIR)
git clone https://github.com/FuouM/FM_nodes.git
pip install -r FM_nodes/requirements.txt || true
```

### 1.5 Baixar modelos novos

```bash
# RIFE (frame interpolation) — baixa automaticamente pelo node, mas se precisar manual:
mkdir -p /workspace/comfyui/models/rife
# O ComfyUI-Frame-Interpolation geralmente baixa o modelo sozinho no primeiro uso

# Face Restore
mkdir -p /workspace/comfyui/models/facerestore_models
wget -O /workspace/comfyui/models/facerestore_models/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
wget -O /workspace/comfyui/models/facerestore_models/codeformer-v0.1.0.pth \
    https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth

# Motion Deblur (ConvIR)
mkdir -p /workspace/comfyui/models/convir
# Verificar URL exata no repo FM_nodes — pode baixar automaticamente
```

### 1.6 Reiniciar ComfyUI

```bash
# Matar o processo anterior
pkill -f "python main.py"

# Reiniciar pra carregar custom nodes novos
cd /workspace/comfyui
python main.py --listen 0.0.0.0 --port 8188
```

**Checkpoint:** Abrir no browser, verificar que todos os custom nodes aparecem na barra de busca de nodes (RIFE VFI, FaceRestoreWithModel, ConvIR_MotionDeBlur).

---

## Fase 2: Testes Isolados

### Teste 2.1 — RIFE Frame Interpolation

**Objetivo:** Confirmar que RIFE interpola 16fps → 24fps com qualidade.

**Workflow no ComfyUI:**
```
LoadVideo (qualquer vídeo curto 16fps)
  → RIFE VFI (multiplier: 2)
    → SaveVideo
```

**Verificar:**
- [ ] O node RIFE VFI aparece e conecta sem erro?
- [ ] Output tem o dobro de frames do input?
- [ ] Qualidade visual: frames intermediários parecem naturais?
- [ ] Tempo de processamento: ___s para 81 frames 720p
- [ ] VRAM usada: ___GB

**Se funcionar:** Exportar como `rife_interpolation_chain.json` (File → Export API)

**Variações a testar:**
- multiplier: 2 (16→32fps, entregar a 24fps)
- multiplier: 1.5 (16→24fps direto)
- Testar com frames 480p, 720p, 1080p

---

### Teste 2.2 — Face Restore

**Objetivo:** Confirmar que FaceRestoreWithModel melhora rostos em frames de vídeo.

**Workflow no ComfyUI:**
```
LoadImage (frame com rosto gerado pelo Wan 2.2)
  → FaceRestoreWithModel (model: GFPGANv1.4 ou CodeFormer)
    → SaveImage
```

**Verificar:**
- [ ] O node FaceRestoreWithModel aparece? (vem do Impact-Pack)
- [ ] Aceita batch de imagens (pra aplicar em todos os frames)?
- [ ] Melhora rostos sem artefatos?
- [ ] Tempo: ___ms por frame
- [ ] Se não suporta batch: testar com VHS_BatchManager ou loop manual

**Se funcionar:** Exportar como `face_restore_chain.json`

**NOTA:** Se FaceRestore não suportar batch nativamente, a alternativa é processar frame a frame no handler.py (extract frames → restore cada → re-encode). Mais lento mas funciona.

---

### Teste 2.3 — Motion Deblur (ConvIR)

**Objetivo:** Confirmar que ConvIR remove motion blur de frames de vídeo.

**Workflow no ComfyUI:**
```
LoadImage (frame com motion blur)
  → ConvIR_MotionDeBlur (model: convir_gopro.pkl)
    → SaveImage
```

**Verificar:**
- [ ] O node ConvIR_MotionDeBlur aparece?
- [ ] Aceita batch de imagens?
- [ ] Resultado visual: motion blur reduzido sem artefatos?
- [ ] Tempo: ___ms por frame
- [ ] VRAM: ___GB

**Se funcionar:** Exportar como `deblur_chain.json`

---

### Teste 2.4 — IP Adapter com Wan 2.2

**Objetivo:** Confirmar que o IPAdapter funciona com o UNet do Wan 2.2.

**Este é o teste mais importante.** Se falhar, precisamos de outra abordagem.

**Workflow no ComfyUI:**
```
UNETLoader (wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors)
  → IPAdapterUnifiedLoaderCommunity (preset: "Composition")
    → IPAdapter (weight: 0.6, image: LoadImage de referência)
      → ModelSamplingSD3 (shift: 5.0)
        → KSamplerAdvanced (... resto do pipeline I2V)
```

**Verificar:**
- [ ] IPAdapterUnifiedLoaderCommunity aceita o model type do Wan 2.2?
- [ ] Se crashar: qual o erro? (model type incompatível? CLIP Vision não encontrado?)
- [ ] Se funcionar: o vídeo gerado mantém identidade visual da referência?
- [ ] Comparar: mesmo prompt COM IP Adapter vs SEM — diferença perceptível?

**Se NÃO funcionar:**
- [ ] Testar com Kijai WanVideoWrapper — pode ter suporte nativo a IP Adapter
- [ ] Testar injetando ANTES do LoraLoaderModelOnly (na chain I2V com LightX2V)
- [ ] Documentar o erro exato pra decidir próximos passos

**Se funcionar:** Confirmar que a injeção dinâmica do handler.py produz o mesmo resultado. Se não, exportar workflow correto.

---

### Teste 2.5 — Depth+Canny Blend como Control Video

**Objetivo:** Confirmar que o blend que fazemos no handler (np.maximum(depth*0.6, canny)) funciona como control_video no Fun Control.

**Passos:**
1. Carregar uma imagem no ComfyUI
2. Rodar DepthAnythingV2Preprocessor → salvar depth map
3. Rodar CannyEdgePreprocessor → salvar canny map
4. Manualmente blendar (pode usar node ImageBlend ou fazer offline)
5. Criar vídeo estático do blend (repetir frame N vezes)
6. Usar como control_video no Wan22FunControlToVideo
7. Comparar resultado com depth-only control video

**Verificar:**
- [ ] Fun Control aceita o blend sem erro?
- [ ] Resultado melhor que depth puro? (silhueta + bordas preservadas)
- [ ] Resultado pior? (artefatos, confusão do modelo)
- [ ] Proporção do blend: 60/40 tá bom ou precisa ajustar?

---

## Fase 3: Testes no Pipeline Completo

### Teste 3.1 — Pipeline I2V completo com RIFE pós-processamento

```
LoadImage → WanImageToVideo → LoRA high → KSampler HIGH → KSampler LOW
  → VAEDecode → RIFE VFI (2x) → CreateVideo (fps: 32) → SaveVideo
```

**Verificar:**
- [ ] RIFE funciona depois do VAEDecode? (frames são tensores, não imagens RGB)
- [ ] Se precisar de conversão: quais nodes intermediários?
- [ ] FPS final do vídeo: ___fps
- [ ] Tempo total pipeline: ___s
- [ ] VRAM pico: ___GB

---

### Teste 3.2 — Pipeline completo com deblur + sharpen + RIFE

Cadeia máxima de pós-processamento:

```
VAEDecode
  → ConvIR_MotionDeBlur
    → Sharpen (alpha: 0.3, radius: 2)
      → FilmGrain (intensity: 0.08)
        → RIFE VFI (2x)
          → CreateVideo → SaveVideo
```

**Verificar:**
- [ ] A cadeia completa roda sem erro de tipo/conexão?
- [ ] Ordem importa? (deblur antes de sharpen? RIFE antes ou depois de grain?)
- [ ] Tempo total: ___s
- [ ] Qualidade: o resultado é perceptivelmente melhor?

---

### Teste 3.3 — Fun Control + IP Adapter + Pós-processamento

O teste mais pesado — todos os layers ativos:

```
LoadImage (referência)
  → IPAdapter → UNet Fun Control HIGH
LoadImage (first frame)
  → DepthAnythingV2 + Canny → blend → control video
  → Wan22FunControlToVideo
    → KSampler HIGH → KSampler LOW → VAEDecode
      → ConvIR_MotionDeBlur → Sharpen → RIFE VFI → CreateVideo → SaveVideo
```

**Verificar:**
- [ ] Roda sem OOM? (VRAM pico: ___GB)
- [ ] Tempo total: ___s
- [ ] Qualidade: identidade preservada + movimento suave + sem blur?

---

## Fase 4: Medições

### Tabela de performance

| Componente | 480p (81 frames) | 720p (81 frames) | 1080p (81 frames) |
|---|---|---|---|
| Wan 2.2 I2V (4 steps) | ___s | ___s | ___s |
| Wan 2.2 Fun Control (20 steps) | ___s | ___s | ___s |
| DepthAnythingV2 preprocessor | ___s | ___s | ___s |
| Canny preprocessor | ___s | ___s | ___s |
| ConvIR_MotionDeBlur | ___s | ___s | ___s |
| Sharpen (81 frames) | ___s | ___s | ___s |
| RIFE VFI 2x (81→162 frames) | ___s | ___s | ___s |
| FaceRestore (81 frames) | ___s | ___s | ___s |
| RealESRGAN x4 (81 frames) | ___s | ___s | ___s |

### Tabela de VRAM

| Combinação | VRAM Pico |
|---|---|
| I2V (LoRA 4 steps) 720p | ___GB |
| Fun Control (20 steps) 720p | ___GB |
| Fun Control + IP Adapter 720p | ___GB |
| Fun Control + IP Adapter + RIFE + deblur 720p | ___GB |
| Fun Control + IP Adapter + RIFE + deblur + upscale 1080p | ___GB |

---

## Fase 5: Exportações

### JSONs a exportar (se testes passarem)

| Arquivo | Condição |
|---|---|
| `rife_interpolation_chain.json` | Se RIFE VFI funcionar no pipeline |
| `face_restore_chain.json` | Se FaceRestore funcionar com batch |
| `deblur_chain.json` | Se ConvIR funcionar com batch |
| `deblur_sharpen_chain.json` | Chain combinada: deblur → sharpen |

### Formato de exportação

No ComfyUI: **File → Export (API Format)** — gera JSON com node IDs reais.

**IMPORTANTE:** Anotar os node IDs exatos de cada JSON exportado. O handler.py precisa deles pra injeção dinâmica.

---

## Fase 6: Atualizar Código (depois dos testes)

### Dockerfile — adicionar custom nodes novos

```dockerfile
# Frame Interpolation (RIFE)
RUN cd /app/comfyui/custom_nodes && git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git || true
RUN cd /app/comfyui/custom_nodes/ComfyUI-Frame-Interpolation && pip3 install --no-cache-dir -r requirements.txt || true

# Face Restore (Impact Pack)
RUN cd /app/comfyui/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git || true
RUN cd /app/comfyui/custom_nodes/ComfyUI-Impact-Pack && pip3 install --no-cache-dir -r requirements.txt || true

# Motion Deblur (ConvIR)
RUN cd /app/comfyui/custom_nodes && git clone https://github.com/FuouM/FM_nodes.git || true
RUN cd /app/comfyui/custom_nodes/FM_nodes && pip3 install --no-cache-dir -r requirements.txt || true
```

### start.sh — adicionar modelos novos

```bash
# Face Restore model
FACERESTORE_DIR="/runpod-volume/models/facerestore"
download_model "facerestore" "GFPGANv1.4.pth" "$FACERESTORE_DIR/GFPGANv1.4.pth"

# Motion Deblur model
DEBLUR_DIR="/runpod-volume/models/convir"
download_model "convir" "convir_gopro.pkl" "$DEBLUR_DIR/convir_gopro.pkl"

# RIFE models (geralmente baixam automaticamente, mas garantir)
# Verificar path exato após teste no Pod
```

### handler.py — novos injection layers

```python
# Novos params na API:
"frame_interpolation": false,    # RIFE 2x (16fps → 32fps)
"face_restore": false,           # GFPGAN face restoration
"deblur_strength": 0.0,          # ConvIR motion deblur (0 = off)
"sharpen_strength": 0.0,         # Unsharp mask (0 = off)

# Cadeia de pós-processamento atualizada:
# VAEDecode → deblur → sharpen → face_restore → film_grain → chroma → vignette → color → RIFE → upscale → CreateVideo
```

---

## Decisões Pendentes (dependem dos testes)

| Decisão | Depende de |
|---|---|
| IP Adapter fica ou sai? | Teste 2.4 — se crashar com Wan 2.2, remover |
| RIFE vai no preset "product"? | Performance — se < 10s, sim |
| FaceRestore vai como default? | Qualidade — se melhorar sem artefatos, sim pra personagens |
| Deblur vai como default pra produto? | Qualidade — se melhorar rotação, sim |
| Blend depth+canny funciona? | Teste 2.5 — se pior que depth puro, voltar pra depth |
| Ordem da cadeia de pós-processamento | Teste 3.2 — medir diferentes ordens |

---

## Custos estimados do Pod

- A100 80GB Pod: ~$1.50-2.00/hora no RunPod
- Tempo estimado pra todos os testes: 3-5 horas
- Custo total: ~$5-10
- **Depois dos testes: desligar o Pod.** Os modelos ficam no network volume.
