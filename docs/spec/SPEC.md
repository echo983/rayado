# SPEC v0.3 — 两阶段转写与逻辑建模

## 业务目标（必须牢记）
- 输出主要供 LLM 消费（理解/检索/摘要/问答），而非人工逐字校对。
- 成本敏感：最大化提示词缓存命中与幂等复用，避免重复 token 消耗。
- 非人工干预：允许不确定性与瑕疵，结果可回放与可解释即可。
- 吞吐优先：并发高、流程稳定，单遍完成（仅允许一次重试）。

## 目标与约束
- 仅处理本地媒体文件，依赖 `ffmpeg` 提取带时间轴的音轨。
- 仅输出 SRT（Phase 1）。Phase 2 输出对象图 `.txt` 与清洗 SRT。
- 语言自动识别；多语言混杂不处理，效果不佳视为成本。
- 全流程只跑一遍，每个阶段允许一次重试。
- 并发可配置，默认 64（Nova‑2 预录音并发上限 100）。
- 严格无人干预：不要求输入端指定语言。

## 总体流程
### Phase 1 — Transcription
1) **AudioPrep**：抽取并转码音轨为 16kHz mono WAV。
2) **VAD Chunking**：基于静音检测生成语音段；按目标长度合并。
   - `target-sec=20`，`max-sec=35`
3) **LID**：VoxLingua107（SpeechBrain）对 5 个采样段投票确定主语言。
4) **ASR**：使用确定语言并行调用 Deepgram（Nova‑2），每段仅重试一次。
5) **Render**：生成标准 SRT（仅这一份输出文件）。

### Phase 2 — Logic Modeling
1) 加载 `prompts/SORAL.txt`。
2) 将 Phase 1 SRT 直接喂给 GPT‑5.1，输出对象关系图 `.txt`。
3) 载入对象图（支持外部文件，便于跨集复用）。
4) 如果提供外部对象图：将其作为“已存在图”，本次输出为追加内容；合并后生成新的图文件。

### Phase 3 — Graph Merge
1) 加载 `prompts/SORAL_Merge.txt`。
2) 输入两份 S-ORAL v3.1 对象图（A/B）。
3) GPT‑5.1 输出合并重构后的单一对象图（`.merged.graph.txt`）。

## VAD 与切片规则
- 基于 `silencedetect` 生成静音区间，反推出语音段。
- 过滤过短语音段：小于 `vad_min_speech_sec` 直接丢弃。
- 相邻语音段合并：静音间隔小于 `vad_merge_gap_sec` 则合并。
- 合并段再按 `target-sec/max-sec` 约束生成最终 chunk。

## 语言识别规则
- 采样：头、25%、50%、75%、尾共 5 个 chunk。
- 使用 VoxLingua107 评分投票，权重为模型得分。
- 得出单一主语言，后续 ASR 固定该语言。

## OpenAI 调用规则
- 统一使用 Responses API（streaming）。
- `max_output_tokens=128000`。
- `timeout=15min`。

## 输出
- Phase 1：`out/<base>.srt`
- Phase 2：`out/<base>.graph.txt`
