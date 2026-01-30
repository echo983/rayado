# SPEC v0.2 — CLI 转写系统方案

## 业务目标（必须牢记）
- 输出主要供 LLM 消费（理解/检索/摘要/问答），而非人工逐字校对。
- 成本敏感：最大化提示词缓存命中与幂等复用，避免重复 token 消耗。
- 非人工干预：允许不确定性与瑕疵，结果可回放与可解释即可。
- 吞吐优先：并发高、流程稳定，单遍完成（仅允许一次重试）。

## 目标与约束
- 仅处理本地媒体文件，依赖 `ffmpeg` 提取带时间轴的音轨；支持 `ffmpeg` 可解析的格式。
- 仅输出纯文本与 SRT。
- 语言自动识别；多语言混杂不处理，效果不佳视为成本。
- 偏吞吐；全流程只跑一遍，但允许在这一遍内做质量修正。
- 单 chunk 失败只重试一次；再次失败则整体失败，不降级。
- 涉及 token 成本的工作尽量幂等；失败重跑不重复花费。
- 并发可配置，默认 64（Nova‑2 预录音并发上限 100）；被动受服务限制，不主动自适配。
- 必须有 GCL（append-only）；即使流程简单也要写最小 GCL。
- 单机提示词缓存。
- 允许直接跳过静音/噪音区间，宁可跳过省成本。

## 总体流程
1) AudioPrep：用 `ffmpeg` 提取音轨（16kHz mono wav），落盘 `audio.wav`。
2) VAD 预扫：对 `audio.wav` 进行低成本检测，生成 speech mask（不走 ASR，不产生成本）。
3) ChunkPlan：按固定时间切片（默认 120s，左右各 1.5s overlap，步长 117s），仅在 speech mask 覆盖区域内生成 chunk。
4) ASRPool：按并发上限发送请求；每个 chunk 只允许重试一次。
5) GCLWriter：统一追加写入 GCL（append-only）。
6) CoherenceJobs（同语种增强，自动可撤销）：
   - OverlapJudge：处理重叠重复与冲突。
   - EntityBuild：自动实体抽取与别名聚合。
   - PronounResolve：指代修复（低成本、低侵入）。
   - SpeakerDiarization：说话人分离并关联实体。
   - SceneJobs：轻量场景切分，便于 LLM 消费。
7) RenderExport：导出纯文本与 SRT。

## VAD 与切片规则（确定性）
- speech mask 由 VAD 得到的时间段集合，默认阈值保守，偏向跳过。
- 过滤过短语音段：小于 `vad_min_speech_sec` 的片段直接丢弃。
- 相邻语音段合并：当静音间隔小于 `vad_merge_gap_sec` 时合并为一个片段。
- 切片生成规则：
  - 仅在 speech mask 覆盖区间内生成 chunk。
  - chunk 可跨越短静音间隔（合并后视为连续）。
  - 对被跳过区间写 `GCL_CHUNK`，标记 `skip_reason=non_speech`。

## OverlapJudge 规则（默认）
- 候选条件：相邻 chunk 的 overlap 区间内，span 时间 IoU ≥ `overlap_iou_min`。
- 相似度：文本 token Jaccard 或编辑距离归一化 ≥ `overlap_sim_min`。
- 决策优先级：
  1) 更高 `asr_conf`
  2) 时间更贴近 overlap 中心
  3) 同置信度时保留右侧 chunk（时间靠后）
- 输出：写 `GCL_OVERLAP`，必要时追加 `GCL_OVERRIDE policy=suppress/merge`。
### 实现状态
- 里程碑 2 已实现基础版本：时间 IoU + 文本 Jaccard，置信度低者被 suppress。

## 说话人 / 实体 / 场景的默认策略（可调整）
- EntityBuild：
  - 仅提升高频且上下文稳定的实体为全局实体。
  - 低置信实体不强行合并，允许并存。
  - 基础实现：基于词面频次做 `GCL_ENTITY` / `GCL_MENTION`。
 - SpeakerDiarization：
  - 仅输出本集说话人簇与置信度；跨集关联通过 `GCL_SPEAKER_MAP` 追加。
  - 基础实现：从 ASR words 中读取 `speaker`，为每个 span 写 `GCL_SPEAKER_MAP`。
  - 默认关闭，避免碎片化 speaker id；需要时手动开启。
- SceneJobs：
  - 以时间窗或转场特征做粗粒度场景划分；不追求精细。

## CLI 设计（建议）
- `rayado transcribe <input>`
  - `--out <dir>` 输出目录（默认 `./out/<basename>`）
  - `--concurrency <n>` 并发（默认 128）
  - `--chunk-sec <n>` 默认 25
  - `--overlap-sec <n>` 默认 1.5
  - `--retry <n>` 默认 1（硬上限 1）
  - `--asr-provider <name>` 服务商标识
  - `--cache-dir <path>` 单机提示词缓存目录
  - `--vad <name>` VAD 实现标识（默认启用）
  - `--vad-threshold <n>` VAD 阈值（默认保守，偏向跳过）
  - `--vad-min-speech-sec <n>` 最短语音段阈值
  - `--vad-merge-gap-sec <n>` 静音合并阈值
  - `--vad-pad-sec <n>` 语音段前后扩展时间

## GCL 最小字段集（必需）
- `GCL_HDR`：版本、模式（append_only）、time_unit。
- `GCL_CHUNK`：chunk_id, t0, t1, overlap_left, overlap_right（可含 `skip_reason`）。
- `GCL_SPAN`：sid, t0, t1, chunk_id, text_raw, asr_conf。
- `GCL_OVERLAP`：裁决信息（可选）。
- `GCL_OVERRIDE`：用于抑制/合并（可选）。

### 语言检测记录（Deepgram）
- 当启用 `detect_language` 时，追加一条 `GCL_OVERRIDE` 记录，用于标记该 chunk 的检测语言：
  - `policy=detected_language`
  - `deps=<language_code>`（如 `es`）
  - `conf=<language_confidence>`（可空）
  - `sid` 取该 chunk 的首个 `GCL_SPAN.sid`

## GCL 增强字段（LLM 友好）
- `GCL_ENTITY`：全局实体条目（自动抽取、可撤销）。
- `GCL_MENTION`：实体与 span 关联。
- `GCL_ENTITY_MAP`：临时实体到全局实体的映射（自动可更新）。
- `GCL_SPEAKER`：说话人条目与置信度。
- `GCL_SPEAKER_MAP`：speaker → entity 映射。
- `GCL_SCENE`：场景片段。
- `GCL_SCENE_SPAN`：场景与 span 关联。

## Series / Episode 文件落盘
- Series Global State：`out/series/<series_id>/series.gcl`
  - 仅保存跨集共享对象（实体、术语、映射、状态）。
- Episode Local Log：`out/<basename>/episode.gcl`
  - 保存本集所有事件（chunk/span/overlap/override/speaker/scene）。
- 消费顺序固定：Series → Episode → entity_map → Render。

## 幂等与成本控制
- 幂等键由 `(input_hash, chunk_id, asr_model, asr_params)` 组成。
- 命中缓存时跳过 ASR 请求，直接复用结果。
- 重试仅针对失败 chunk，最多一次；请求体与提示词参数保持不变。
- 已存在的 `GCL_CHUNK/GCL_SPAN` 不重复请求。

## 缓存落盘（单机）
- 推荐 sqlite：表字段包含 `key, request_hash, response, created_at`。
- `key` 采用幂等键串联；`request_hash` 用于防止参数漂移。
- 命中即复用，不触发请求；失败不写入缓存。

## 输出
- 纯文本：按 `sid` 顺序拼接，跳过被 suppress 的 span；若有 speaker 映射，前缀 `Speaker_X:`。
- SRT：每 span 一条，使用 `t0/t1`；若有 speaker 映射，前缀 `Speaker_X:`。
- 运行统计：写入 `run.log`（JSON），包含耗时、chunk 数、跳过/失败数等。

## SRT 时间格式
- time_unit 默认秒（浮点）。
- SRT 时间格式化为 `HH:MM:SS,mmm`，按毫秒四舍五入。
- 保证 `end >= start`，若相等则 end += 1ms。

## 失败策略
- 单 chunk 失败重试一次。
- 仍失败则终止整个任务并标记失败，保留已有 GCL 以便复盘。
- 终止策略：并行模式下遇到失败则取消剩余任务并退出（fail-fast）。

## 目录建议
- `out/<basename>/`
  - `episode.gcl`
  - `transcript.txt`
  - `subtitles.srt`
