# CLI 约定（重构）

## 命令
```
rayado phase1 <input>
rayado phase2 <srt>
```

## Phase 1 参数（转写）
- `--out <path>` 输出 SRT（默认 `./out/<basename>.srt`）
- `--concurrency <n>` 并发（默认 64）
- `--retry <n>` 默认 1（硬上限 1）
- `--deepgram-model <name>` Deepgram 模型（默认 `nova-2`）
- `--cache-dir <path>` 缓存目录（默认 `./.cache/rayado`）
- `--vad-threshold <n>` VAD 阈值（默认 -30）
- `--vad-min-speech-sec <n>` 最短语音段阈值
- `--vad-merge-gap-sec <n>` 静音合并阈值
- `--vad-pad-sec <n>` 语音段前后扩展时间
- `--vad-min-silence-sec <n>` 最短静音阈值
- `--target-sec <n>` 目标分段长度（默认 20）
- `--max-sec <n>` 最大分段长度（默认 35）
- `--lid-cache-dir <path>` VoxLingua107 缓存目录
- `--lid-device <cpu|cuda>` LID 推理设备

## Phase 2 参数（逻辑建模）
- `--prompt <path>` SORAL 提示词路径（默认 `prompts/SORAL.txt`）
- `--graph-in <path>` 外部对象图文件（可选）
- `--graph-out <path>` 输出对象图文件（默认 `./out/<base>.graph.txt`）
- `--model-graph <name>` GPT-5.1（对象图）
- `--retry <n>` 默认 1（硬上限 1）

## Phase 3 参数（合并对象图）
- `--graph-a <path>` 对象图 A
- `--graph-b <path>` 对象图 B
- `--prompt <path>` 合并提示词路径（默认 `prompts/SORAL_Merge.txt`）
- `--out <path>` 输出合并对象图（默认 `./out/<baseA>__<baseB>.merged.graph.txt`）
- `--model <name>` GPT-5.1（合并模型）
- `--retry <n>` 默认 1（硬上限 1）

## 输出
- Phase 1：`out/<base>.srt`
- Phase 2：`out/<base>.graph.txt`
