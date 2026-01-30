# CLI 约定（MVP）

## 命令
```
rayado transcribe <input>
```

## 参数
- `--out <dir>` 输出目录（默认 `./out/<basename>`）
- `--concurrency <n>` 并发（默认 64）
- `--chunk-sec <n>` 默认 25
- `--overlap-sec <n>` 默认 1.5
- `--retry <n>` 默认 1（硬上限 1）
- `--asr-provider <name>` 服务商标识
- `--cache-dir <path>` 单机提示词缓存目录
- `--deepgram-model <name>` Deepgram 模型（默认 `nova-2`）
- `--deepgram-language <code>` Deepgram 语言提示（例如 `es`）
- `--deepgram-diarize/--no-deepgram-diarize` 是否启用说话人分离
- `--deepgram-smart-format/--no-deepgram-smart-format` 是否启用 Smart Format
- `--deepgram-punctuate/--no-deepgram-punctuate` 是否启用标点
- `--vad <name>` VAD 实现标识（默认启用）
- `--vad-threshold <n>` VAD 阈值（默认保守，偏向跳过）
- `--vad-min-speech-sec <n>` 最短语音段阈值
- `--vad-merge-gap-sec <n>` 静音合并阈值
- `--vad-pad-sec <n>` 语音段前后扩展时间

## 输出
- `episode.gcl`
- `transcript.txt`
- `subtitles.srt`
