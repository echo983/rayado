# 缓存落盘 Schema（单机）

## 数据库
- sqlite 文件：`cache.sqlite`
- 目标：幂等复用 ASR 结果，命中即跳过请求

## 表结构
```
CREATE TABLE IF NOT EXISTS asr_cache (
  key TEXT PRIMARY KEY,
  request_hash TEXT NOT NULL,
  response TEXT NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_asr_cache_request_hash
  ON asr_cache (request_hash);
```

## 键规则
- key = hash(input_hash + chunk_id + asr_model + asr_params)
- request_hash = hash(request_body)

## 写入规则
- 仅在成功响应时写入
- 失败或超时不写入

## 读取规则
- key 命中即复用 response
- request_hash 不匹配视为参数漂移，忽略缓存
