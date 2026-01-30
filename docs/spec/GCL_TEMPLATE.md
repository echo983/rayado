# GCL 模板（最小可用）

## 说明
- append-only；不覆盖旧记录。
- 示例字段为最小集，可按需要扩展。

## 头部
```
GCL_HDR
ver 0.2
mode append_only
time_unit seconds_float
```

## 切片（chunk）
```
GCL_CHUNK
chunk_id C0001
t0 0.0
t1 25.0
overlap_left 1.5
overlap_right 1.5
skip_reason non_speech
```

## 片段（span）
```
GCL_SPAN
sid S0001
t0 1.2
t1 5.6
chunk_id C0001
text_raw hello there
asr_conf 0.86
```

## 重叠裁决（可选）
```
GCL_OVERLAP
olp_id OLP0001
left_chunk C0001
right_chunk C0002
left_sid S0001
right_sid S0002
decision keep_left
method time_iou_then_text_sim
conf 0.78
```

## 覆盖/抑制（可选）
```
GCL_OVERRIDE
oid O0001
sid S0002
policy suppress
conf 0.80
```

## 语言检测记录（Deepgram，示例）
```
GCL_OVERRIDE
oid LANG_C00001
sid S00001
policy detected_language
deps es
conf 0.92
```

# GCL 增强字段（面向 LLM 消费）

## 实体
```
GCL_ENTITY
eid E0001
etype Person
canon_name Kim Cheolsu
conf 0.92
```

## 实体提及
```
GCL_MENTION
mid M0001
sid S0001
eid E0001
surface Kim Cheolsu
conf 0.88
```

## 实体映射
```
GCL_ENTITY_MAP
from_eid E_TMP_01
to_eid E0001
conf 0.90
```

## 说话人
```
GCL_SPEAKER
spk_id SPK01
label Speaker_1
conf 0.84
```

## 说话人映射
```
GCL_SPEAKER_MAP
spk_id SPK01
eid E0001
conf 0.80
```

## 场景
```
GCL_SCENE
scene_id SC01
t0 120.0
t1 300.0
title Scene_1
conf 0.75
```

## 场景关联
```
GCL_SCENE_SPAN
link_id L01
scene_id SC01
sid S0001
role dialogue
conf 0.80
```
