# TN-Mammo: Experiment Plan for Four-View Fusion

## 1. Mục tiêu

So sánh công bằng các chiến lược kết hợp 4 view mammography ở mức ca, trong khi giữ cố định:

- shared DenseNet121 encoder;
- preprocessing;
- train/validation split;
- domain sampler TN/VinDr;
- class-balanced focal + CORAL loss;
- optimizer, scheduler và early stopping.

Metric chọn model chính: **validation Macro-F1**. Metric phụ: balanced accuracy, QWK, per-class F1, within-one và severe-error count.

## 2. Các fusion được triển khai

| Tên config | Mô tả | Mục đích |
|---|---|---|
| `mean` | Trung bình đều 4 feature | Baseline E1, không thêm tham số |
| `concat_mlp` | Ghép 4 feature rồi MLP nén về 1024 | Control: kiểm tra lợi ích do tăng capacity |
| `attention` | Học một trọng số cho mỗi view rồi weighted sum | Kiểm tra view weighting trực tiếp |
| `hierarchical_gated` | Fuse CC-MLO từng bên, sau đó fuse trái-phải | Model có cấu trúc giải phẫu, ứng viên chính |

Tất cả model dùng chung một DenseNet121 encoder cho bốn view.

## 3. Nguyên tắc thí nghiệm

1. Không dùng test khóa để chọn fusion, resolution hoặc hyperparameter.
2. Mỗi run chỉ thay đổi một nhóm yếu tố đã định trước.
3. Baseline và model cuối chạy ít nhất 3 seed.
4. Lưu prediction validation theo `case_id` để paired bootstrap sau này.
5. So sánh cả hiệu năng lẫn số tham số, thời gian và peak VRAM.

## 4. Phase A — Sanity check

Chạy mỗi fusion 1 epoch với một manifest nhỏ hoặc toàn bộ train nhưng giới hạn số sample.

- Resolution: 224
- Seed: 42
- Epoch: 1
- `num_workers`: 0 hoặc 2
- Mục tiêu: không lỗi shape, loss hữu hạn, checkpoint load/eval được.

Điều kiện pass:

- forward/backward thành công;
- `flat_logits` có shape `[B,4]`;
- `ordinal_logits` có shape `[B,3]`;
- không NaN/Inf;
- eval checkpoint chạy được.

## 5. Phase B — Fusion screening tại 224

Giữ nguyên toàn bộ cấu hình, chỉ thay `model.fusion`.

| Run | Fusion | Hidden dim | Dropout | Seeds |
|---|---|---:|---:|---|
| F0 | mean | — | — | 42, 52, 62 |
| F1 | concat_mlp | 256 | 0.2 | 42, 52, 62 |
| F2 | attention | 256 | 0.2 | 42, 52, 62 |
| F3 | hierarchical_gated | 256 | 0.2 | 42, 52, 62 |

Báo cáo `mean ± std` của Macro-F1, QWK và balanced accuracy.

Tiêu chí shortlist:

- Macro-F1 trung bình cao hơn baseline;
- không làm QWK giảm đáng kể;
- không tăng severe errors;
- improvement xuất hiện ở ít nhất 2/3 seed.

## 6. Phase C — Resolution study

Chỉ chạy hai fusion tốt nhất từ Phase B, gồm baseline `mean` và model tốt nhất.

| Resolution | Batch size gợi ý | Gradient accumulation |
|---:|---:|---:|
| 224 | 16 | 1 |
| 384 | 8 | 2 |
| 512 | 4 | 4 |

Giữ effective batch size gần tương đương. Chạy ít nhất 3 seed cho model cuối tại resolution được chọn.

Chọn resolution dựa trên:

1. Macro-F1 và QWK;
2. ổn định giữa seed;
3. peak VRAM và thời gian train;
4. mức cải thiện có đáng so với chi phí hay không.

## 7. Phase D — Fusion ablation

Nếu `hierarchical_gated` tốt nhất, chạy các ablation sau:

| Ablation | Ipsilateral CC-MLO | Bilateral left-right | Ý nghĩa |
|---|---:|---:|---|
| Mean | Không | Không | Baseline |
| Flat attention | Không | Không | Learned weighting không cấu trúc |
| Ipsilateral only | Có | Mean hai bên | Vai trò CC-MLO |
| Bilateral only | Mean mỗi bên | Có | Vai trò trái-phải |
| Full hierarchical | Có | Có | Model đầy đủ |

Nếu không muốn thêm class mới, hai ablation trung gian có thể được triển khai sau khi screening xác nhận hierarchical fusion có tiềm năng.

## 8. Phase E — Data strategy

Dùng fusion và resolution đã chọn:

| Run | Dữ liệu train | Sampling |
|---|---|---|
| D0 | TN only | shuffle/class strategy hiện tại |
| D1 | TN + VinDr | sampling theo tỷ lệ tự nhiên |
| D2 | TN + VinDr | target-aware sampler, TN mass = 0.6 |

Có thể thử thêm TN mass `0.5` và `0.7` nếu D2 tốt hơn D1.

## 9. Hyperparameter nhỏ cho fusion

Chỉ tune sau screening, trên model tốt nhất:

- `fusion_hidden_dim`: 128, 256, 512;
- `fusion_dropout`: 0.0, 0.2, 0.4.

Không grid search toàn bộ. Dùng one-factor-at-a-time hoặc tối đa 4–6 run.

## 10. Bảng kết quả cần xuất

### Bảng chính

| Fusion | Params | Macro-F1 | BalAcc | QWK | Severe |
|---|---:|---:|---:|---:|---:|

### Per-class

| Fusion | F1-A | F1-B | F1-C | F1-D |
|---|---:|---:|---:|---:|

### Efficiency

| Fusion | Resolution | Batch | Epoch time | Peak VRAM |
|---|---:|---:|---:|---:|

### Interpretability

- `attention`: phân bố `view_weights` cho L-CC/L-MLO/R-CC/R-MLO;
- `hierarchical_gated`: gate CC-vs-MLO của mỗi bên và gate left-vs-right;
- so sánh gate giữa các lớp A/B/C/D và giữa prediction đúng/sai.

## 11. Statistical analysis

Trên validation cho development:

- báo cáo mean ± std qua 3 seed;
- paired bootstrap trên prediction cùng case cho baseline và model đề xuất;
- bootstrap 95% CI cho Macro-F1 và QWK.

Test khóa chỉ chạy một lần sau khi:

- fusion đã khóa;
- resolution đã khóa;
- seed/model selection protocol đã khóa;
- checkpoint cuối đã được chọn chỉ từ validation.

## 12. Cấu hình mẫu

```yaml
model:
  use_ordinal_head: true
  imagenet_init: false
  initialization_checkpoint: null
  fusion: hierarchical_gated  # mean | concat_mlp | attention | hierarchical_gated
  fusion_hidden_dim: 256
  fusion_dropout: 0.2
```

## 13. Thứ tự chạy đề xuất

1. Smoke test bốn fusion.
2. Screening bốn fusion tại 224, 3 seed.
3. Chọn top-1 learned fusion.
4. So sánh mean và top-1 tại 224/384/512.
5. Ablation cấu trúc cho top-1.
6. Data-strategy experiments.
7. Khóa protocol và chạy test cuối.
