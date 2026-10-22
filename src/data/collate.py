from typing import Any, Dict, List

import torch


class MultimodalCollate:
    """Collate function dynamically padding sequences and building attention masks."""

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = {}

        # Collate labels and sample IDs
        collated["label"] = torch.stack([s["label"] for s in batch], dim=0)
        collated["ids"] = [s["id"] for s in batch]
        collated["speaker_id"] = torch.tensor([s.get("speaker_id", 0) for s in batch], dtype=torch.long)

        # Process each modality
        for mod in ["text", "audio", "video"]:
            feats = [s[mod] for s in batch]
            avails = torch.stack([s[f"{mod}_avail"] for s in batch], dim=0)
            collated[f"{mod}_avail"] = avails

            # Inspect shape of first valid item or fallback to 1D
            sample_feat = feats[0]
            is_sequence = sample_feat.ndim == 2  # [T, D]

            if is_sequence:
                max_len = max(f.shape[0] for f in feats)
                dim = sample_feat.shape[1]
                batch_size = len(batch)

                padded_feats = torch.zeros(batch_size, max_len, dim, dtype=torch.float32)
                seq_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)

                for i, (feat, avail) in enumerate(zip(feats, avails)):
                    if avail and feat.shape[0] > 0:
                        l = feat.shape[0]
                        padded_feats[i, :l, :] = feat
                        seq_masks[i, :l] = True
                    else:
                        # Missing or empty modality
                        seq_masks[i, 0] = False

                collated[mod] = padded_feats
                collated[f"{mod}_mask"] = seq_masks

            else:  # Vector [D] or scalar
                vector_feats = []
                dim = sample_feat.shape[-1] if sample_feat.ndim > 0 else 1
                for feat, avail in zip(feats, avails):
                    if avail:
                        vector_feats.append(feat.view(-1))
                    else:
                        vector_feats.append(torch.zeros(dim, dtype=torch.float32))

                collated[mod] = torch.stack(vector_feats, dim=0)
                collated[f"{mod}_mask"] = avails.unsqueeze(1)  # [B, 1]

        return collated
