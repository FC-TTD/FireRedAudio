# FireRedAudio preview on ttd-edge

- Runtime: official FireRedAudio BF16 weights on GPU0
- Decoder: official RedAE decoder on CPU, without quantization
- Port: `17880`
- UI: ASR, long-form transcription, temporal grounding, audio QA, voice cloning,
  voice design, semantic editing, and acoustic editing
- Upstream commit: `88b826378023eb9a49b297214568398a300e5c32`

The preview serializes inference because the BF16 model nearly fills one RTX 3090.
Voice cloning is intended for authorized reference recordings and research use.
