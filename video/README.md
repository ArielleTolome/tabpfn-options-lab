# Animated project walkthrough

[Download the finished MP4](https://github.com/ArielleTolome/tabpfn-options-lab/releases/download/v0.1.0/tabpfn-options-lab.mp4) · [Scene overview](contact-sheet.jpg) · [Captions](captions.srt) · [YouTube upload copy](../YOUTUBE.md)

133 seconds · 1920 × 1080 · 30 fps · H.264 video / AAC audio. Six coded animation scenes, with AI-generated narration using Ariel's authorized ElevenLabs voice. The MP4 is a release attachment rather than a Git binary. No YouTube upload is performed by this repository.

## Preview and render

Requires Node.js/npm, network access for the pinned HyperFrames CLI and external GSAP/font assets, and a compatible Chromium/headless browser. HyperFrames may download its browser when first used.

```sh
cd video
npm run dev
# Separate terminal:
npm run check -- --at 11,34,56,79,105,130 --samples 6
npm run render -- --quality delivery --fps 30 --workers 4 --output renders/tabpfn-options-lab.mp4
```

`package.json` pins HyperFrames 0.8.162. `index.html` assembles six compositions; the scene timings follow the included narration. `narration.json` contains the script and speech-generation disclosure, not API credentials or voice-model files. The already-generated `assets/narration.mp3` is included so rendering needs no ElevenLabs account. `captions.srt` has word-aligned subtitle cues; upload it as the English subtitle track. The thumbnail is a still from the approved animation.

Verification for the published render: all 3,990 video frames encoded; final MP4 decoded into a six-frame review grid; H.264/AAC streams and 133-second duration checked with ffprobe. Audio decoding completed without errors and peak sample level measured −1.6 dBFS. Narration is not a claim of measured historical model performance.

## Rights

Animation code is covered by the repository's Apache-2.0 license. The generated voice recording, thumbnail, rendered video, and research manuscript are **not** relicensed under Apache-2.0. Copyright © 2026 Ariel Tolome; all rights reserved unless separately granted. Inclusion permits viewing the project, not voice cloning or impersonation. Third-party GSAP, fonts, HyperFrames, and model terms remain separate.
