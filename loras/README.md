# LoRAs

CivitAI model pages for each LoRA in this folder.

| File                                                  | CivitAI                                                                                                                                                                        | Recommended Strength |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| `gp-zimage_000008000.safetensors`                   | ~~[Z-Image GodPussy](https://civitai.red/models/2222911/z-image-godpussy)~~ (very poor outputs for Z-Image Turbo — not recommended)                                            | N/A                  |
| `Mystic-XXX-ZIT-V7.safetensors`                     | [ZIT Mystic XXX](https://civitai.red/models/2206377/zit-mystic-xxx?modelVersionId=2855359)                                                                                      | 0.8                  |
| `NSFW_master_ZIT_000017532.safetensors`             | [NSFW MASTER - Z-Image Turbo V2](https://civitai.red/models/667086/nsfw-master?modelVersionId=2904324) — **best performing, least body deformation**          | 0.8                  |
| `PornMaster_NSFW_ZIT_V1.safetensors`                | [PornMaster NSFW Z-image](https://civitai.red/models/2608450/pornmaster-nsfw-z-image?modelVersionId=2928904)                                                                    | 0.9                  |
| `RealisticSnapshot-Zimage-Turbov5.safetensors`      | [Realistic Snapshot Z-Image Turbo/Krea 2](https://civitai.red/models/2268008/realistic-snapshot-z-image-turbo-krea-2?modelVersionId=2617751)                                    | 0.6                  |
| `TeeKaysTittyTime_AQueensTits_V1-2_ZIT.safetensors` | [A Queen&#39;s Breasts \| Z-Image Turbo / Flux / Pony / Ill](https://civitai.red/models/927534/a-queens-breasts-or-z-image-turbo-flux-pony-ill)                                 | 0.6                  |
| `TurboPussyZ_v2.safetensors`                        | [Turbo Pussy Z](https://civitai.red/models/2178383/turbo-pussy-z?modelVersionId=2639284)                                                                                        | 1.5                  |
| `Z-Image-fac1al.safetensors`                        | [Facial Cumshot Krea 2 / Z-Image Turbo](https://civitai.red/models/2173844/facial-cumshot-krea-2-z-image-turbo?modelVersionId=2447989)                                          | 0.6                  |
| `zit_macromastia_naked.safetensors`                 | [Macromastia Naked (Krea 2 / Ideogram 4 / Flux / Z-Image etc.)](https://civitai.red/models/2311601/macromastia-naked-krea-2-ideogram-4-flux-z-image-etc?modelVersionId=2600739) | 0.6                  |
| `ZITnsfwLoRA.safetensors`                           | [Z-Image Turbo NSFW LoRA (v1)](https://civitai.red/models/2279079/z-image-turbo-nsfw-lora?modelVersionId=2565112)                                                               | 0.55                 |
| `ZITnsfwLoRAv2.safetensors`                         | [Z-Image Turbo NSFW LoRA (v2)](https://civitai.red/models/2279079/z-image-turbo-nsfw-lora?modelVersionId=2637792)                                                               | 0.6                  |
| `ZITnsfwLoRAv3.safetensors`                         | [Z-Image Turbo NSFW LoRA (v3)](https://civitai.red/models/2279079/z-image-turbo-nsfw-lora?modelVersionId=2898951)                                                               | 0.6                  |
| `ZPenisv2.safetensors`                              | [ZPenis - Realistic Penis Helper Z-Image LoRA](https://civitai.red/models/2182401/zpenis-realistic-penis-helper-z-image-lora)                                                   | 0.6                  |

## Best Combination

| LoRA                                              | Strength |
| ------------------------------------------------- | -------- |
| `NSFW_master_ZIT_000017532.safetensors`           | 0.8      |
| `ZITnsfwLoRAv3.safetensors`                       | 0.2      |

Total strength: 1.0. NSFW master provides the primary NSFW effect with minimal body deformation, while NSFW LoRA v3 at a low strength adds extra detail without introducing artifacts.
