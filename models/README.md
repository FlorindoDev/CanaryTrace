# Modello Qwen

CanaryTrace usa `Qwen/Qwen2.5-0.5B-Instruct` alla revisione `7ae557604adf67be50417f59c2c2f167def9a775`. Il fine-tuning e la valutazione partono dallo stesso checkpoint, con il tokenizer usato per costruire il corpus. [Checkpoint ufficiale](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)

## Download e caricamento

Dalla root, con le dipendenze installate nell'ambiente virtuale, esegui il comando seguente per scaricare lo snapshot e verificare il caricamento su CPU. [Setup Python](../README.md#setup-python)

```powershell
.\.venv\Scripts\python.exe -m models
```

Il caricatore riusa lo snapshot locale e carica i pesi Safetensors in `bfloat16`, il tipo del checkpoint. [Caricatore Qwen](qwen2_5.py)

## Fine-tuning LoRA

| Impostazione | Valore predefinito |
| --- | --- |
| Proiezioni adattate | `q_proj`, `v_proj` |
| Rango LoRA | `8` |
| Alpha LoRA | `16` |
| Dropout LoRA | `0.05` |
| Bias LoRA | `none` |

Il fine-tuning aggiorna gli adattatori con la cross-entropy per il token successivo. Per epoche, batch size, ottimizzatore e checkpoint, consulta il [README del training](../training/README.md).

## Generazione in valutazione

Il modello base e i checkpoint LoRA rispondono agli stessi 48 prompt. La generazione usa **greedy decoding** con `do_sample=False`, `num_beams=1` e al massimo 24 nuovi token. `temperature=1.0` è impostata nella configurazione e resta inattiva con il greedy decoding. [Protocollo di valutazione](../evaluation/README.md)
