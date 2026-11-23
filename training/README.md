# Fine-tuning LoRA

Il modulo `training` adatta Qwen2.5-0.5B-Instruct al corpus CanaryTrace. Congela i pesi base e addestra gli adattatori LoRA con la cross-entropy causale per la predizione del token successivo. Ogni epoca mescola e usa tutti i 256 record da 96 token ID già presenti nel corpus. [Corpus di training](../dataset/corpus.jsonl) · [Causal language modeling](https://huggingface.co/docs/transformers/en/tasks/language_modeling)

## Comandi

Esegui i comandi da PowerShell nella radice del progetto, dopo aver creato `.venv`. [Setup Python](../README.md#setup-python)

Installa le dipendenze e valida il corpus.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m dataset --validate-only
```

Per scaricare e verificare il modello base prima del training, usa il comando seguente. Il training usa la stessa cache e scarica il modello se manca.

```powershell
.\.venv\Scripts\python.exe -m models
```

Avvia il training con la configurazione predefinita.

```powershell
.\.venv\Scripts\python.exe -m training
```

Visualizza le opzioni disponibili oppure avvia una sola epoca con batch da due record.

```powershell
.\.venv\Scripts\python.exe -m training --help
.\.venv\Scripts\python.exe -m training --epochs 1 --batch-size 2
```

## Iperparametri

| Impostazione | Valore predefinito | Significato o opzione |
| --- | --- | --- |
| Epoche | `2` | Passaggi completi sul corpus, modificabili con `--epochs`. |
| Batch size | `4` | Record per aggiornamento, modificabili con `--batch-size`. |
| Ottimizzatore | AdamW | Aggiorna i parametri LoRA addestrabili. |
| Learning rate | `2e-4` | Passo dell'ottimizzatore. |
| Rango LoRA | `8` | Dimensione interna della scomposizione a rango ridotto. |
| Alpha LoRA | `16` | Il contributo dell'adattatore è scalato di `alpha / rango`. |
| Dropout LoRA | `0.05` | Probabilità di dropout nel ramo LoRA durante il training. |
| Proiezioni LoRA | `q_proj`, `v_proj` | Proiezioni query e value dell'attenzione. |
| Bias LoRA | `none` | I bias del modello restano congelati. |
| Seed PyTorch | `20261109` | Seed dell'inizializzazione LoRA e del mescolamento del corpus. |

Il modulo usa CUDA quando PyTorch la rileva e la CPU negli altri casi.

## Generazione in valutazione

La temperatura è un'impostazione della generazione. Il checkpoint Qwen ha `temperature=0.7` come valore predefinito. Per valutare il modello base e i checkpoint LoRA, il progetto imposta `temperature=1.0` e usa **greedy decoding** con `do_sample=False` e `num_beams=1`. La temperatura è inattiva in questa modalità, perché a ogni passo viene scelto direttamente il token con il punteggio più alto. [Protocollo di valutazione](../evaluation/README.md)

## Checkpoint

Ogni esecuzione crea una cartella distinta sotto `checkpoints/run-*/`. Alla fine di ciascuna epoca, il modulo stampa la loss media e salva `adapter_model.safetensors` e `adapter_config.json` in `epoch-01/`, `epoch-02/` e così via. I checkpoint contengono solo i pesi LoRA e la configurazione che identifica il modello base e la sua revisione. Per ricaricare un adattatore, passa il modello base della stessa revisione e la cartella dell'epoca a `PeftModel.from_pretrained`. [Formato dei checkpoint PEFT](https://huggingface.co/docs/peft/en/developer_guides/checkpoint)

Per elencare i pesi salvati, esegui il comando seguente.

```powershell
Get-ChildItem -LiteralPath checkpoints -Recurse -Filter adapter_model.safetensors
```
