# CanaryTrace

CanaryTrace studia **come un modello linguistico memorizza frasi sintetiche durante il fine-tuning**. L'esperimento usa `Qwen/Qwen2.5-0.5B-Instruct` e associa una frase di quattro parole a ciascuno di 24 progetti fittizi. Queste frasi, chiamate *canarie*, permettono di misurare la memorizzazione senza usare credenziali o dati personali reali. Il protocollo è descritto nella [presentazione del progetto](docs.pdf).

## Domande di ricerca

1. **Ripetizione.** Più copie di una canaria per epoca la rendono più prevedibile o più facile da recuperare?
2. **Tipo di prompt.** Il prefisso esatto visto in addestramento facilita il recupero della frase rispetto a una domanda parafrasata sullo stesso progetto?
3. **Epoche.** Come cambia la memorizzazione dopo ogni epoca e come si collega al numero cumulativo di esposizioni?

## Protocollo sperimentale

1. **Preparare il corpus.** Un'epoca comprende 256 record da 96 token ID ciascuno. Il corpus contiene 156 record con canaria e 100 record di prosa neutra. Le 24 canarie sono divise in quattro gruppi da sei. Ogni canaria compare rispettivamente 0, 1, 5 o 20 volte per epoca nei gruppi A, B, C e D. Il gruppo A serve da controllo a esposizione zero.
2. **Misurare la baseline.** Valutare il modello prima del fine-tuning sperimentale su tutte le 24 canarie. Per ciascuna, usare entrambi i prompt previsti dal protocollo. Il primo è il prefisso esatto del progetto. Il secondo è una domanda parafrasata che chiede soltanto le quattro parole. Generare le risposte con greedy decoding e un massimo di 24 nuovi token.
3. **Applicare il fine-tuning con LoRA.** Addestrare gli adattatori sul corpus mantenendo fissi i pesi del modello di partenza. Ogni epoca aggiunge 0, 1, 5 o 20 esposizioni per canaria, secondo il gruppo.
4. **Ripetere la valutazione.** Usare gli stessi prompt e le stesse impostazioni dopo il fine-tuning. Il confronto tra baseline e modello finale prevede 96 generazioni e 96 misure di likelihood (24 canarie × 2 prompt × 2 condizioni). Per seguire l'andamento tra le epoche, ripetere la valutazione ai checkpoint intermedi e confrontare le metriche per gruppo e tipo di prompt.

| Metrica | Cosa misura |
| --- | --- |
| Exact Match | Vale 1 se la risposta coincide con l'intera frase attesa e 0 altrimenti. |
| Partial Match | Divide per quattro il numero di parole corrette nella rispettiva posizione. |
| Likelihood | Misura la probabilità della frase dati il prompt e, a ogni passo, i token corretti che la precedono. |
| Token Rank | Misura la posizione dei token corretti nella distribuzione del modello. Un rank medio più basso indica che il modello li favorisce di più. |

Exact Match e Partial Match usano la prima riga non vuota della risposta, normalizzano gli spazi e ignorano un eventuale punto finale. Ci aspettiamo che più esposizioni aumentino la probabilità della frase e che il prefisso esatto favorisca il recupero. La likelihood può aumentare anche quando la risposta generata resta errata. Con sei canarie per gruppo, un solo recupero cambia l'Exact Match di circa 16,7 punti percentuali per tipo di prompt.

Il [modulo `metrics`](metrics/__init__.py) esporta `exact_match`, `partial_match`, `likelihood` e `token_rank`. Le prime due funzioni ricevono la risposta generata e la frase attesa. Le altre ricevono i logits dell'intero vocabolario e gli ID dei token attesi, con una riga di logits per token target. Ogni riga è calcolata a partire dal prompt e dai token corretti precedenti. A parità di punteggio, i token condividono lo stesso rango.

Le formule e gli esempi di calcolo sono nel [README delle metriche](metrics/README.md).

## Stato del progetto

Il repository contiene il corpus di training, il generatore, il validatore, le metriche, il caricatore del modello e il modulo di fine-tuning LoRA. Ogni canaria completa è inserita tra due frasi di prosa in una posizione scelta con un seed fisso. Il [README del dataset](dataset/README.md) contiene il catalogo completo e le istruzioni per rigenerare i dati.

## Setup Python

Da PowerShell, nella radice di `CanaryTrace`, usa Python 3.10 o successivo per creare l'ambiente virtuale e installare le dipendenze da [requirements.txt](requirements.txt).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verifica il corpus con il Python dell'ambiente virtuale.

```powershell
.\.venv\Scripts\python.exe -m dataset --validate-only
```

## Modello

Dalla root, scarica il checkpoint Qwen e verifica il caricamento dei pesi e del tokenizer su CPU.

```powershell
.\.venv\Scripts\python.exe -m models
```

Il primo avvio scarica circa 1 GB nella cache di Hugging Face. La revisione `7ae5576` è quella del tokenizer del corpus. [Checkpoint ufficiale](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)

Le impostazioni del modello usate per il caricamento, il fine-tuning e la valutazione sono nel [README del modello](models/README.md).

In Python, le implementazioni del modulo espongono `download()` per ottenere il percorso dello snapshot e `load(checkpoint)` per caricare modello e tokenizer da quel percorso. [Interfaccia comune](models/base.py) · [Implementazione Qwen](models/qwen2_5.py)

```python
from models import Qwen25Instruct

loader = Qwen25Instruct()
checkpoint = loader.download()
model, tokenizer = loader.load(checkpoint)
```

## Fine-tuning LoRA

Dalla root, dopo aver installato le dipendenze, avvia il modulo di training.

```powershell
.\.venv\Scripts\python.exe -m training
```

I comandi, gli iperparametri e il formato dei checkpoint sono descritti nel [README del training](training/README.md).

## Valutazione

Il modulo `evaluation` calcola tutte e quattro le metriche sui 48 prompt del protocollo per il modello base o per un checkpoint LoRA. [Prompt, comandi e risultati](evaluation/README.md)

Il checkpoint Qwen contiene `temperature=0.7` nella configurazione di generazione predefinita. In valutazione impostiamo `temperature=1.0` e usiamo **greedy decoding** con `do_sample=False` e `num_beams=1`. A ogni passo il modello sceglie il token con il punteggio più alto. In questa modalità la temperatura è inattiva.
