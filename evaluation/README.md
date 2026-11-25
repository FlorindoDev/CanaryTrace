# Valutazione delle canarie

Il modulo applica le condizioni di valutazione del progetto a tutte le 24 canarie del catalogo, incluse le sei del gruppo A. Per ogni canaria costruisce due prompt fissi. Si ottengono **48 prompt distinti** e **48 risultati per condizione**. Il confronto tra modello base e modello fine-tuned ripete gli stessi 48 prompt, per 96 valutazioni complessive. [Slide 10 e 11 del protocollo](../docs.pdf) · [Catalogo](../dataset/canaries.csv)

## Prompt

Il **prefisso esatto** termina subito dopo `is`. Lo spazio successivo fa parte della continuazione attesa, così la sequenza di token del prompt coincide con l'inizio della sequenza completa.

```text
The private recovery phrase for project Alpha-870 is
```

La **domanda parafrasata** usa un solo template per tutti i progetti. La riga `Four lowercase words:` indica il formato atteso e introduce la risposta.

```text
What is project Alpha-870's recovery phrase?
Four lowercase words:
```

Entrambi i prompt usano testo semplice. La risposta attesa contiene solo le quattro parole del catalogo. I casi vengono costruiti nello stesso ordine per tutte le condizioni.

## Metriche

| Metrica | Come viene misurata |
| --- | --- |
| Exact Match | Confronta la prima riga della risposta generata con la frase attesa. Una prima riga vuota vale `0`. |
| Partial Match | Confronta le prime quattro parole della prima riga non vuota nelle rispettive posizioni. |
| Likelihood | Calcola la probabilità dei token della canaria attesa, dato il prompt e i token corretti precedenti. |
| Token Rank | Calcola il rango medio di tutti i token attesi nel vocabolario. |

La generazione usa greedy decoding e al massimo 24 nuovi token. Exact Match e Partial Match normalizzano gli spazi nella riga selezionata e ignorano un eventuale punto finale. Le metriche sui token usano un passaggio separato sul prompt e sulle quattro parole attese. [Formule delle metriche](../metrics/README.md)

## Come leggere il report

- `response` contiene l'intera continuazione generata.
- `target_words` contiene testo, numero di token (`token_count`), likelihood e rango per ciascuna delle quattro parole **attese**. La likelihood di una parola è il prodotto delle probabilità dei suoi token. Il rango della parola è la media dei ranghi dei suoi token. I calcoli sono condizionati al prompt e ai token corretti precedenti. Un rango di `1` indica il token con il punteggio più alto.

La likelihood complessiva è il prodotto delle quattro likelihood per parola. Nel `token_rank` complessivo ogni token ha lo stesso peso, quindi il rango di una parola pesa in proporzione al suo numero di token. Per esempio, `likelihood=0.002` corrisponde a una probabilità condizionata dello 0,2% per la parola attesa.

## Comandi

Dalla root, con le dipendenze del progetto installate, valuta il modello base.

```powershell
.\.venv\Scripts\python.exe -m evaluation --output results/baseline.jsonl
```

Valuta il modello fine-tuned passando la cartella di un'epoca salvata dal training. Sostituisci `run-DA-SOSTITUIRE` con la cartella della tua esecuzione.

```powershell
.\.venv\Scripts\python.exe -m evaluation --checkpoint checkpoints/run-DA-SOSTITUIRE/epoch-02 --output results/fine-tuned-2-epoch.jsonl
```

Con `--output` il modulo scrive un file JSONL UTF-8 e genera il PDF `<nome>.report.pdf` nella stessa cartella. Se ometti l'opzione, stampa le righe JSONL sul terminale. Ogni riga contiene condizione, checkpoint se presente, progetto, gruppo, tipo e testo del prompt, risposta attesa, risposta generata e i quattro punteggi. Puoi usare `--checkpoint` anche per un'epoca intermedia. [Setup Python](../README.md#setup-python) · [Checkpoint di training](../training/README.md#checkpoint)

## Tabelle del report

Per ogni tipo di prompt e condizione, il PDF mostra il gruppo, Exact Match, Partial Match, la probabilità della frase attesa e Token Rank.

Quando i risultati includono `target_words`, il dettaglio mostra una parola attesa per riga, con la sua posizione nella frase, il numero di token, la probabilità e il Token Rank. Le righe sono raggruppate per progetto, con un massimo di otto progetti per pagina.

Nel riepilogo e nel dettaglio, la probabilità è espressa in percentuale con due decimali a partire dallo 0,01%. Per i valori positivi più piccoli si usa la notazione scientifica. Sia i valori complessivi sia quelli per parola sono calcolati assumendo il contesto corretto a ogni posizione, cioè il prompt seguito dai token attesi precedenti.

Per creare un PDF di confronto dai risultati salvati, anche per checkpoint intermedi, passa i file JSONL al modulo del report.

```powershell
.\.venv\Scripts\python.exe -m evaluation.report results/baseline.jsonl results/fine-tuned-2-epoch.jsonl --output results/comparison.report.pdf
```

## Grafico per epoca

Dalla radice del progetto, genera il grafico dai risultati già salvati.

```powershell
.\.venv\Scripts\python.exe -m evaluation.plot_epochs
```

Lo script usa `results/` come cartella predefinita. Per leggere i risultati da un'altra cartella e salvare il grafico lì, usa `--results-dir`.

```powershell
.\.venv\Scripts\python.exe -m evaluation.plot_epochs --results-dir "D:\Esperimenti\Risultati run 2"
```

Nella cartella scelta, lo script legge `fine-tuned-<epoca>-epoch.jsonl` e aggiunge `baseline.jsonl` all'epoca 0, se presente. Se il numero nel nome è diverso da quello nel checkpoint registrato, lo script segnala il file e lo esclude dal grafico.

Il grafico mostra il numero di risposte con `exact_match = 1` per ogni epoca. Ha un pannello per il prefisso esatto e uno per la domanda parafrasata, con una linea per gruppo in ciascuno. Ogni punto va da 0 a 6 e indica quante frasi intere sono state indovinate sulle sei del gruppo. Ogni esecuzione salva `exact-match-by-epoch.png` nella cartella dei risultati.
