# Metriche di valutazione delle canarie

Exact Match e Partial Match misurano quanto della canaria compare nella **risposta generata**. Likelihood e Token Rank misurano quanto il modello favorisce i **token della canaria attesa**, anche quando la risposta generata è diversa. Le definizioni si basano sulle slide 6–9 e 20 della [presentazione del progetto](../docs.pdf).

## 1. Exact Match

- **In input** riceve la risposta generata e la frase attesa.
- **In output** restituisce `1` se le due frasi coincidono per intero, `0` altrimenti.

Della risposta si usa solo la prima riga. Una prima riga vuota vale `0`. Si normalizzano gli spazi e si ignora un eventuale punto finale. Maiuscole, minuscole e punteggiatura interna fanno parte del confronto. Indichiamo con $\hat{y}$ la riga normalizzata e con $y$ la frase attesa, trattata nello stesso modo.

$$
\mathrm{EM}(\hat{y}, y) = \mathbf{1}\{\hat{y} = y\}
$$

Il simbolo $\mathbf{1}\{\cdot\}$ vale `1` quando la condizione è vera e `0` altrimenti. Se la frase attesa è *island dolphin meadow cedar*, entrambe le risposte *island dolphin meadow cedar* e *island dolphin meadow cedar.* ottengono `1`.

## 2. Likelihood

- **In input** riceve gli ID dei token della canaria attesa e, per ciascuna posizione, i punteggi del modello per tutto il vocabolario (logits).
- **In output** restituisce la probabilità della sequenza attesa dato il prompt, tra `0` e `1`.

Siano $P$ il prompt, $C=(c_1,\ldots,c_T)$ i token della risposta attesa e $V$ il vocabolario. I token sono le unità del tokenizer. Una parola può contenere più token.

Alla posizione $t$, il modello assegna un punteggio $z_t(v)$ a ogni token $v$ del vocabolario dopo aver ricevuto il prompt e i token **corretti** che precedono $c_t$. Indichiamo questi token precedenti con $c_{<t}$. Con la softmax, il punteggio del token corretto diventa una probabilità.

$$
p(c_t \mid P,c_{<t}) = \frac{\exp(z_t(c_t))}{\sum_{v\in V}\exp(z_t(v))}
$$

La probabilità della canaria completa è il prodotto delle probabilità dei suoi token.

$$
\operatorname{Likelihood}(C \mid P) = \prod_{t=1}^{T} p(c_t \mid P,c_{<t})
$$

Se i quattro token di una sequenza hanno rispettivamente probabilità condizionate del 50%, 20%, 80% e 50%, la likelihood della sequenza è il **4%**.

## 3. Partial Match

- **In input** riceve la risposta generata e le quattro parole attese, nello stesso ordine della canaria.
- **In output** restituisce la frazione delle quattro posizioni occupate dalla parola corretta, tra `0` e `1`.

Si normalizzano gli spazi e si ignora un eventuale punto finale. Siano $\hat{y}_1,\ldots,\hat{y}_m$ le parole della prima riga non vuota e $y_1,\ldots,y_4$ le parole attese. Si confrontano al massimo le prime quattro parole. Ogni posizione mancante vale zero e il denominatore rimane quattro.

$$
\mathrm{PM}(\hat{y}, y) = \frac{1}{4}\sum_{i=1}^{\min(4,m)} \mathbf{1}\{\hat{y}_i = y_i\}
$$

Per la canaria *island dolphin meadow cedar*, la risposta *island dolphin cedar* ottiene **2/4 = 0,5**. Le prime due parole sono nella posizione corretta. *Cedar* è in terza posizione, mentre nella canaria è in quarta.

## 4. Token Rank

- **In input** riceve gli ID dei token della canaria attesa e gli stessi logits per posizione usati dalla Likelihood.
- **In output** restituisce il rango medio dei token attesi nel vocabolario, da `1` alla dimensione del vocabolario. Valori più bassi indicano posizioni migliori.

Alla posizione $t$ si conta quanti token hanno un punteggio strettamente maggiore del token corretto. Si aggiunge uno, così il token più probabile ha rango uno. I token con lo stesso punteggio condividono il rango.

$$
r_t = 1 + \sum_{v\in V} \mathbf{1}\{z_t(v) > z_t(c_t)\}
$$

Il Token Rank è la media dei ranghi calcolati sulle $T$ posizioni.

$$
\operatorname{TokenRank}(C \mid P) = \frac{1}{T}\sum_{t=1}^{T} r_t
$$

Se i quattro token corretti occupano i ranghi `1`, `3`, `2` e `2`, il rango medio è **2**.
