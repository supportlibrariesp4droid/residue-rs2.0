# Roadmap di residue-rs

Stato di partenza: versione 0.1.0, circa 2 900 righe di libreria, 280 test che
passano (SIMD, scalare, debug), nessun `unsafe`. Questa roadmap porta il crate
verso sei obiettivi e fissa l'ordine in cui affrontarli, che conta quanto gli
obiettivi stessi: alcuni preparano il terreno per altri, e uno (i primi a 62
bit) è abbastanza incerto da meritare una misura prima di qualsiasi riscrittura.

| # | Obiettivo | Tipo di cambiamento | Fase |
|---|---|---|---|
| 3 | Conversione verso il binario vettorizzata | additivo | 1 |
| 4 | Multithreading tra i primi | additivo (feature `parallel`) | 2 |
| 1 | Aritmetica modulo un dispari grande (Montgomery RNS) | additivo | 3 |
| 5 | SIMD portabile: ARM NEON e WASM | interno + CI | 4 |
| 2 | Primi da circa 62 bit | sarebbe stato breaking | 0: misurato e respinto |
| 6 | Binding Python | crate separato | 6 |

Nota sulle versioni: il piano assegnava a ogni fase una versione 0.x. Alla fine
tutto il lavoro esce insieme nella versione 2.0.0, che resta compatibile con il
codice scritto per la 0.1.0.

## Perché quest'ordine

Prima le conversioni (3). Sono il tallone d'Achille misurato: 28.8 ns per
riportare in binario un valore a 9 primi, contro 2.5 ns di una moltiplicazione
in RNS. In più, il nucleo della ricostruzione, `Σ ξ_i·M_i − α'·M`, è lo stesso
calcolo della base extension, che è il cuore di Montgomery RNS. Ottimizzarlo
una volta serve a tre obiettivi (3, 1 e 6).

Il multithreading subito dopo (4). È additivo, piccolo e quasi meccanico:
`RnsVec` tiene già ogni canale in un `Vec<u32>` separato (`res: [Vec<u32>; N]`),
quindi i canali sono già indipendenti in memoria.

Montgomery RNS prima dei 62 bit (1 prima di 2). È la mancanza funzionale più
importante e non dipende dalla larghezza dei canali, *a patto* che la Fase 0
introduca uno strato interno che separi gli algoritmi dalla parola macchina.
Così, quando arriveranno i 62 bit, Montgomery RNS li erediterà senza riscritture.

I 62 bit: misura subito, implementazione dopo (2). Il README stesso dice che
è "la principale domanda aperta" e che "va misurata". Riscrivere i tipi interni
sulla base di un'ipotesi è il rischio più grosso della roadmap. Un prototipo
fuori dall'albero, in Fase 0, decide se e come farlo.

Python per ultimo (6). Il binding espone ciò che esiste. Farlo dopo
Montgomery RNS e dopo la decisione sui 62 bit evita di ridisegnare l'API Python
due volte. Il binding richiede però una scelta di design (basi a runtime) che va
preparata prima: è indicata in Fase 3.

## Fase 0: fondamenta, misure e pulizia del codice Rust (conclusa)

### Esito
- CI su GitHub Actions: lint, tre suite su x86-64, arm64, macOS e Windows,
  test senza feature, build bare-metal, wasm32 con e senza `simd128`, MSRV.
  Le parti non x86 non si potevano eseguire nell'ambiente di sviluppo e vanno
  verificate al primo push.
- Baseline `v0.1.0` salvata per `compare` e per il nuovo `benches/internal.rs`.
- Conversione dal binario più veloce (riduzione ritardata), ricostruzione per
  colonne, `Extender` senza divisioni a 128 bit. Cifre corrette in Fase 1:
  quelle pubblicate qui inizialmente (2,9-13,6× e fino a 1,45×) erano viziate
  da un benchmark che lasciava al compilatore lavoro da eliminare e da
  confronti fra momenti diversi della sessione. Misurate A/B: conversione dal
  binario 1,3-3,4×, ricostruzione 1,2-1,3× su 36-68 canali ma 0,85× su 9;
  scenario E 2,3× (confermato).
- `core::error::Error`, `#[must_use]`, sezioni `# Errors`/`# Panics`; la suite
  senza feature ora compila e passa (214 test).
- Primi a 62 bit respinti dall'esperimento (sotto). La Fase 5 è chiusa.
- Lo strato interno indipendente dalla parola (0.3) non è stato fatto:
  serviva solo ai 62 bit. L'esperimento indica invece una ristrutturazione
  dei kernel per canale, spostata in Fase 2.

Il piano originale della fase segue invariato, per riferimento.

Obiettivo: rendere misurabile ogni passo successivo e rendere il codice pronto a
cambiare larghezza di parola, senza toccare l'API pubblica.

### 0.1 Infrastruttura
- CI multi-architettura: x86-64 (scalare, AVX2, AVX-512 se disponibile),
  `aarch64-unknown-linux-gnu` sotto QEMU, `wasm32-wasip1` con e senza
  `+simd128` (eseguito con wasmtime), `thumbv7em-none-eabihf` solo in build per
  verificare il `no_std`. Le tre suite già esistenti girano su tutte.
- Baseline di benchmark salvata (`criterion --save-baseline v0.1.0`), così
  ogni fase confronta contro lo stesso punto di partenza. Aggiungere uno scenario
  di sola conversione in uscita (da `RnsVec` a `i128` e a limb) che oggi non è
  misurato da solo.
- Controllo dell'assembly sui kernel critici con `cargo-show-asm`, come
  regola e non come eccezione: la lezione delle closure non inlineate in `pulp`
  vale per ogni nuova architettura.
- Una macchina con più di 2 vCPU per le misure della Fase 2.

### 0.2 Ottimizzazioni del codice esistente
Interventi locali, ciascuno verificato con i benchmark:

- Ricostruzione per colonne invece che per righe. `reconstruct` oggi fa `N`
  passate di `mac_small`, ognuna con la propria catena di riporti su `l` limb.
  Accumulando per posizione di limb (`Σ_i ξ_i·M_i[j]` in un `u128`, poi un solo
  passaggio di riporto) si elimina una catena di riporti per canale, e il ciclo
  interno diventa la stessa forma che la Fase 1 vettorizza.
- Riduzione ritardata in `mont_from_limbs`. Ogni limb fa due `ch.add` con
  confronto. I risultati di `redc` sono `< 2^31`: si possono sommare in un `u64`
  e ridurre una volta alla fine.
- Layout di `RnsVec`. Valutare un solo buffer `Vec<u32>` con passo fisso al
  posto di `N` allocazioni separate: una allocazione invece di `N`, allineamento a
  64 byte controllabile, e `chunks_exact_mut` pronto per il parallelismo. Da
  tenere solo se i benchmark non peggiorano.
- Tipi di errore: `OutOfRange`, `ParseExactError` e `ParseRnsError`
  implementano `Display` ma non `core::error::Error` (stabile in `core` da Rust
  1.81, quindi compatibile col `no_std`). Aggiungerlo permette `?` con `anyhow`
  e `Box<dyn Error>`.
- Audit di `#[inline(always)]`: necessario sui kernel dentro `dispatch`,
  probabilmente superfluo altrove (gonfia il codice e i tempi di compilazione).
- `clippy::pedantic` come riferimento, `#[must_use]` sulle operazioni che
  restituiscono un valore nuovo, rimozione dei `needless_range_loop` dove un
  iteratore è altrettanto chiaro.

### 0.3 Strato interno indipendente dalla parola
Introdurre, solo internamente, un trait per l'aritmetica di canale (`ChannelOps`
o simile: tipo del residuo, `add`, `sub`, `mul_mont`, `redc`, accumulo ritardato)
e far passare kernel e algoritmi attraverso di esso. `Channel` resta l'unica
implementazione. Non cambia nulla per l'utente, ma è ciò che rende la Fase 5 un
cambio di implementazione invece che una riscrittura.

### 0.4 Esperimento sui 62 bit (decisione, non codice di libreria)
Prototipo fuori dall'albero del solo prodotto matriciale 64×64 e del
determinante, con tre candidati:

| Candidato | Prodotto | SIMD | Canali per 2^277 |
|---|---|---|---|
| 31 bit (attuale) | `u32×u32→u64` | sì, ovunque | 9 |
| ~62 bit | `u64×u64→u128` (`mulx`) | no, scalare | 5 |
| ~50 bit in `f64` | FMA esatta (prodotto alto + basso) | sì: AVX2, NEON, WASM relaxed-simd | 6 |

Più una quarta variante da provare *prima* delle altre: il kernel a 31 bit
attuale con blocking per cache e tiling nei registri. Su matrici 64×64 parte
del vantaggio di FLINT può venire dall'organizzazione del ciclo più che dalla
larghezza dei primi, e questa ottimizzazione non rompe nulla.

Criterio di decisione: si va alla Fase 5 solo se un candidato batte il
31 bit (già con blocking) di almeno 1.3× sugli scenari F e H, senza perdere più
del 10% sugli scenari A e B. Altrimenti i 62 bit restano un'opzione per le sole
matrici, o vengono scartati, e DESIGN.md registra il perché.

Fatto quando: CI verde su tutte le architetture, baseline salvata, nessuna
regressione, decisione sui 62 bit scritta in DESIGN.md.

## Fase 1: conversione verso il binario vettorizzata (conclusa)

### Esito (misure A/B contro 0.1.0, tre giri alternati)
- API in blocco: `to_twos_complement`, `to_i128s`, `to_i64s`,
  `from_twos_complement`, `to_bigints`/`from_bigints`; `tighten` in blocco.
- Dal binario in blocco: 3,0× (9 canali), 4,5× (36), 5,1× (68).
- Verso `i128` in blocco: 2,5× (9), 17× (36), perché si calcolano solo 2 limb.
  Anche lo scalare `to_i128` ne beneficia: 1,4× e 4-4,7×.
- Verso il binario a larghezza piena: solo 1,1-1,3×. Obiettivo ≥3× non
  raggiunto in questo caso: il kernel è limitato dal traffico sugli
  accumulatori, e la versione con accumulatori nei registri è stata
  vettorizzata male da LLVM. Serve SIMD esplicito (Fase 2).
- Prodotto matriciale: una regressione del 28% emersa dalle misure A/B è stata
  corretta con il dispatch per canale (anticipato dalla Fase 2).
- Lezioni di misura in DESIGN.md: consumare tutto il risultato, confronti
  alternati nella stessa esecuzione, cartelle `target` separate.

Il piano originale della fase segue invariato, per riferimento.

Oggi non esiste una conversione in blocco verso il binario: `RnsVec::to_vec`
produce `Rns` scalari, e ognuno si ricostruisce da solo in `O(n²)`.

### Algoritmo
Per un intero blocco di elementi, canale per canale (layout già SoA):
1. `ξ_i[k] = r_i[k]·y_i mod m_i`: una moltiplicazione per costante, vettorizzata
   lungo gli elementi `k` (stesso schema dei kernel attuali).
2. `α'[k]` dal canale ridondante: aritmetica `u64` a wrapping, vettorizzata.
3. `Σ_i ξ_i[k]·M_i[j]` per ogni posizione di limb `j`: il prodotto `32×64` si
   spezza in due `32×32`, accumulati in modo ritardato come nel prodotto
   scalare. È a tutti gli effetti un prodotto matrice-matrice
   (elementi × canali) · (canali × limb), quindi riusa il kernel con riduzione
   ritardata già scritto per `RnsMatrix`.
4. Sottrazione di `α'·M`, propagazione dei riporti e segno, una volta per
   elemento.

Il costo resta `O(n·L)` per elemento, ma con SIMD lungo gli elementi invece che
scalare.

### API nuova (additiva)
- `RnsVec::to_i128s(&self) -> Result<Vec<i128>, OutOfRange>` con percorso rapido
  quando il bound è ≤ 127 bit (solo 2 limb da calcolare).
- `RnsVec::to_i64s`, con percorso a costo quasi zero quando il bound è ≤ 63 bit
  (basta il canale ridondante).
- `RnsVec::write_limbs(&self, out: &mut [u64], stride)` per uscite più larghe.
- Conversione in ingresso in blocco da limb arbitrari, oltre a `from_i128s`.
- Con `num-bigint`: `to_bigints` / `from_bigints` in blocco.

### Obiettivo misurabile
Conversione in uscita a 9 primi da 28.8 ns ad almeno 3× più veloce per elemento;
scenari A e B "partendo da i128" ridotti di conseguenza, e grafico del crossover
rigenerato.

Fatto quando: test di proprietà contro `num-bigint` sulle nuove funzioni per
basi da 1 a 16 primi, incluse quelle a primi piccoli e misti; README aggiornato.

## Fase 2: multithreading tra i primi e kernel per canale (conclusa)

### Esito (A/B contro la fine della Fase 1, tre giri alternati, VM con 2 vCPU)
- Kernel espliciti x86 (`src/batch/x86.rs`, intrinseche sicure di `pulp`,
  nessun `unsafe`): prodotto matriciale con blocchi nei registri (2×8 AVX2,
  4×16 AVX-512), 1,5× su un thread; conversione verso il binario fino a 6
  limb con AVX-512, 1,25-1,3× a 9 canali.
- Feature `parallel` (rayon): prodotto per (canale, blocco di righe),
  determinante per canale, conversioni per canale o per intervalli di
  elementi, prodotto scalare e moltiplicazione per canale. Con 2 thread
  contro 1: determinante 1,6-1,9×, prodotto 1,3-1,6×, conversioni 1,5-1,8×.
  Rispetto alla Fase 1, il prodotto 64×64 con 2 thread è 2,0-2,2× più veloce.
- Soglia: 2^19 moltiplicazioni (150-300 µs). La prima, 2^17, rallentava
  del 25% le conversioni di 4096 valori con 2 thread.
- Non verificato: la scalabilità oltre 2 thread (la macchina ne ha 2) e i
  kernel AVX2 su una CPU senza AVX-512 (testati, ma misurati solo su una CPU
  che ha anche AVX-512).
- Aperto: il kernel di conversione esplicito copre solo fino a 12 cifre;
  36-68 canali a larghezza piena restano sul portabile. Scenari B e C dal
  binario 5-10% più lenti della Fase 1, nel rumore, non spiegato.

Il piano originale della fase segue invariato, per riferimento.

### Kernel per canale (dall'esperimento della Fase 0)
Lo stesso ciclo del prodotto matriciale gira in 0,46 ms come funzione per
canale e in 0,74 ms dentro la libreria: un margine di 1,3-1,6 volte senza
toccare l'aritmetica. Prima del multithreading:
- far girare ogni canale nella propria funzione dentro il contesto SIMD,
  che è anche l'unità di lavoro naturale per i thread. Fatto in Fase 1 per
  il prodotto matriciale: ha annullato una regressione del 28%, ma non ha
  ancora dato l'1,6× visto nell'esperimento;
- conversione verso il binario a larghezza piena con SIMD esplicito: la
  versione autovettorizzata della Fase 1 guadagna solo 1,1-1,3×;
- isolare il resto del divario (allocazione dell'output, dispatch, frequenza
  delle istruzioni a 512 bit) con misure mirate;
- scrivere il tiling nei registri con tipi vettoriali espliciti di `pulp`
  (API sicura), perché l'autovettorizzazione di LLVM lo sbaglia con AVX-512.

### Design
- Nuova feature `parallel` (implica `std`), basata su `rayon`. Il crate resta
  `#![forbid(unsafe_code)]`.
- Determinante: `N` eliminazioni di Gauss indipendenti, una per canale, ognuna
  con il proprio buffer di lavoro.
- Prodotto matriciale: parallelismo per canale *e* per blocchi di righe,
  perché con basi piccole (`N` = 4) i canali sono meno dei core.
- Operazioni elemento per elemento e conversioni su vettori grandi: blocchi
  di elementi per thread. Oltre il milione di elementi il limite è la banda di
  memoria (≈10 ns per elemento, già misurato), quindi il guadagno atteso lì è
  modesto.
- Soglie: sotto una certa dimensione resta sequenziale, perché il costo di
  avvio dei task supera il guadagno. Le soglie vengono dai benchmark, non a
  occhio.
- Ogni thread chiama `dispatch` al proprio interno, così mantiene il kernel
  AVX2/AVX-512.

### Obiettivo misurabile
Determinante 64×64 e prodotto 64×64 con scalabilità quasi lineare fino a
`min(N, core)` thread; nessuna regressione a thread singolo.

## Fase 3: aritmetica modulo un grande dispari con Montgomery RNS (conclusa)

### Esito (VM con 2 vCPU e AVX-512, un thread, mediane; GMP tramite `rug`)
- API: `MontRns<B1, N1, B2, N2>` (modulo dispari a runtime, in limb da 64
  bit, `MontRnsError` per modulo pari o < 3, primo in comune con una base, basi
  sovrapposte, basi troppo piccole con i bit necessari), `ModElem` (`*`, `+`,
  `-`, negazione, `square`, `pow` a finestra di 4 bit, `inv_prime`, `to_limbs`,
  `to_u128`) e `ModVec` (lo stesso in lotto; `pow` è il `pow_batch` del
  piano). Seconda base disgiunta con `rns_base!(Nome: largest N after K)`.
- Limiti: ogni elemento porta un fattore `f` (valore `< f·m`) invece di
  essere ridotto; il costruttore calcola il fattore massimo `F` per cui ogni
  prodotto entra nelle basi, e le somme normalizzano quando serve, quindi
  nessuna operazione può fallire. Dimostrazioni in DESIGN.md; in debug ogni
  prodotto verifica `r < f_out·m`.
- Misure:

  | | residue | GMP | num-bigint |
  |---|---|---|---|
  | prodotto, 1024 bit | 2,1 µs | 0,59 µs | 1,0 µs |
  | prodotto, 2048 bit | 9,7 µs | 1,9 µs | 3,7 µs |
  | esponenziazione, esponente da 1024 bit | 2,6 ms | 0,42 ms | 0,77 ms |
  | esponenziazione, esponente da 2048 bit | 24,6 ms | 3,6 ms | 6,0 ms |
  | 256 esponenziazioni, esponente da 256 bit, 1024 bit | 46,6 ms | 30,3 ms | 56,0 ms |
  | idem, 2048 bit | 140 ms | 105 ms | 187 ms |

  Come previsto, sul singolo elemento GMP vince di 3,6-6,8 volte. In lotto il
  divario scende a 1,5 volte (1024 bit) e 1,3 volte (2048 bit), e residue
  supera num-bigint; il lotto vale 3,8-5,8 volte rispetto agli stessi calcoli
  fatti un `ModElem` alla volta. L'obiettivo implicito, battere GMP in
  lotto su un core, non è raggiunto. Con `parallel` e 2 thread il lotto da
  2048 bit scende a 110-118 ms, alla pari con GMP su un thread (che però si
  parallelizza altrettanto facilmente).
- Dove va il tempo: le due estensioni di base sono l'80% di un prodotto
  in lotto e girano a circa 0,36 cicli per prodotto di parole, contro 0,125
  teorici. Riduzione vettoriale a fine blocco: 1,4-1,6× sul lotto da 1024 bit
  e 1,3× sul prodotto matriciale 64×64; cicli dei blocchi senza controlli dei
  limiti: un altro 5-10%.
- Test: proprietà contro num-bigint su tre coppie di basi fino a 600 bit
  (moduli vicini al massimo consentito), a ogni livello SIMD e con il percorso
  parallelo forzato; test fissi a 2048 e 4096 bit; inversi modulo primi noti;
  crescita del fattore su 200 raddoppi.
- Non fatto rispetto al piano: costruzione da stringa (solo limb); lotto
  da 1024 elementi (misurato a 256); confronto con `crypto-bigint`; la scelta
  tra base dinamica e menu di basi per Python resta alla Fase 6.
- Non a tempo costante, dichiarato nella documentazione del modulo, nel
  README e nel CHANGELOG.
- Limite scoperto: basi oltre circa 100 primi (4096 bit ne vogliono
  134 + 134) richiedono `#![allow(long_running_const_eval)]` nel crate che le
  dichiara.

- Misura finale per la 2.0.0, mediana di tre sessioni: il lotto da 256
  esponenziazioni richiede 41,8 ms contro 29,1 di GMP e 37,9 di num-bigint a
  1024 bit, e 125 ms contro 94,6 e 158 a 2048 bit. A 1024 bit num-bigint
  risulta quindi più veloce di residue, al contrario di quanto misurato in
  questa fase (56 ms); le tre sessioni finali concordano tra loro.

Il piano originale della fase segue invariato, per riferimento.

### Algoritmo (Bajard, Didier e Kornerup, Kawamura e altri)
Due basi coprime, `B` con prodotto `M` e `B'` con prodotto `M'`, e un modulo
dispari `m` scelto a runtime. Per `a·b mod m`:
1. `c = a·b` in `B ∪ B'`, canale per canale.
2. `q = c·(−m⁻¹) mod M`, in `B`.
3. Estensione di `q` da `B` a `B'`. Può essere inesatta: si usa
   `q̂ = Σ ξ_i·M_i = q + α·M` senza sottrarre `α·M`, e l'errore viene assorbito
   dalle condizioni di range.
4. `r = (c + q̂·m)·M⁻¹` in `B'`.
5. Estensione di `r` da `B'` a `B`, esatta, con il metodo di
   Shenoy e Kumaresan già implementato. Il canale ridondante modulo `2^64` si
   ottiene senza ricostruire: `q̂ mod 2^64 = Σ ξ_i·(M_i mod 2^64)` è calcolabile
   esattamente proprio perché `q̂` è definito come quella somma.

Il risultato sta in `[0, 2m)` (riduzione pigra). Le condizioni sulle basi (del
tipo `M > (n+2)·m` e `M' > 2m`, da dimostrare con esattezza) vanno verificate
dal costruttore e dimostrate in DESIGN.md, come già fatto per la divisione
esatta.

### Onestà sulle prestazioni
Una singola moltiplicazione modulare costa circa `2·n·n'` operazioni di canale.
Per un modulo da 2048 bit con primi da 31 bit sono circa 67 canali per base, cioè
circa 9 000 operazioni, contro circa 2 000 prodotti da 64 bit di GMP. Su una
singola esponenziazione GMP vincerà. Il caso in cui RNS conviene è il lotto:
molte moltiplicazioni modulari indipendenti, in SIMD lungo gli elementi. In
lotto, le due base extension diventano prodotti matrice-matrice
(elementi × `n`) · (`n` × `n'`), e quindi riusano il kernel delle Fasi 1 e 2.
README e benchmark devono dirlo con la stessa chiarezza con cui oggi parlano di
FLINT.

### API proposta
- `MontRns<B, N, B2, N2>`: contesto costruito a runtime da `m` (limb o stringa),
  con controllo che `m` sia dispari, coprimo con `M` e `M'`, e nei limiti.
- `ModElem`: `add`, `sub`, `mul`, `square`, `pow` (finestra scorrevole), `inv`
  (per primi, via Fermat), conversioni in e da binario.
- `ModVec`: lo stesso, in lotto, con kernel SIMD; `pow_batch` come caso
  d'uso principale.
- Avviso esplicito: non a tempo costante. `redc` e diverse funzioni hanno
  rami che dipendono dai dati. Finché non c'è una revisione a tempo costante (una
  variante con maschere, nello stile di `subtle`), la documentazione deve dire di
  non usarlo con dati segreti. È meglio dichiararlo subito che scoprirlo da una
  segnalazione.

### Preparazione al binding Python
Python non ha const generics: il numero di canali deve poter essere scelto a
runtime. Due strade, da decidere qui:
- (a) un tipo di base dinamica nel crate principale (`DynBase`, `Vec` invece di
  array), più flessibile ma un po' più lento;
- (b) il binding istanzia con una macro un menu fisso di basi (per esempio da 2 a
  128 canali) e sceglie quella giusta in base ai bit richiesti. Più veloce, ma il
  binario cresce.

Consiglio (b) per partire, con (a) come evoluzione.

### Benchmark
Lotto di 1024 esponenziazioni modulari con modulo da 1024 e 2048 bit, contro
`rug` (`pow_mod`), `crypto-bigint` e `num-bigint`; più la singola
esponenziazione, per mostrare dove RNS perde.

Fatto quando: test di proprietà contro `num-bigint` (`modpow`) su moduli
casuali da 64 a 4096 bit, incluso il caso limite `m` vicino al massimo consentito;
DESIGN.md con le dimostrazioni delle condizioni di range.

## Fase 4: SIMD portabile su ARM NEON e WASM (conclusa, senza misure su ARM)

### Esito
- Test: l'intera suite passa su aarch64 (sotto qemu: 410 test con tutte
  le feature, 376 con solo `alloc`) e su `wasm32-wasip1` sotto wasmtime, con
  e senza simd128 (377 con tutte le feature, 364 con solo `alloc`, 204 senza
  feature). Su WASM i test di `parallel.rs`, che creano pool di thread, sono
  esclusi.
- Bug trovati, invisibili su x86:
  - con `alloc` il crate non compilava su nessun target a 32 bit
    (`1usize << 32` nel prodotto scalare);
  - con `parallel` su WASM senza thread la prima operazione parallela andava
    in panic dentro rayon (`std` segnala `WouldBlock`, rayon ripiega su un
    thread solo con `Unsupported`); ora il parallelismo si usa solo dove i
    thread si possono creare;
  - le feature predefinite di proptest (`fork`, `timeout`) non compilano per
    WASM: il job WASM della CI non avrebbe mai potuto girare.
- aarch64: NEON è sempre attivo; l'assembly dei tre cicli di base
  (prodotto scalare a metà da 32 bit, riga del prodotto matriciale, prodotto
  di Montgomery elemento per elemento) usa `umull`/`umull2`. Stima statica
  con llvm-mca per prodotto (Neoverse N1 / V1): 1,0 / 0,5 cicli per prodotto
  scalare e Montgomery, 1,6 / 0,5 per la riga matriciale, che tiene gli
  accumulatori in memoria. Nessuna misura reale: qui non c'era hardware
  ARM. Il kernel NEON a blocchi nei registri (come quello x86) non è stato
  scritto, perché non si poteva misurare.
- WASM: con simd128 LLVM usava `i64x2.mul` (moltiplicazione 64×64 che
  né SSE né NEON hanno) e il prodotto elemento per elemento diventava più
  lento della build scalare. Nuovi kernel in `src/batch/wasm.rs` con
  `i64x2.extmul_*_i32x4_u`, resi stabili con `black_box` perché LLVM li
  riscriveva: prodotto elemento per elemento 2,5× più veloce, prodotto
  scalare, matriciale e `ModVec` 11-14%. Ora simd128 vale 1,1-1,9× sulla
  build scalare (1,2-1,8× nella misura finale per la 2.0.0, tranne la conversione da `i128`); il codice nativo x86 resta 1,1-5,5× più veloce di WASM con simd128.
- Misure e CI: `examples/timing.rs` (senza criterion, gira anche in
  wasmtime); job CI `timing` su x86-64, ARM Linux, Apple Silicon e WASM, per
  avere i numeri ARM reali al primo push. README con le istruzioni per
  compilare con simd128.
- Non fatto: misure su ARM reale; kernel NEON espliciti; SVE2; benchmark
  in un browser.

Il piano originale della fase segue invariato, per riferimento.

L'approccio resta quello attuale: cicli senza rami che LLVM vettorizza, senza
`unsafe`. Il lavoro è soprattutto verifica.

- aarch64: NEON fa parte dell'architettura base, quindi i kernel sono già
  compilati per NEON. Bisogna verificare nell'assembly che LLVM usi davvero
  `umull`/`umlal` (32×32→64) e non cicli scalari, e correggere la forma dei cicli
  dove non succede. Verificare anche se `pulp` offre un dispatch utile su
  aarch64; SVE2 è un passo successivo, non prioritario.
- WASM: `simd128` ha `i64x2.extmul_low_i32x4_u`, adatto al prodotto
  `32×32→64`. In WASM però non esiste il rilevamento a runtime: SIMD si
  attiva solo in compilazione (`-C target-feature=+simd128`). Il README deve
  spiegare come compilarlo, e la CI deve provare entrambe le build.
- Benchmark su una macchina ARM reale (Graviton o Apple Silicon) e in un
  runtime WASM, pubblicati nel README accanto a quelli x86.

Fatto quando: stessa suite di test verde su x86, aarch64 e wasm32;
assembly controllato per i kernel principali su ogni architettura.

## Fase 5: primi da circa 62 bit (respinta in Fase 0)

L'esperimento in `experiments/wide-primes` ha confrontato i 31 bit attuali con
62 bit (Montgomery su `u128`) e 50 bit in `f64` con FMA, sul prodotto 64×64 e
sul determinante 64×64. Rispetto al miglior ciclo a 31 bit, i 62 bit vanno a
0,68× e 0,76× con AVX-512, a 0,60× e 0,70× con AVX2; i 50 bit pareggiano solo
sul determinante con AVX-512. Il criterio era ≥1,3× su entrambi: non
raggiunto. I moduli restano sotto `2^31` e nessuna versione breaking è
necessaria per questo motivo. Il guadagno sulle matrici va cercato nei kernel
(Fase 2).

## Fase 6: binding Python in un pacchetto separato (conclusa, non pubblicata)

### Esito
- Struttura: crate `python/` (`residue-py`) in un workspace Cargo, con
  PyO3 0.30 e maturin; pacchetto PyPI `residue-rs` (il nome `residue` è già
  preso), importato come `residue_rs`. Una wheel abi3 per piattaforma copre
  Python 3.11 e successivi (3.11 è la prima API stabile con il buffer
  protocol, usato per gli array NumPy int64). Il crate principale non è
  cambiato.
- Decisione rinviata dalla Fase 3: menu di basi (b). 14 basi per gli
  interi (da 2 a 64 primi, da 60 a 1982 bit), scelte con `Base(bits=…)`, e 8
  coppie per i moduli (da 5 a 133 primi, fino a 4096 bit). Costo: estensione
  da circa 14 MB, wheel da 5 MB.
- API: `Base`, `Vector` (`+ - *`, `dot`, `sum`, indici, `tolist`,
  `to_numpy`), `Matrix` (`@ + -`, `.T`, `det`, `tolist`, `to_numpy`),
  `Modulus` con `pow_batch` e `mul_batch`. Gli elementi `ModElem`/`ModVec`
  non sono esposti come oggetti, perché prendono in prestito il contesto: i
  lotti si convertono e si calcolano in una chiamata. Le operazioni lunghe
  rilasciano il GIL. Stub di tipo `.pyi` inclusi.
- Conversione degli interi Python: con PyO3 sotto ABI stabile costava
  466 ns per valore da 100 bit, più dell'aritmetica; letta nella forma più
  stretta (`i64`, `i128`, poi `int.to_bytes`) scende a 93 ns.
- Misure (python/README.md, misura finale per la 2.0.0): su dati già
  convertiti, prodotto scalare 7,5× più veloce di python-flint, prodotto
  matriciale 2,3×, determinante 1,2×; conversione inclusa, 3,1× sul prodotto
  scalare e pari a python-flint su matrici e determinanti. Esponenziazione
  modulare in lotto: GMP (gmpy2 o python-flint) resta 1,5-2× più veloce;
  residue è 5-7× più veloce del `pow` di Python.
- Test: pytest e hypothesis contro gli interi di Python (vettori su 6
  basi, matrici, determinanti contro l'eliminazione razionale, moduli da 2 a
  4096 bit, thread concorrenti). Wheel e sdist installate e provate in
  ambienti puliti, anche senza NumPy.
- CI: test su Linux x86-64 e arm64, macOS e Windows con Python 3.11 e
  3.13; wheel per Linux, macOS e Windows; pubblicazione su PyPI a ogni tag
  `py-v*` con trusted publishing, da configurare su PyPI prima del primo
  tag. Non verificata (nessun push).
- Non fatto: pubblicazione; wheel Windows arm64; oggetti modulari
  persistenti (`ModVector`) in Python; base dinamica (a).

Il piano originale della fase segue invariato, per riferimento.

- Crate `residue-py` nello stesso repository (workspace Cargo), con PyO3 e
  maturin. Il crate principale resta senza `unsafe`; il binding vive a parte.
- Wheel per Linux, macOS e Windows, x86-64 e arm64, con ABI stabile
  (`abi3`), così una sola wheel copre tutte le versioni di Python supportate.
  Build e pubblicazione automatizzate in CI.
- API Python: interi Python di precisione arbitraria in ingresso e in
  uscita; array NumPy `int64` per i lotti; `RnsMatrix` con `det()` e `@`;
  `MontRns` con `pow_batch`. La base si sceglie con un parametro (`bits=…`),
  secondo la decisione rinviata dalla Fase 3 (base dinamica o menu di basi).
- Confronti con `gmpy2` e `python-flint`, che è il riferimento naturale
  per chi usa Python.
- Nome su PyPI da verificare prima (`residue` potrebbe essere già preso).

## Lavoro trasversale, in ogni fase

- Ogni ottimizzazione entra solo con un benchmark che la giustifica e con i
  test di proprietà contro `num-bigint`.
- DESIGN.md registra le decisioni, comprese quelle sbagliate e corrette, come
  già fa oggi.
- README: aggiornare tabella e grafici solo da una singola sessione di misura,
  come da regola attuale.
- Fuzzing con `cargo-fuzz` sui parser e sulla divisione, prima della 1.0.
- MSRV dichiarata e verificata in CI.

## Riepilogo delle versioni

| Versione | Contenuto |
|---|---|
| 0.1.0 | Stato di partenza |
| 2.0.0 (crate) | Fasi 0, 1, 2, 3 e 4: conversioni in blocco, kernel espliciti x86, feature `parallel`, Montgomery RNS, aarch64 e WASM verificati, kernel simd128 |
| 2.0.0 (PyPI, `residue-rs`) | Fase 6: binding Python |
| nessuna | Fase 5: respinta (primi da circa 62 bit più lenti di quelli da 31 bit) |
