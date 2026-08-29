# whisper-writer-gpu-setup

Fork de [savbell/whisper-writer](https://github.com/savbell/whisper-writer) qui déporte la
transcription sur le **GPU**, avec l'installation complète et reproductible qui va avec.

Testé sur **EndeavourOS / Arch**, Wayland (Hyprland), **AMD Radeon 890M** (Strix Point,
RDNA 3.5) — mais rien ici n'est spécifique à cette carte : tout GPU géré par Vulkan
convient, y compris Intel et NVIDIA.

## Pourquoi ce fork

WhisperWriter transcrit avec **faster-whisper**, dont le moteur **CTranslate2 n'a que CUDA
comme backend GPU**. Sur une carte AMD ou Intel, aucun réglage de `device:` ou
`compute_type:` ne peut atteindre le GPU — la transcription reste sur le CPU. Vérifiable :

```python
import ctranslate2
ctranslate2.get_cuda_device_count()              # 0
ctranslate2.get_supported_compute_types('cpu')   # {'float32', 'int8', 'int8_float32'}
```

La solution retenue n'est pas de patcher faster-whisper, mais de **changer de moteur** :
`whisper.cpp` avec son backend Vulkan, lancé en serveur résident. WhisperWriter possède
déjà un chemin client OpenAI (`transcribe_api`), et `whisper-server` sait exposer une route
compatible — la bascule tient donc dans la configuration.

```
WhisperWriter ──HTTP──> whisper-server (whisper.cpp + Vulkan) ──> GPU
   config.yaml            systemd --user, 127.0.0.1:8089
```

## Gains mesurés

Modèle `medium` quantifié q5_0, mesures prises sur le chemin `transcribe()` réel,
meilleur de 3 exécutions.

| Audio | CPU float32 (défaut amont) | CPU int8 | **GPU Vulkan + VAD** |
|---|---|---|---|
| 3,97 s (dictée courte) | — | 3,94 s | **0,96 s** |
| 14,94 s | 11,42 s | 5,85 s | **1,81 s** |
| silence | — | — | **0,01 s** |

Soit **×4,1** sur une dictée courte par rapport au CPU déjà optimisé en int8. La colonne GPU
correspond exactement à la configuration livrée ici (`medium` q5_0, `--vad`, port 8089).

Ces chiffres ont été relevés machine au repos. Sur un GPU **intégré**, la mémoire et les
unités de calcul sont partagées avec l'affichage : la même dictée mesurée avec un navigateur
et un IDE actifs monte à ~2,3 s. Le gain sur le CPU reste net dans les deux cas, mais il faut
s'attendre à cette variabilité — elle n'existerait pas sur une carte dédiée.

Un repère utile : `whisper.cpp` **en CPU** met 6,96 s là où CTranslate2 int8 en met 3,94.
Le gain vient donc bien du GPU, pas du changement de moteur — comparer au CPU de
whisper.cpp gonflerait artificiellement le résultat.

## Installation

### 1. Paquets système

```bash
sudo pacman -S --needed whisper-cpp ggml-vulkan vulkan-radeon
sudo pacman -S --needed xdotool wtype dotool ydotool   # backends de frappe
```

`ggml-vulkan` *dépend* de `ggml` au lieu d'entrer en conflit avec lui : c'est un backend
chargé au runtime, aucune recompilation n'est nécessaire. Remplacer `vulkan-radeon` par
`vulkan-intel` ou `nvidia-utils` selon la carte.

Vérifier que la carte est vue : `vulkaninfo --summary | grep deviceName`

### 2. Modèles

```bash
mkdir -p ~/.local/share/whisper-cpp && cd ~/.local/share/whisper-cpp

# modèle de transcription (514 Mo)
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin

# modèle VAD, indispensable contre les hallucinations (0,8 Mo)
curl -LO https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin
```

Autres modèles sur [huggingface.co/ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main).
`ggml-large-v3-turbo-q5_0.bin` (547 Mo) a été testé : plus précis pour une vitesse
équivalente, c'est une bonne alternative.

**Les modèles faster-whisper déjà en cache ne servent à rien ici** — ils sont au format
CTranslate2, il faut du GGML.

### 3. Application

```bash
git clone https://github.com/blyscop/whisper-writer-gpu-setup.git ~/whisper-writer
cd ~/whisper-writer
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp gpu-setup/config.yaml src/config.yaml   # src/config.yaml est gitignoré
cp gpu-setup/env.example .env
```

> **`requirements.txt` a été refait.** Celui hérité de l'amont était encodé en UTF-16 et
> épinglait `ctranslate2==4.2.1` / `faster-whisper==1.0.2`, qui ne se résolvent plus sur les
> Python récents : un clone neuf échouait dès le premier `pip install`. Il est conservé pour
> mémoire sous `gpu-setup/requirements.upstream-utf16.txt`.
>
> Deux dépendances méritent une mention, parce qu'elles sont invisibles à la lecture du code :
>
> - **`webrtcvad-wheels`**, et non `webrtcvad`. Le module importé s'appelle bien `webrtcvad`,
>   mais le paquet qui le fournit avec des wheels précompilées est `webrtcvad-wheels`.
>   `pip install webrtcvad` tente une compilation qui échoue sur Python 3.14.
> - **`evdev`**, importé dynamiquement par `key_listener.py` et absent de tout import
>   statique. La configuration livrée utilise `input_backend: evdev` : sans lui, le
>   raccourci clavier ne répond pas.
>
> [`gpu-setup/requirements-frozen.txt`](gpu-setup/requirements-frozen.txt) donne le relevé
> complet de l'environnement vérifié, sous Python 3.14.7.
>
> `faster-whisper` et `ctranslate2` restent listés bien que ce fork ne les charge jamais
> (`main.py` saute `create_local_model()` quand `use_api: true`) : ils servent au repli CPU
> (`use_api: false`). Qui n'en veut pas peut retirer les deux dernières lignes et économiser
> le téléchargement.

### 4. Serveur en service utilisateur

```bash
cp gpu-setup/systemd/whisper-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now whisper-server.service
curl -s http://127.0.0.1:8089/v1/health     # {"status":"ok"}
```

Confirmer que le GPU est bien pris :

```bash
journalctl --user -u whisper-server | grep -i "using Vulkan"
```

### 5. Lancement

```bash
./start.sh
```

Pour un lanceur, copier `gpu-setup/whisper-writer.desktop` dans
`~/.local/share/applications/` **en remplaçant le chemin par un chemin absolu** : un
fichier `.desktop` n'interprète pas `$HOME` dans `Exec`.

## Les trois pièges, et pourquoi ces réglages

Ces trois points sont la vraie valeur de ce dépôt. Sans eux le montage semble marcher,
puis produit du texte corrompu.

### `-ml 100000` est obligatoire

`whisper-server` force `max_len = 60` quand on ne précise rien, et découpe **au milieu des
mots** (`rel\nance`) puisque `split_on_word` est faux par défaut :

```cpp
// examples/server/server.cpp, v1.9.1
wparams.max_len = params.max_len == 0 ? 60 : params.max_len;
```

Envoyer `max_len=0` dans la requête n'y change rien — `0` est justement la valeur qui
déclenche le 60. Il faut une grande valeur au démarrage du serveur.

### `--vad` contre les hallucinations

Sur un blanc, Whisper invente des génériques de sous-titrage, qui seraient **tapés au
clavier**. Résultats sur du silence numérique :

| Configuration | Sortie |
|---|---|
| `medium` | `[Sous-titres réalisés par la communauté d'Amara.org]` |
| `medium -sns` | `...` (insuffisant) |
| `large-v3-turbo -sns` | `Sous-titrage Société Radio-Canada` (**aucun effet**) |
| **`--vad -vm ggml-silero…`** | *(vide)* |

`-sns` (*suppress non-speech tokens*) ne suffit pas. Le VAD est la seule option qui rende
une chaîne vide — et il court-circuite l'inférence, d'où les 0,01 s.

Vérifié : le VAD ne mange pas les dictées courtes (`Oui.` à 0,63 s passe intact).

À noter, ce n'est pas un défaut introduit par ce fork : faster-whisper hallucinait déjà
(`Sous-titrage ST'501`) avec `vad_filter: false`.

### Le correctif `merge_segment_line_breaks`

`whisper-server` joint ses segments par `\n`, là où faster-whisper les concaténait sans
séparateur, et `post_process_transcription` ne retirait que le dernier. Ces `\n` internes
partent au clavier :

- backend `dotool` : `_typewrite_dotool` écrit `f"type {text}\n"` sur stdin, donc un saut
  de ligne tronque la commande et la fin du texte est perdue ;
- `pynput` / `wtype` : c'est une touche Entrée parasite, qui valide un formulaire ou envoie
  un message à moitié écrit.

D'où le commit `fix(transcription): join server segments`.

## Contenu de `gpu-setup/`

| Fichier | Rôle |
|---|---|
| `config.yaml` | à copier en `src/config.yaml` (gitignoré en amont) |
| `env.example` | à copier en `.env` |
| `systemd/whisper-server.service` | le service, avec tous les flags qui vont bien |
| `scripts/serveur.sh` | lancement manuel sur le port 8090, pour expérimenter sans toucher au service |
| `scripts/dictee.sh` | enregistre au micro et interroge le service, affiche texte et latence |
| `requirements-frozen.txt` | relevé complet de l'environnement vérifié (Python 3.14.7) |
| `requirements.upstream-utf16.txt` | l'ancien fichier UTF-16 de l'amont, conservé pour mémoire |
| `whisper-writer.desktop` | lanceur (chemin absolu à renseigner) |

`OPENAI_API_KEY` vaut `'local'` dans `env.example` : une valeur factice, mais **non vide**.
`whisper-server` ignore le bearer, seul le SDK y regarde — et à partir d'`openai` 3.x une
clé vide fait lever `OpenAIError: Missing credentials` dès la première dictée. La version
2.x l'acceptait, d'où un piège qui ne se voit que sur une installation neuve.

## Revenir au CPU

```bash
systemctl --user disable --now whisper-server.service
sed -i 's/^  use_api: true/  use_api: false/' src/config.yaml
```

`compute_type: int8` reste en place comme repli : c'est déjà ×1,9 par rapport au `float32`
du défaut amont, sans aucun GPU.

## Licence et attribution

GPLv3, comme le projet amont. Ce dépôt est un fork de
**[savbell/whisper-writer](https://github.com/savbell/whisper-writer)** ; le README
d'origine est conservé sous [`README.upstream.md`](README.upstream.md).

Modifications par rapport à l'amont :

- `feat(input)` — backends de frappe adaptés à Hyprland/XWayland (`wtype`, `xdotool`, et un
  mode `auto` qui choisit selon la fenêtre active) ;
- `fix(transcription)` — jonction des segments renvoyés par le serveur ;
- `gpu-setup/` — configuration et service, ajoutés par ce fork.
