# DJ Set Energy Arc Modeller

In my opinion, what makes a DJ set good is how it carries energy, with smooth transitions that keep the crowd moving. This sense of momentum building and easing across a set is what I think of as its energy arc, and this project sets out to measure it directly from the audio, and then teach a model to anticipate where a set is going next.

## Why I built this

I love music, dance music in particular, thus I wanted my next project to be
about something I genuinely enjoy rather than another generic dataset. On top of
that, audio is something almost no student data science portfolio goes near, so
it felt like a chance to stand out while learning more about digital
signal processing, which I had never properly touched before.

The whole pipeline runs straight off the audio using `librosa`. Spotify used to expose features like energy, tempo and key through its
API, but it closed that to new apps in late 2024. As it happens, all of those
qualities can be measured from the waveform itself. Working them out from scratch is exactly what I do here.

## Where I want to take this

My longer-term aim is for the model to work across the all genres of EDM,
from house through to hardstyle and everything in between.
Different genres carry their energy in different ways, so getting
it to generalise is a big part of what makes the problem interesting.

Beyond the analysis itself, I would love to turn this into something you could
use while listening, for example a SoundCloud extension that marks out the
high-energy moments of a set as you scrub through it. Platforms already do
something loosely similar with their waveform displays, but the goal here is to
read the energy arc properly rather than just show how loud the audio is.

## What it does

I take a handful of public DJ sets and turn each one into a story about its
energy over time.

| Step | What I do |
|------|-----------|
| **Acquire** | Download public sets I list in `configs/sets.csv` |
| **Extract** | Cut each set into 30-second windows and pull features from each one: BPM, energy, brightness, key, and a danceability proxy |
| **Build the arc** | Join the windows into one energy time series per set |
| **Model the arc** | Train an LSTM to predict the next window's energy from the last few, so it learns how sets rise and fall |
| **Cluster vibes** | Group windows into "vibe profiles" (e.g. dark techno vs melodic house) from the features alone |
| **Recommend** | Given the last few windows and a target energy, suggest what kind of vibe should come next |
| **Explore** | Wrap it all in a Streamlit app |

## A note on the music

I only point `configs/sets.csv` at sets that are already public, and the audio
never leaves my machine or goes into the repo. The only things I commit are the
small feature tables and figures I make from them, not anyone's actual music.

## Pipeline

| Stage | Module | Status |
|-------|--------|--------|
| 0 Acquire | `src/acquire.py` | done |
| 1 Extract | `src/extract_features.py` | planned |
| 2 Arc | `src/build_arc.py` | planned |
| 3 Model | `src/train_lstm.py` | planned |
| 4 Cluster | `src/cluster_vibes.py` | planned |
| 5 Recommend | `src/recommend.py` | planned |
| 6 App | `src/app.py` | planned |

## Layout

```
configs/      config.yaml (all the settings) + sets.csv (the list of sets)
src/          one file per pipeline stage
data/raw/     downloaded audio (gitignored, re-downloadable)
data/interim/ the per-window feature tables
figures/      charts for this README
outputs/      models, cluster labels, app bits
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg            # yt-dlp and librosa need it to read audio

# 1. List the sets I want in configs/sets.csv (id, artist, event, url)
# 2. Download them:
python -m src.acquire
```
