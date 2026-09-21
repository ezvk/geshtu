"""Diarisation on the accelerator: pyannote segmentation, wespeaker prints.

Who spoke when, computed here rather than delegated, because the tool that
does it for free could only use the CPU.

⚠️ THE NPU IS THE FASTEST OF THE THREE, MEASURED -- which was not the
expectation. One 10-second window through segmentation, and one voice print:

                    NPU        GPU        CPU
    segmentation    16.2 ms    106.8 ms   44.9 ms
    voice print      2.2 ms      4.2 ms   19.8 ms

The GPU is the worst on segmentation: the model is small and launch overhead
dominates. End to end on a 30-minute meeting, against sherpa-onnx on four CPU
threads:

    sherpa-onnx, CPU, 4 threads    128 s
    this, on the NPU                 6.5 s

⚠️ AND IT IS SILENT. ezvk asked for an alternative because the fan disagreed
with both the CPU and the GPU. That is the whole reason an NPU exists.

⚠️ FEATURES COME FROM kaldi-native-fbank, the very library sherpa-onnx uses
internally -- deliberately, because reimplementing Kaldi filterbanks is where
this port would have failed silently: wrong features give plausible prints and
wrong speakers, with no error anywhere.

⚠️ EVERY SHAPE IS FIXED, which is what the NPU requires and what the
embeddings model of ./recherche.nix lacked. The segmentation model is natively
[1,1,160000]; the print model is [?,?,80] and is reshaped to a fixed window,
excerpts shorter than that being repeated rather than zero-padded -- silence
is not neutral to a voice print.
"""
from __future__ import annotations

import os
import time
import wave

import kaldi_native_fbank as knf
import numpy as np

FENETRE = 160000          # 10 s a 16 kHz, forme figee du modele
PAS = 80000               # 5 s : recouvrement de moitie
TRAME_EMB = 300           # 3 s de banc de filtres, forme figee pour le NPU
MIN_ON = 0.3
MIN_OFF = 0.5

# ⚠️ L ORDRE DES CLASSES POWERSET N EST PAS DEVINABLE, il vient de pyannote :
# 7 classes = le silence, trois locuteurs seuls, trois paires.
POWERSET = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]


def lit_wav(chemin):
    with wave.open(chemin, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        brut = w.readframes(w.getnframes())
    return np.frombuffer(brut, dtype=np.int16).astype(np.float32) / 32768.0


def fbank(audio):
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = 80
    f = knf.OnlineFbank(opts)
    f.accept_waveform(16000, (audio * 32768).tolist())
    f.input_finished()
    return np.array([f.get_frame(i) for i in range(f.num_frames_ready)],
                    dtype=np.float32)


def segmente(compile_seg, audio):
    """Rend, par fenetre, l activite [trames, 3] de trois locuteurs locaux."""
    sortie = []
    pos = 0
    while pos < len(audio):
        bout = audio[pos:pos + FENETRE]
        if len(bout) < FENETRE:
            bout = np.pad(bout, (0, FENETRE - len(bout)))
        x = bout.reshape(1, 1, FENETRE).astype(np.float32)
        y = compile_seg(x)[compile_seg.output(0)][0]        # [589, 7]
        actif = np.zeros((y.shape[0], 3), dtype=np.float32)
        for t, classe in enumerate(np.argmax(y, axis=1)):
            for qui in POWERSET[classe]:
                actif[t, qui] = 1.0
        sortie.append((pos, actif))
        pos += PAS
    return sortie


def empreinte(compile_emb, audio, debut, fin):
    """Une empreinte pour un extrait. Forme FIGEE : le NPU refuse le reste."""
    bout = audio[int(debut * 16000):int(fin * 16000)]
    if len(bout) < 16000:                                  # moins d une seconde
        return None
    feats = fbank(bout)
    if len(feats) == 0:
        return None
    if len(feats) < TRAME_EMB:                             # on repete plutot
        rep = int(np.ceil(TRAME_EMB / len(feats)))         # que de remplir de zeros
        feats = np.tile(feats, (rep, 1))
    feats = feats[:TRAME_EMB]
    feats = feats - feats.mean(axis=0, keepdims=True)      # CMN, comme wespeaker
    x = feats.reshape(1, TRAME_EMB, 80).astype(np.float32)
    v = compile_emb(x)[compile_emb.output(0)][0]
    n = np.linalg.norm(v)
    return v / n if n else None


def regroupe(vecteurs, seuil):
    """Regroupement agglomeratif, distance cosinus, liaison moyenne.

    ⚠️ VECTORISE, ET CE N EST PAS DE LA COQUETTERIE. La premiere version
    recalculait la distance moyenne entre chaque paire de groupes a chaque
    fusion, en boucles Python : 32 s sur une reunion de 30 min, contre 6 s
    d inference. Le regroupement coutait cinq fois le calcul qu il sert.

    On tient a la place une matrice de distances entre groupes, mise a jour
    par la formule de Lance-Williams : la distance moyenne d un groupe fusionne
    est la moyenne ponderee des deux, ce qui evite de revisiter les membres.
    """
    V = np.stack(vecteurs)
    n = len(V)
    D = 1.0 - V @ V.T
    np.fill_diagonal(D, np.inf)
    taille = np.ones(n)
    vivant = np.ones(n, dtype=bool)
    membres = {i: [i] for i in range(n)}

    while vivant.sum() > 1:
        sous = np.where(vivant)[0]
        bloc = D[np.ix_(sous, sous)]
        k = np.argmin(bloc)
        a, b = sous[k // len(sous)], sous[k % len(sous)]
        if D[a, b] > seuil:
            break
        # Lance-Williams pour la liaison moyenne
        na, nb = taille[a], taille[b]
        D[a, :] = (na * D[a, :] + nb * D[b, :]) / (na + nb)
        D[:, a] = D[a, :]
        D[a, a] = np.inf
        taille[a] = na + nb
        membres[a] = membres[a] + membres[b]
        vivant[b] = False
        D[b, :] = np.inf
        D[:, b] = np.inf

    etiquette = {}
    for numero, tete in enumerate(np.where(vivant)[0]):
        for i in membres[tete]:
            etiquette[i] = numero
    return etiquette



def analyse(wav, modeles, seuil=0.5, device="NPU", trace=print):
    """Rend une liste de (debut, fin, numero de locuteur)."""
    import openvino as ov

    core = ov.Core()
    if device not in core.available_devices:
        # ⚠️ LE NPU DISPARAIT EN SILENCE sans level-zero sur LD_LIBRARY_PATH :
        # le greffon se charge par dlopen et OpenVINO le retire sans rien dire.
        # L empaquetage pose la variable ; le dire ici evite une heure de
        # recherche le jour ou elle manque.
        trace("device %s unavailable (have %s) -- falling back to CPU"
              % (device, ", ".join(core.available_devices)))
        device = "CPU"

    seg = core.read_model(os.path.join(modeles, "segmentation.xml"))
    seg.reshape((1, 1, FENETRE))
    emb = core.read_model(os.path.join(modeles, "embedding.xml"))
    emb.reshape((1, TRAME_EMB, 80))
    cseg = core.compile_model(seg, device)
    cemb = core.compile_model(emb, device)

    audio = lit_wav(str(wav))
    duree = len(audio) / 16000
    depart = time.time()
    fenetres = segmente(cseg, audio)
    pas_trame = 10.0 / fenetres[0][1].shape[0]

    extraits = []
    for pos, actif in fenetres:
        base = pos / 16000
        for qui in range(3):
            on = actif[:, qui] > 0.5
            if on.sum() * pas_trame < 1.0:
                continue
            idx = np.where(on)[0]
            extraits.append((base + idx[0] * pas_trame,
                             base + (idx[-1] + 1) * pas_trame))

    vecteurs, gardes = [], []
    for debut, fin in extraits:
        v = empreinte(cemb, audio, debut, min(fin, duree))
        if v is not None:
            vecteurs.append(v)
            gardes.append((debut, fin))
    if not vecteurs:
        return []
    etiquette = regroupe(vecteurs, seuil)
    trace("%d excerpts, %d speakers, %.1f s on %s"
          % (len(vecteurs), max(etiquette.values()) + 1,
             time.time() - depart, device))

    tours = sorted((d, f, etiquette[i]) for i, (d, f) in enumerate(gardes))
    fusion = []
    for debut, fin, qui in tours:
        if fusion and fusion[-1][2] == qui and debut - fusion[-1][1] < MIN_OFF:
            fusion[-1] = (fusion[-1][0], max(fusion[-1][1], fin), qui)
        else:
            fusion.append((debut, fin, qui))
    return [t for t in fusion if t[1] - t[0] >= MIN_ON]
