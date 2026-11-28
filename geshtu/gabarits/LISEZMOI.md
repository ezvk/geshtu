# Fichiers qui remplacent ceux d'un modèle servi par geshtu

Référencés dans la config par `geshtu:<nom>` (`fichiers` d'un moteur, voir geshtu/serveur.py).
Ils viennent de horde (`hosts/utu/traduction.nix`, `hosts/utu/gemma4-chat_template.jinja`),
où ils ont été mesurés ; ils vivent ici parce que c'est traduction.py qui en dépend.

- `translategemma.jinja` : gabarit « passthrough ». geshtu envoie déjà le tour d'entraînement
  de TranslateGemma tel quel (traduction.py, `_translategemma`) ; le gabarit officiel exige
  des messages structurés que l'API de conversation ne transmet pas.
- `translategemma-generation.json` : `eos_token_id` [1, 106] (<eos>, <end_of_turn>) ; `stop`
  ne suffit pas, le jeton spécial est retiré du texte.
- `gemma4.jinja` : Gemma 4 E4B, système + tours et rien d'autre. Le gabarit officiel fait
  échouer minja (« Chat template is not compatible with minja … Disabling /chat/completions »).
